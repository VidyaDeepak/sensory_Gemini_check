"""
pc_perceptual: PCA perceptual map queries.
  - "Show the perceptual map for <category>"
  - "Where does <product> sit on the perceptual map?"
  - "Which products are positioned closest together?" / "most different?"
  - "Nearest neighbours of <product>"
  - "Show product clusters"
  - "Identify whitespace opportunities"
  - "Which benchmark is most similar to <product>?" / "Compare <A> and <B>"
Source: pca_coordinate.csv (record_type = pca_coordinate)
"""
from __future__ import annotations

import itertools
import os
import threading
from typing import Optional

import numpy as np
import pandas as pd

from .utils import best_match, is_compare_query, split_compare_halves

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "pca_coordinate.csv")

_df: Optional[pd.DataFrame] = None
_categories: list[str] = []
_products: list[str] = []
_lock = threading.Lock()

MAX_POINTS = 60

# The source PCA notebook's own validation plots give these axes real
# sensory meaning (fixed axis ranges, labeled "Thin"/"Thick"/"Smooth"/
# "Drag" at the plot extremes) rather than leaving them as opaque
# component numbers — carrying that through here so the chart is
# interpretable instead of just showing "PC1"/"PC2".
X_LABEL = "PC1 (Thin \u2194 Thick)"
Y_LABEL = "PC2 (Smooth \u2194 Drag)"

NEIGHBOR_KEYWORDS = ["nearest neighbour", "nearest neighbor", "most similar", "closest to", "similar to"]
CLOSEST_PAIR_KEYWORDS = ["closest together", "positioned closest"]
DIFFERENT_KEYWORDS = ["most different", "least similar", "furthest apart", "farthest apart"]
CLUSTER_KEYWORDS = ["cluster"]
WHITESPACE_KEYWORDS = ["whitespace", "white space", "gap in the market", "opportunity"]


def _load():
    global _df, _categories, _products
    if _df is None:
        with _lock:
            if _df is None:
                df = pd.read_csv(DATA_PATH)
                _categories = sorted(df["benefit"].unique().tolist())
                _products = sorted(df["product_name"].unique().tolist())
                _df = df
    return _df


def preload():
    """Eagerly load data into memory to prevent cold-start latency."""
    _load()


def categories() -> list[str]:
    _load()
    return _categories


def products() -> list[str]:
    _load()
    return _products


def _category_of(product: str) -> Optional[str]:
    df = _load()
    row = df[df["product_name"] == product]
    return None if row.empty else row.iloc[0]["benefit"]


def _points_for(category: str, cap: int = MAX_POINTS, must_include: Optional[str] = None) -> pd.DataFrame:
    df = _load()
    subset = df[df["benefit"] == category]
    if len(subset) > cap:
        subset = subset.assign(mag=(subset["pc1"] ** 2 + subset["pc2"] ** 2) ** 0.5).sort_values("mag", ascending=False)
        if must_include and must_include in subset["product_name"].values:
            kept = subset.head(cap)
            if must_include not in kept["product_name"].values:
                extra = subset[subset["product_name"] == must_include]
                kept = pd.concat([kept.iloc[: cap - 1], extra])
            subset = kept
        else:
            subset = subset.head(cap)
    return subset


def _default_map(category: str, highlight_product: Optional[str] = None) -> dict:
    subset = _points_for(category, must_include=highlight_product)
    if subset.empty:
        return {"ok": False, "module": "pc_perceptual", "error": "no_data", "category": category}
    points = [
        {"x": round(r.pc1, 2), "y": round(r.pc2, 2), "label": r.product_name, "highlight": (r.product_name == highlight_product)}
        for r in subset.itertuples()
    ]
    table = [{"Product": p["label"], "PC1": p["x"], "PC2": p["y"]} for p in points]
    if highlight_product:
        hp = next(p for p in points if p["highlight"])
        summary = f"'{highlight_product}' sits at (PC1={hp['x']}, PC2={hp['y']}) on the '{category}' perceptual map, shown among {len(points)} products in that category."
    else:
        summary = f"Perceptual map for '{category}': {len(points)} products plotted by PC1 (x-axis) and PC2 (y-axis)."
    return {
        "ok": True, "module": "pc_perceptual", "title": f"Perceptual map: {category}",
        "category": category, "product": highlight_product,
        "chart": {"type": "scatter", "points": points, "x_label": X_LABEL, "y_label": Y_LABEL},
        "table": table, "summary": summary,
    }


def _nearest_neighbors(product: str, k: int = 5) -> dict:
    category = _category_of(product)
    if not category:
        return {"ok": False, "module": "pc_perceptual", "error": "no_data"}
    subset = _points_for(category, cap=200)
    target = subset[subset["product_name"] == product].iloc[0]
    others = subset[subset["product_name"] != product].copy()
    others["distance"] = np.sqrt((others["pc1"] - target["pc1"]) ** 2 + (others["pc2"] - target["pc2"]) ** 2)
    nearest = others.sort_values("distance").head(k)

    points = [{"x": round(target["pc1"], 2), "y": round(target["pc2"], 2), "label": product, "highlight": True}]
    points += [{"x": round(r.pc1, 2), "y": round(r.pc2, 2), "label": r.product_name, "highlight": True} for r in nearest.itertuples()]
    table = [{"Product": r.product_name, "Distance": round(r.distance, 2)} for r in nearest.itertuples()]
    lead = nearest.iloc[0]
    summary = f"The nearest neighbour of '{product}' on the '{category}' perceptual map is '{lead['product_name']}' (distance {round(lead['distance'],2)})."
    return {
        "ok": True, "module": "pc_perceptual", "title": f"Nearest neighbours of {product}",
        "category": category, "product": product,
        "chart": {"type": "scatter", "points": points, "x_label": X_LABEL, "y_label": Y_LABEL},
        "table": table, "summary": summary,
    }


def _closest_pair(category: str) -> dict:
    subset = _points_for(category, cap=80)
    if len(subset) < 2:
        return {"ok": False, "module": "pc_perceptual", "error": "not_enough_products", "category": category}

    best_pair, best_dist = None, float("inf")
    rows = list(subset.itertuples())
    for a, b in itertools.combinations(rows, 2):
        d = ((a.pc1 - b.pc1) ** 2 + (a.pc2 - b.pc2) ** 2) ** 0.5
        if d < best_dist:
            best_dist, best_pair = d, (a, b)

    a, b = best_pair
    points = [
        {"x": round(r.pc1, 2), "y": round(r.pc2, 2), "label": r.product_name, "highlight": (r.product_name in (a.product_name, b.product_name))}
        for r in rows
    ]
    table = [{"Product": a.product_name, "vs": b.product_name, "Distance": round(best_dist, 2)}]
    summary = f"The two most closely positioned products in '{category}' are '{a.product_name}' and '{b.product_name}' (distance {round(best_dist,2)})."
    return {
        "ok": True, "module": "pc_perceptual", "title": f"Closest-positioned products: {category}",
        "category": category,
        "chart": {"type": "scatter", "points": points, "x_label": X_LABEL, "y_label": Y_LABEL},
        "table": table, "summary": summary,
    }


def _most_different(category: str, product: Optional[str] = None) -> dict:
    subset = _points_for(category, cap=80)
    if len(subset) < 2:
        return {"ok": False, "module": "pc_perceptual", "error": "not_enough_products", "category": category}

    if product:
        target = subset[subset["product_name"] == product]
        if target.empty:
            return {"ok": False, "module": "pc_perceptual", "error": "no_data"}
        target = target.iloc[0]
        others = subset[subset["product_name"] != product].copy()
        others["distance"] = np.sqrt((others["pc1"] - target["pc1"]) ** 2 + (others["pc2"] - target["pc2"]) ** 2)
        far = others.sort_values("distance", ascending=False).iloc[0]
        points = [
            {"x": round(target["pc1"], 2), "y": round(target["pc2"], 2), "label": product, "highlight": True},
            {"x": round(far["pc1"], 2), "y": round(far["pc2"], 2), "label": far["product_name"], "highlight": True},
        ]
        table = [{"Product": far["product_name"], "Distance from " + product: round(far["distance"], 2)}]
        summary = f"The product most different from '{product}' in '{category}' is '{far['product_name']}' (distance {round(far['distance'],2)})."
        title = f"Most different from {product}"
    else:
        best_pair, best_dist = None, -1.0
        rows = list(subset.itertuples())
        for a, b in itertools.combinations(rows, 2):
            d = ((a.pc1 - b.pc1) ** 2 + (a.pc2 - b.pc2) ** 2) ** 0.5
            if d > best_dist:
                best_dist, best_pair = d, (a, b)
        a, b = best_pair
        points = [
            {"x": round(r.pc1, 2), "y": round(r.pc2, 2), "label": r.product_name, "highlight": (r.product_name in (a.product_name, b.product_name))}
            for r in rows
        ]
        table = [{"Product": a.product_name, "vs": b.product_name, "Distance": round(best_dist, 2)}]
        summary = f"The two most different products in '{category}' are '{a.product_name}' and '{b.product_name}' (distance {round(best_dist,2)})."
        title = f"Most different products: {category}"

    return {
        "ok": True, "module": "pc_perceptual", "title": title, "category": category,
        "chart": {"type": "scatter", "points": points, "x_label": X_LABEL, "y_label": Y_LABEL},
        "table": table, "summary": summary,
    }


def _clusters(category: str, k: int = 3) -> dict:
    subset = _points_for(category, cap=100)
    if len(subset) < k:
        return {"ok": False, "module": "pc_perceptual", "error": "not_enough_products", "category": category}

    coords = subset[["pc1", "pc2"]].to_numpy()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(coords), size=k, replace=False)
    centroids = coords[idx].copy()
    for _ in range(25):
        dists = np.linalg.norm(coords[:, None, :] - centroids[None, :, :], axis=2)
        assignments = dists.argmin(axis=1)
        new_centroids = np.array([
            coords[assignments == c].mean(axis=0) if (assignments == c).any() else centroids[c]
            for c in range(k)
        ])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    products_list = subset["product_name"].tolist()
    points = [
        {"x": round(float(coords[i][0]), 2), "y": round(float(coords[i][1]), 2), "label": products_list[i], "cluster": int(assignments[i])}
        for i in range(len(products_list))
    ]
    sizes = pd.Series(assignments).value_counts().sort_index()
    table = [{"Cluster": f"Cluster {c+1}", "Products": int(sizes.get(c, 0))} for c in range(k)]
    summary = f"'{category}' products split into {k} clusters by position, sized {', '.join(str(int(s)) for s in sizes)}."
    return {
        "ok": True, "module": "pc_perceptual", "title": f"Product clusters: {category}", "category": category,
        "chart": {"type": "scatter", "points": points, "x_label": X_LABEL, "y_label": Y_LABEL},
        "table": table, "summary": summary,
    }


def _whitespace(category: str) -> dict:
    subset = _points_for(category, cap=100)
    if subset.empty:
        return {"ok": False, "module": "pc_perceptual", "error": "no_data", "category": category}

    coords = subset[["pc1", "pc2"]].to_numpy()
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 12), np.linspace(y_min, y_max, 12))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    dists = np.linalg.norm(grid[:, None, :] - coords[None, :, :], axis=2).min(axis=1)
    top_idx = dists.argsort()[::-1][:3]

    whitespace_points = [{"x": round(float(grid[i][0]), 2), "y": round(float(grid[i][1]), 2), "label": f"Whitespace {n+1}", "highlight": True} for n, i in enumerate(top_idx)]
    product_points = [{"x": round(r.pc1, 2), "y": round(r.pc2, 2), "label": r.product_name, "highlight": False} for r in subset.itertuples()]
    table = [{"Region": p["label"], "PC1": p["x"], "PC2": p["y"], "Distance to nearest product": round(float(dists[top_idx[n]]), 2)} for n, p in enumerate(whitespace_points)]
    summary = f"The most under-served region of the '{category}' perceptual map is around (PC1={whitespace_points[0]['x']}, PC2={whitespace_points[0]['y']}), {round(table[0]['Distance to nearest product'],2)} units from the nearest existing product."
    return {
        "ok": True, "module": "pc_perceptual", "title": f"Whitespace opportunities: {category}", "category": category,
        "chart": {"type": "scatter", "points": product_points + whitespace_points, "x_label": X_LABEL, "y_label": Y_LABEL},
        "table": table, "summary": summary,
    }


def _compare(a_text: str, b_text: str) -> dict:
    a = best_match(a_text, _products, threshold=0.45)
    b = best_match(b_text, _products, threshold=0.45)
    if not a or not b:
        return {"ok": False, "module": "pc_perceptual", "error": "no_product_match", "known_products": _products[:15]}
    cat_a, cat_b = _category_of(a), _category_of(b)
    if cat_a != cat_b:
        return {"ok": False, "module": "pc_perceptual", "error": "different_categories", "detail": f"{a} is in {cat_a}, {b} is in {cat_b}"}
    return _pair_distance(cat_a, a, b)


def _pair_distance(category: str, a: str, b: str) -> dict:
    subset = _points_for(category, cap=200)
    ra = subset[subset["product_name"] == a].iloc[0]
    rb = subset[subset["product_name"] == b].iloc[0]
    dist = ((ra["pc1"] - rb["pc1"]) ** 2 + (ra["pc2"] - rb["pc2"]) ** 2) ** 0.5
    points = [
        {"x": round(r.pc1, 2), "y": round(r.pc2, 2), "label": r.product_name, "highlight": (r.product_name in (a, b))}
        for r in subset.itertuples()
    ]
    table = [{"Product": a, "PC1": round(ra["pc1"], 2), "PC2": round(ra["pc2"], 2)}, {"Product": b, "PC1": round(rb["pc1"], 2), "PC2": round(rb["pc2"], 2)}]
    summary = f"'{a}' and '{b}' are {round(dist,2)} units apart on the '{category}' perceptual map."
    return {
        "ok": True, "module": "pc_perceptual", "title": f"Perceptual map: {a} vs {b}", "category": category,
        "product": a, "compare_product": b,
        "chart": {"type": "scatter", "points": points, "x_label": X_LABEL, "y_label": Y_LABEL},
        "table": table, "summary": summary,
    }


def structured(category: str, mode: str = "map", product: Optional[str] = None,
               compare_product: Optional[str] = None, k: int = 3) -> dict:
    """Explicit-parameter counterpart to answer(). `mode` selects which of
    the already-parameterized helpers below to call:
      map (default) / nearest / closest / most_different / clusters /
      whitespace / compare (needs product + compare_product)."""
    _load()
    if category not in _categories:
        return {"ok": False, "module": "pc_perceptual", "error": "no_data", "category": category}

    if mode == "nearest" and product:
        return _nearest_neighbors(product, k=5)
    if mode == "closest":
        return _closest_pair(category)
    if mode == "most_different":
        return _most_different(category, product)
    if mode == "clusters":
        return _clusters(category, k=k)
    if mode == "whitespace":
        return _whitespace(category)
    if mode == "compare" and product and compare_product:
        return _pair_distance(category, product, compare_product)
    return _default_map(category, highlight_product=product)


def filters() -> dict:
    _load()
    return {
        "categories": _categories, "products": _products,
        "modes": ["map", "nearest", "closest", "most_different", "clusters", "whitespace", "compare"],
    }


def answer(question: str, top_n: int = 8) -> dict:
    _load()
    q = question.lower()

    if is_compare_query(question):
        halves = split_compare_halves(question)
        if halves:
            result = _compare(halves[0], halves[1])
            if result.get("ok"):
                return result

    category = best_match(question, _categories, threshold=0.75)
    product = best_match(question, _products, threshold=0.55)

    if any(k in q for k in NEIGHBOR_KEYWORDS) and product:
        return _nearest_neighbors(product, k=min(top_n, 5))

    if any(k in q for k in CLOSEST_PAIR_KEYWORDS):
        cat = category or (_category_of(product) if product else None) or _categories[0]
        return _closest_pair(cat)

    if any(k in q for k in DIFFERENT_KEYWORDS):
        cat = category or (_category_of(product) if product else None) or _categories[0]
        return _most_different(cat, product=product)

    if any(k in q for k in CLUSTER_KEYWORDS):
        cat = category or (_category_of(product) if product else None) or _categories[0]
        return _clusters(cat)

    if any(k in q for k in WHITESPACE_KEYWORDS):
        cat = category or (_category_of(product) if product else None) or _categories[0]
        return _whitespace(cat)

    if product and not category:
        cat = _category_of(product)
        if not cat:
            return {"ok": False, "module": "pc_perceptual", "error": "no_data"}
        return _default_map(cat, highlight_product=product)

    if category:
        return _default_map(category)

    return {"ok": False, "module": "pc_perceptual", "error": "no_match", "known_categories": _categories}
