"""
spider_profile: attribute-profile (radar) queries.
  - "Show the attribute profile for <product/brand>"
  - "What are the strongest/weakest attributes of <product>?"
  - "Compare <productA> and <productB>"
  - "What makes <product> unique?" / "Which attributes distinguish <product>?"
  - "Show sensory gaps against benchmark" / "...relative to Benchmark Y"
Source: spider_attribute_score.csv (record_type = spider_attribute_score)
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import pandas as pd

from .utils import best_match, is_compare_query, split_compare_halves

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "spider_attribute_score.csv")

_df: Optional[pd.DataFrame] = None
_products: list[str] = []
_brands: list[str] = []
_groups: list[str] = []
_product_types: list[str] = []
_lock = threading.Lock()

MAX_AXES = 12
WEAKEST_KEYWORDS = ["weakest"]
UNIQUE_KEYWORDS = ["unique", "distinguish", "distinctive"]
GAP_KEYWORDS = ["gap", "benchmark", "improvement relative", "relative to"]


def _load():
    global _df, _products, _brands, _groups, _product_types
    if _df is None:
        with _lock:
            if _df is None:
                df = pd.read_csv(DATA_PATH)
                _products = sorted(df["product_name"].unique().tolist())
                _brands = sorted(df["brand"].unique().tolist())
                _groups = sorted(df["attribute_group"].unique().tolist())
                _product_types = sorted(df["product_type"].dropna().unique().tolist())
                _df = df
    return _df


def preload():
    """Eagerly load data into memory to prevent cold-start latency."""
    _load()


def products() -> list[str]:
    _load()
    return _products


def brands() -> list[str]:
    _load()
    return _brands


def groups() -> list[str]:
    _load()
    return _groups


def product_types() -> list[str]:
    _load()
    return _product_types


def structured(products: list[str], attr_group: Optional[str] = None, product_type: Optional[str] = None) -> dict:
    """Explicit multi-product overlay — the structured counterpart to a
    filter-panel UI ("Attribute Groups" / "Product Type" / "Product
    Names" dropdowns): every attribute in the chosen group, plotted for
    each named product, no fuzzy text matching involved."""
    df = _load()
    subset = df
    if attr_group and attr_group in _groups:
        subset = subset[subset["attribute_group"] == attr_group]
    if product_type and product_type in _product_types:
        subset = subset[subset["product_type"] == product_type]

    valid = [p for p in products if p in set(_products)]
    if not valid or subset.empty:
        return {"ok": False, "module": "spider_profile", "error": "no_data"}

    axes = sorted(subset["attribute"].unique().tolist())[:MAX_AXES]
    if not axes:
        return {"ok": False, "module": "spider_profile", "error": "no_data"}

    series, table = [], []
    for p in valid[:6]:
        s = subset[subset["product_name"] == p].drop_duplicates(subset=["attribute"]).set_index("attribute")["value"]
        values = [round(float(s.get(a, 0)), 1) for a in axes]
        series.append({"name": p, "values": values})
        for a, v in zip(axes, values):
            table.append({"Product": p, "Attribute": a, "Value": v})

    scope_bits = [b for b in (attr_group, product_type) if b]
    scope = f" ({', '.join(scope_bits)})" if scope_bits else ""
    summary = f"Spider plot across {len(axes)} attributes{scope} for {', '.join(valid[:6])}."
    return {
        "ok": True, "module": "spider_profile",
        "title": "Spider Plot" + (f": {attr_group}" if attr_group else ""),
        "product": valid[0], "products": valid, "attr_group": attr_group, "product_type": product_type,
        "chart": {"type": "radar", "axes": axes, "series": series},
        "table": table, "summary": summary,
    }


def filters() -> dict:
    df = _load()
    # product_type -> its products, so a filter-panel UI can cascade the
    # Product Type dropdown into Product Names without another round trip.
    products_by_type = (
        df.groupby("product_type")["product_name"].unique().apply(lambda a: sorted(a.tolist())).to_dict()
    )
    return {
        "groups": _groups, "product_types": _product_types, "products": _products,
        "products_by_type": products_by_type,
    }


def _match_product(text: str) -> Optional[str]:
    product = best_match(text, _products, threshold=0.55)
    if product:
        return product
    brand = best_match(text, _brands, threshold=0.6)
    if brand:
        df = _load()
        rows = df[df["brand"] == brand]
        return None if rows.empty else rows.iloc[0]["product_name"]
    return None


def _profile(product: str, attr_group: Optional[str], top_n: int, ascending: bool = False) -> pd.DataFrame:
    df = _load()
    subset = df[df["product_name"] == product]
    if attr_group:
        subset = subset[subset["attribute_group"] == attr_group]
    subset = subset.drop_duplicates(subset=["attribute"])
    return subset.sort_values("value", ascending=ascending).head(top_n)


def _single(question: str, top_n: int) -> dict:
    q = question.lower()
    product = _match_product(question)
    if not product:
        return {"ok": False, "module": "spider_profile", "error": "no_match", "known_brands": _brands[:15]}

    attr_group = best_match(question, _groups, threshold=0.5)
    label = product

    if any(k in q for k in UNIQUE_KEYWORDS):
        return _uniqueness(product, top_n)

    ascending = any(k in q for k in WEAKEST_KEYWORDS)
    ranked = _profile(product, attr_group, top_n, ascending=ascending)
    if ranked.empty:
        return {"ok": False, "module": "spider_profile", "error": "no_data", "product": product}

    axes = ranked["attribute"].tolist()
    values = [round(v, 1) for v in ranked["value"]]
    table = [{"Attribute": r.attribute, "Value": round(r.value, 1), "Group": r.attribute_group} for r in ranked.itertuples()]
    lead = ranked.iloc[0]
    which = "weakest" if ascending else "strongest"
    summary = f"For '{label}', the {which} attribute is '{lead['attribute']}' at {round(lead['value'],1)}, across {len(axes)} attributes measured."
    return {
        "ok": True, "module": "spider_profile", "title": f"Attribute profile: {label}", "product": product,
        "chart": {"type": "radar", "axes": axes, "series": [{"name": label, "values": values}]},
        "table": table, "summary": summary,
    }


def _uniqueness(product: str, top_n: int) -> dict:
    df = _load()
    prod_rows = df[df["product_name"] == product].drop_duplicates(subset=["attribute"])
    if prod_rows.empty:
        return {"ok": False, "module": "spider_profile", "error": "no_data", "product": product}

    means = df.groupby("attribute")["value"].mean()
    prod_rows = prod_rows.assign(deviation=prod_rows.apply(lambda r: r["value"] - means.get(r["attribute"], r["value"]), axis=1))
    ranked = prod_rows.reindex(prod_rows["deviation"].abs().sort_values(ascending=False).index).head(top_n)

    axes = ranked["attribute"].tolist()
    values = [round(v, 1) for v in ranked["value"]]
    table = [{"Attribute": r.attribute, "Value": round(r.value, 1), "Category avg": round(float(means.get(r.attribute, 0)), 1), "Deviation": round(r.deviation, 1)} for r in ranked.itertuples()]
    lead = ranked.iloc[0]
    direction = "higher" if lead["deviation"] > 0 else "lower"
    summary = f"What makes '{product}' distinctive: '{lead['attribute']}' is {round(abs(lead['deviation']),1)} points {direction} than the category average."
    return {
        "ok": True, "module": "spider_profile", "title": f"What makes {product} unique", "product": product,
        "chart": {"type": "radar", "axes": axes, "series": [{"name": product, "values": values}]},
        "table": table, "summary": summary,
    }


def _compare(a_text: str, b_text: str, top_n: int, gap_mode: bool = False) -> dict:
    a = _match_product(a_text)
    b = _match_product(b_text)
    if not a or not b:
        return {"ok": False, "module": "spider_profile", "error": "no_match", "known_brands": _brands[:15]}

    df = _load()
    sa = df[df["product_name"] == a].drop_duplicates(subset=["attribute"]).set_index("attribute")["value"]
    sb = df[df["product_name"] == b].drop_duplicates(subset=["attribute"]).set_index("attribute")["value"]
    shared = sorted(set(sa.index) & set(sb.index))
    if not shared:
        return {"ok": False, "module": "spider_profile", "error": "no_shared_attributes"}

    diffs = sorted(shared, key=lambda a_: abs(sa[a_] - sb[a_]), reverse=True)[:top_n]
    va = [round(float(sa[x]), 1) for x in diffs]
    vb = [round(float(sb[x]), 1) for x in diffs]

    if gap_mode:
        table = [{"Attribute": x, a: va[i], b + " (benchmark)": vb[i], "Gap": round(vb[i] - va[i], 1)} for i, x in enumerate(diffs)]
        needs_improvement = sorted(table, key=lambda r: r["Gap"], reverse=True)[0]
        summary = (
            f"Comparing '{a}' against benchmark '{b}': the biggest gap is on '{needs_improvement['Attribute']}', "
            f"where the benchmark scores {needs_improvement['Gap']} points higher — the area most needing improvement."
        )
        title = f"Sensory gaps: {a} vs benchmark {b}"
    else:
        table = [{"Attribute": x, a: va[i], b: vb[i]} for i, x in enumerate(diffs)]
        lead = diffs[0]
        summary = f"Comparing '{a}' and '{b}': the biggest difference is on '{lead}' ({round(sa[lead],1)} vs {round(sb[lead],1)})."
        title = f"Attribute profile: {a} vs {b}"

    return {
        "ok": True, "module": "spider_profile", "title": title, "product": a, "compare_product": b,
        "chart": {"type": "radar", "axes": diffs, "series": [{"name": a, "values": va}, {"name": b, "values": vb}]},
        "table": table, "summary": summary,
    }


def answer(question: str, top_n: int = 8) -> dict:
    _load()
    top_n = min(top_n, MAX_AXES)
    q = question.lower()

    if is_compare_query(question):
        halves = split_compare_halves(question)
        if halves:
            gap_mode = any(k in q for k in GAP_KEYWORDS)
            result = _compare(halves[0], halves[1], top_n, gap_mode=gap_mode)
            if result.get("ok"):
                return result

    if any(k in q for k in GAP_KEYWORDS) and "benchmark" in q:
        # "gaps against benchmark" without an explicit "compare A and B" — try to
        # find two product/brand mentions in the question directly.
        product = _match_product(question)
        if product:
            # find a second, different product mention for the benchmark
            remaining = q.replace(product.lower(), "")
            benchmark = _match_product(remaining)
            if benchmark and benchmark != product:
                return _compare(product, benchmark, top_n, gap_mode=True)

    return _single(question, top_n)
