"""
bh_blindhut: "Which products score highest on <benefit>?" /
             "How does <product> score across benefits?" /
             "Compare <productA> and <productB>"
Source: blind_hut_benefit_score.csv (record_type = blind_hut_benefit_score)
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import pandas as pd

from .utils import best_match, is_compare_query, snake_to_title, split_compare_halves

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "blind_hut_benefit_score.csv")

_df: Optional[pd.DataFrame] = None
_benefits_readable: list[str] = []
_benefit_map: dict[str, str] = {}
_products: list[str] = []
_lock = threading.Lock()


def _load():
    global _df, _benefits_readable, _benefit_map, _products
    if _df is None:
        with _lock:
            if _df is None:
                df = pd.read_csv(DATA_PATH)
                raw_benefits = sorted(df["benefit"].unique().tolist())
                _benefit_map = {snake_to_title(b): b for b in raw_benefits}
                _benefits_readable = list(_benefit_map.keys())
                _products = sorted(df["product_name"].unique().tolist())
                _df = df
    return _df


def preload():
    """Eagerly load data into memory to prevent cold-start latency."""
    _load()


def benefits() -> list[str]:
    _load()
    return _benefits_readable


def products() -> list[str]:
    _load()
    return _products


def _short(name: str, n: int = 28) -> str:
    return name if len(name) <= n else name[: n - 3] + "..."


def _by_product(product: str, top_n: int) -> dict:
    df = _load()
    subset = df[df["product_name"] == product].sort_values("value", ascending=False)
    if subset.empty:
        return {"ok": False, "module": "bh_blindhut", "error": "no_data"}
    top = subset.head(top_n)
    labels = [snake_to_title(b) for b in top["benefit"]]
    values = [round(v, 2) for v in top["value"]]
    table = [{"Benefit": snake_to_title(r.benefit), "Score": round(r.value, 2)} for r in top.itertuples()]
    lead = top.iloc[0]
    summary = f"'{product}' scores highest on '{snake_to_title(lead['benefit'])}' ({round(lead['value'],2)}) among its blind-hut-tested benefits."
    return {
        "ok": True, "module": "bh_blindhut", "title": f"Blind hut benefit profile: {product}",
        "product": product,
        "chart": {"type": "bar", "labels": labels, "values": values, "value_label": "Score"},
        "table": table, "summary": summary,
    }


def _by_benefit(benefit_readable: str, top_n: int) -> dict:
    df = _load()
    benefit_raw = _benefit_map[benefit_readable]
    subset = df[df["benefit"] == benefit_raw].sort_values("value", ascending=False)
    if subset.empty:
        return {"ok": False, "module": "bh_blindhut", "error": "no_data"}
    top = subset.head(top_n)
    labels = [_short(n) for n in top["product_name"]]
    values = [round(v, 2) for v in top["value"]]
    table = [{"Product": r.product_name, "Score": round(r.value, 2)} for r in top.itertuples()]
    lead = top.iloc[0]
    summary = f"For '{benefit_readable}', the top-scoring product in blind-hut testing is '{lead['product_name']}' with a score of {round(lead['value'],2)}."
    return {
        "ok": True, "module": "bh_blindhut", "title": f"Top products for {benefit_readable} (blind hut)",
        "benefit": benefit_readable,
        "chart": {"type": "bar", "labels": labels, "values": values, "value_label": "Score"},
        "table": table, "summary": summary,
    }


def _by_product_and_benefit(product: str, benefit_readable: str, top_n: int) -> dict:
    """The case _single() previously dropped: a question naming BOTH a
    product and a benefit (e.g. "how does X score on Y") was falling
    through to the benefit-only leaderboard and silently discarding the
    product. This returns that product's actual score on that benefit,
    plus its rank among every product tested on it for context."""
    df = _load()
    benefit_raw = _benefit_map[benefit_readable]
    subset = df[df["benefit"] == benefit_raw].sort_values("value", ascending=False).reset_index(drop=True)
    if subset.empty:
        return {"ok": False, "module": "bh_blindhut", "error": "no_data"}

    match_idx = subset.index[subset["product_name"] == product]
    if len(match_idx) == 0:
        return {
            "ok": False, "module": "bh_blindhut", "error": "product_not_tested_on_benefit",
            "product": product, "benefit": benefit_readable,
        }

    rank = int(match_idx[0]) + 1
    total = len(subset)
    score = round(float(subset.loc[match_idx[0], "value"]), 2)

    top = subset.head(top_n)
    labels = [_short(n) for n in top["product_name"]]
    values = [round(v, 2) for v in top["value"]]
    table = [{"Product": r.product_name, "Score": round(r.value, 2)} for r in top.itertuples()]
    if product not in top["product_name"].values:
        # Keep the queried product visible in the table even when it
        # ranks outside the requested top_n — otherwise the chart shows
        # everyone except the product that was actually asked about.
        table.append({"Product": product, "Score": score})

    summary = (
        f"'{product}' scores {score} on '{benefit_readable}', ranking {rank} of {total} "
        f"products tested on this benefit."
    )
    return {
        "ok": True, "module": "bh_blindhut",
        "title": f"{product} on {benefit_readable} (blind hut)",
        "product": product, "benefit": benefit_readable,
        "chart": {"type": "bar", "labels": labels, "values": values, "value_label": "Score"},
        "table": table, "summary": summary,
    }


def structured(benefit: Optional[str] = None, product: Optional[str] = None,
               compare_product: Optional[str] = None, top_n: int = 8) -> dict:
    """Explicit-parameter counterpart to answer(). Provide `product` alone
    for a product's benefit profile, `benefit` alone for the leaderboard on
    one benefit, `product` + `benefit` for that product's score on that one
    benefit, or `product` + `compare_product` for a head-to-head."""
    _load()
    if product and compare_product:
        if product not in _products or compare_product not in _products:
            return {"ok": False, "module": "bh_blindhut", "error": "no_product_match", "known_products": _products[:15]}
        return _compare(product, compare_product, top_n)
    if product and benefit:
        if product not in _products:
            return {"ok": False, "module": "bh_blindhut", "error": "no_data"}
        if benefit not in _benefit_map:
            return {"ok": False, "module": "bh_blindhut", "error": "no_match", "known_benefits": _benefits_readable}
        return _by_product_and_benefit(product, benefit, top_n)
    if product:
        if product not in _products:
            return {"ok": False, "module": "bh_blindhut", "error": "no_data"}
        return _by_product(product, top_n)
    if benefit:
        if benefit not in _benefit_map:
            return {"ok": False, "module": "bh_blindhut", "error": "no_match", "known_benefits": _benefits_readable}
        return _by_benefit(benefit, top_n)
    return {"ok": False, "module": "bh_blindhut", "error": "no_match", "known_benefits": _benefits_readable, "known_products": _products[:10]}


def filters() -> dict:
    _load()
    return {"benefits": _benefits_readable, "products": _products}


def _single(question: str, top_n: int) -> dict:
    df = _load()
    product = best_match(question, _products, threshold=0.5)
    benefit_readable = best_match(question, _benefits_readable)

    if product and benefit_readable:
        return _by_product_and_benefit(product, benefit_readable, top_n)

    if product:
        return _by_product(product, top_n)

    if benefit_readable:
        return _by_benefit(benefit_readable, top_n)

    return {"ok": False, "module": "bh_blindhut", "error": "no_match", "known_benefits": _benefits_readable, "known_products": _products[:10]}


def _compare(p1_text: str, p2_text: str, top_n: int) -> dict:
    df = _load()
    p1 = best_match(p1_text, _products, threshold=0.45)
    p2 = best_match(p2_text, _products, threshold=0.45)
    if not p1 or not p2:
        return {"ok": False, "module": "bh_blindhut", "error": "no_product_match", "known_products": _products[:15]}

    s1 = df[df["product_name"] == p1].set_index("benefit")["value"]
    s2 = df[df["product_name"] == p2].set_index("benefit")["value"]
    shared = sorted(set(s1.index) & set(s2.index))
    if not shared:
        return {"ok": False, "module": "bh_blindhut", "error": "no_shared_benefits"}

    diffs = sorted(shared, key=lambda b: abs(s1[b] - s2[b]), reverse=True)[:top_n]
    labels = [snake_to_title(b) for b in diffs]
    v1 = [round(float(s1[b]), 2) for b in diffs]
    v2 = [round(float(s2[b]), 2) for b in diffs]
    table = [{"Benefit": snake_to_title(b), p1: round(float(s1[b]), 2), p2: round(float(s2[b]), 2)} for b in diffs]

    lead_diff = diffs[0]
    winner = p1 if s1[lead_diff] > s2[lead_diff] else p2
    summary = (
        f"Comparing '{p1}' vs '{p2}' on consumer liking: the largest gap is on "
        f"'{snake_to_title(lead_diff)}', where '{winner}' scores higher "
        f"({round(float(s1[lead_diff]),2)} vs {round(float(s2[lead_diff]),2)})."
    )
    return {
        "ok": True, "module": "bh_blindhut", "title": f"Blind hut comparison: {p1} vs {p2}",
        "product": p1, "compare_product": p2,
        "chart": {"type": "bar", "labels": labels, "series": [{"name": p1, "values": v1}, {"name": p2, "values": v2}], "value_label": "Score"},
        "table": table, "summary": summary,
    }


def answer(question: str, top_n: int = 8) -> dict:
    if is_compare_query(question):
        halves = split_compare_halves(question)
        if halves:
            result = _compare(halves[0], halves[1], min(top_n, 6))
            if result.get("ok"):
                return result
    return _single(question, top_n)
