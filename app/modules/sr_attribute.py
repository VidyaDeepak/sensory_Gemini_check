"""
sr_attribute: "What attributes drive <benefit>?" / "Compare <benefit1> and <benefit2>"
             / "Top attributes for <benefit> by benefit stage" / "... for <benefit> Fragrance stage"
Source: sst_attribute_rank.csv (record_type = sst_attribute_rank)
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import pandas as pd

from .utils import (
    BENEFIT_STAGE_ORDER,
    any_keyword,
    best_match,
    is_compare_query,
    match_stage_category,
    split_compare_halves,
    stage_category,
    wants_benefit_stage_breakdown,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sst_attribute_rank.csv")

# The original Spider_Chart data-prep notebooks (both FC and HBL categories)
# explicitly drop these three attributes as non-perceptual noise before
# export: df[~df['AttribName'].isin(['product_b', 'product_g', 'product_r'])].
# They're raw RGB color-channel calibration readings, not something a
# consumer actually perceives. That same cleaning step was never applied to
# the SST Scoring notebook that feeds *this* dataset, so they slip through
# here and — because their attr_high/attr_low range is a fixed property of
# the test stage rather than the benefit — they end up dominating "top
# driver" for almost every benefit that shares that test stage, masking the
# genuine benefit-specific differentiators underneath them.
_NOISE_ATTRIBUTES = {"product_b", "product_g", "product_r"}

_df: Optional[pd.DataFrame] = None
_benefits: list[str] = []
_stages: list[str] = []
_lock = threading.Lock()

WEAKEST_KEYWORDS = ["weakest", "least important", "lowest impact", "matter least", "smallest impact", "matters least"]


def _load():
    global _df, _benefits, _stages
    if _df is None:
        with _lock:
            if _df is None:
                df = pd.read_csv(DATA_PATH)
                df = df[~df["attribute"].isin(_NOISE_ATTRIBUTES)].reset_index(drop=True)
                df["impact"] = df["attr_high"] - df["attr_low"]
                df["category"] = df["attr_group"].apply(stage_category)
                _benefits = sorted(df["benefit"].unique().tolist())
                _stages = sorted(df["stage"].unique().tolist())
                _df = df
    return _df


def preload():
    """Eagerly load data into memory to prevent cold-start latency."""
    _load()


def benefits() -> list[str]:
    _load()
    return _benefits


def _agg_from(subset: pd.DataFrame, top_n: int, ascending: bool = False):
    # Primary sort key is avg_rank, not avg_impact. The source SST Scoring
    # notebook only ever exports `AttributeRank` as the driver-importance
    # signal for a benefit — it never computes anything like "impact".
    # attr_high/attr_low (and therefore impact = high-low) are a raw
    # measurement range tied to (attribute, test stage), not to the
    # benefit, so the same physical reading repeats identically across
    # every benefit that shares a test stage — it isn't benefit-specific
    # and shouldn't decide "top driver". Rank is what genuinely varies per
    # benefit, matching how the sibling sd_driver module ranks purely by
    # driver_rank. Impact is kept only as a secondary tie-breaker and as
    # supplementary context in the table.
    agg = (
        subset.groupby("attribute")
        .agg(avg_rank=("attribute_rank", "mean"), avg_impact=("impact", "mean"), appearances=("attribute_rank", "size"))
        .reset_index()
        .sort_values(["avg_rank", "avg_impact"], ascending=[not ascending, ascending])
    )
    return agg.head(top_n), agg


def _top_attributes(benefit: str, top_n: int):
    df = _load()
    subset = df[df["benefit"] == benefit]
    top, agg = _agg_from(subset, top_n)
    return subset, top, agg


def _build_response(benefit: str, subset: pd.DataFrame, top_n: int, category: Optional[str] = None,
                     stage: Optional[str] = None, ascending: bool = False) -> dict:
    if subset.empty:
        return {"ok": False, "module": "sr_attribute", "error": "no_data", "benefit": benefit}

    top, _ = _agg_from(subset, top_n, ascending=ascending)
    labels = top["attribute"].tolist()
    values = [round(v, 1) for v in top["avg_impact"].tolist()]
    table = [
        {"Attribute": r.attribute, "Avg rank": round(r.avg_rank, 2), "Avg impact (pts)": round(r.avg_impact, 1), "Seen in": int(r.appearances)}
        for r in top.itertuples()
    ]
    lead = top.iloc[0]
    which = "weakest" if ascending else "top"
    scope_bits = [b for b in (category, stage) if b]
    scope = f" ({', '.join(scope_bits)})" if scope_bits else ""
    summary = (
        f"For '{benefit}'{scope}, the {which} driver is '{lead['attribute']}' with an average rank of "
        f"{round(lead['avg_rank'],2)} (avg impact {round(lead['avg_impact'],1)} pts) across "
        f"{subset['stage_group'].nunique()} test stage group(s)."
    )
    title = ("Weakest attribute drivers of " if ascending else "Attribute drivers of ") + benefit
    title += (f" — {category}" if category else "") + (f" ({stage})" if stage else "")
    return {
        "ok": True, "module": "sr_attribute", "title": title,
        "benefit": benefit, "stage": stage, "category": category,
        "chart": {"type": "bar", "labels": labels, "values": values, "value_label": "Avg impact (pts)"},
        "table": table, "summary": summary,
    }


def _breakdown(benefit: str, subset_all: pd.DataFrame, top_n: int) -> dict:
    """Top attributes for `benefit`, split out by benefit-stage category
    (Skin Feel / Fingertip / Skin Look / Fragrance) instead of lumped
    together — mirrors the grouping analysts expect from the source study
    design, so a color/shade attribute never gets ranked against a
    fragrance attribute."""
    per_cat_n = max(3, min(top_n, 5))
    labels, values, table, lead_lines = [], [], [], []

    for cat in BENEFIT_STAGE_ORDER:
        cat_df = subset_all[subset_all["category"] == cat]
        if cat_df.empty:
            continue
        agg, _ = _agg_from(cat_df, per_cat_n)
        for r in agg.itertuples():
            labels.append(f"{cat}: {r.attribute}")
            values.append(round(r.avg_impact, 1))
            table.append({
                "Benefit stage": cat, "Attribute": r.attribute,
                "Avg impact (pts)": round(r.avg_impact, 1), "Avg rank": round(r.avg_rank, 2),
            })
        lead = agg.iloc[0]
        lead_lines.append(f"{cat} → {lead['attribute']} ({round(lead['avg_impact'],1)} pts)")

    summary = f"For '{benefit}', top driver by benefit stage: " + "; ".join(lead_lines) + "."
    return {
        "ok": True, "module": "sr_attribute",
        "title": f"Attribute drivers of {benefit}, by benefit stage",
        "benefit": benefit, "breakdown": True,
        "chart": {"type": "bar", "labels": labels, "values": values, "value_label": "Avg impact (pts)"},
        "table": table, "summary": summary,
    }


def _single(question: str, top_n: int) -> dict:
    df = _load()
    benefit = best_match(question, _benefits)
    if not benefit:
        return {"ok": False, "module": "sr_attribute", "error": "no_benefit_match", "known": _benefits}

    subset_all = df[df["benefit"] == benefit]
    if subset_all.empty:
        return {"ok": False, "module": "sr_attribute", "error": "no_data", "benefit": benefit}

    category = match_stage_category(question)
    stage = best_match(question, _stages, threshold=0.7)

    if not category and not stage and wants_benefit_stage_breakdown(question):
        return _breakdown(benefit, subset_all, top_n)

    subset = subset_all
    if category:
        subset = subset[subset["category"] == category]
    if stage:
        subset = subset[subset["stage"] == stage]

    ascending = any_keyword(question, WEAKEST_KEYWORDS)
    return _build_response(benefit, subset, top_n, category=category, stage=stage, ascending=ascending)


def _compare(b1_text: str, b2_text: str, top_n: int) -> dict:
    _load()
    b1 = best_match(b1_text, _benefits)
    b2 = best_match(b2_text, _benefits)
    if not b1 or not b2:
        return {"ok": False, "module": "sr_attribute", "error": "no_benefit_match", "known": _benefits}

    _, top1, full1 = _top_attributes(b1, top_n)
    _, top2, full2 = _top_attributes(b2, top_n)
    s1 = full1.set_index("attribute")["avg_impact"]
    s2 = full2.set_index("attribute")["avg_impact"]

    attrs = list(dict.fromkeys(top1["attribute"].tolist() + top2["attribute"].tolist()))
    v1 = [round(float(s1.get(a, 0)), 1) for a in attrs]
    v2 = [round(float(s2.get(a, 0)), 1) for a in attrs]

    table = [{"Attribute": a, b1: x, b2: y} for a, x, y in zip(attrs, v1, v2)]
    shared_top = set(top1["attribute"].head(3)) & set(top2["attribute"].head(3))
    summary = (
        f"Comparing '{b1}' vs '{b2}': top driver for '{b1}' is '{top1.iloc[0]['attribute']}' "
        f"(avg rank {round(top1.iloc[0]['avg_rank'],2)}); top driver for '{b2}' is "
        f"'{top2.iloc[0]['attribute']}' (avg rank {round(top2.iloc[0]['avg_rank'],2)})."
        + (f" Shared top-3 driver(s): {', '.join(shared_top)}." if shared_top else " No overlap in top-3 drivers.")
    )
    return {
        "ok": True, "module": "sr_attribute", "title": f"Attribute drivers: {b1} vs {b2}",
        "benefit": b1, "compare_benefit": b2,
        "chart": {"type": "bar", "labels": attrs, "series": [{"name": b1, "values": v1}, {"name": b2, "values": v2}], "value_label": "Avg impact (pts)"},
        "table": table, "summary": summary,
    }


def structured(benefit: str, category: Optional[str] = None, stage: Optional[str] = None,
               breakdown: bool = False, weakest: bool = False, top_n: int = 8) -> dict:
    """Explicit-parameter counterpart to answer() — every value is checked
    against the real known list instead of fuzzy-matched from free text,
    so a dropdown-driven UI always gets exactly the filter it selected."""
    df = _load()
    if benefit not in _benefits:
        return {"ok": False, "module": "sr_attribute", "error": "no_benefit_match", "known": _benefits}
    subset_all = df[df["benefit"] == benefit]
    if subset_all.empty:
        return {"ok": False, "module": "sr_attribute", "error": "no_data", "benefit": benefit}

    if breakdown:
        return _breakdown(benefit, subset_all, top_n)

    subset = subset_all
    if category and category in BENEFIT_STAGE_ORDER:
        subset = subset[subset["category"] == category]
    if stage and stage in _stages:
        subset = subset[subset["stage"] == stage]

    return _build_response(benefit, subset, top_n, category=category, stage=stage, ascending=weakest)


def filters() -> dict:
    _load()
    return {"benefits": _benefits, "categories": BENEFIT_STAGE_ORDER, "stages": _stages}


def answer(question: str, top_n: int = 8) -> dict:
    if is_compare_query(question):
        halves = split_compare_halves(question)
        if halves:
            result = _compare(halves[0], halves[1], min(top_n, 6))
            if result.get("ok"):
                return result
    return _single(question, top_n)
