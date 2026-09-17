"""
sd_driver: "What test stages/drivers rank highest for <benefit>?" /
           "Which stage contributes most to <benefit>?" /
           "Top drivers for <benefit> by benefit stage" / "... for the Fragrance stage" /
           "Compare <benefit1> and <benefit2>"
Source: sst_driver_rank.csv (record_type = sst_driver_rank)
"""
from __future__ import annotations

import os
import re
import threading
from typing import Optional

import pandas as pd

from .utils import (
    BENEFIT_STAGE_ORDER,
    any_keyword,
    best_match,
    extract_clt_code,
    is_compare_query,
    match_stage_category,
    split_compare_halves,
    stage_category,
    wants_benefit_stage_breakdown,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sst_driver_rank.csv")

WEAKEST_KEYWORDS = ["weakest", "least important", "lowest impact", "matter least", "smallest impact", "matters least"]

_df: Optional[pd.DataFrame] = None
_benefits: list[str] = []
_drivers: list[str] = []
_driver_label_map: dict[str, str] = {}  # stripped descriptive label -> full driver string
_stage_codes: list[str] = []  # e.g. "CLT B7-8"
_lock = threading.Lock()

_CODE_PREFIX_RE = re.compile(r"(?i)^\s*CLT\s+[A-Z0-9-]+\s+")


def _strip_driver_code(driver: str) -> str:
    return _CODE_PREFIX_RE.sub("", driver).strip()


def _load():
    global _df, _benefits, _drivers, _driver_label_map, _stage_codes
    if _df is None:
        with _lock:
            if _df is None:
                df = pd.read_csv(DATA_PATH)
                df["category"] = df["driver"].apply(stage_category)
                df["stage_code"] = df["driver"].apply(extract_clt_code)
                _benefits = sorted(df["benefit"].unique().tolist())
                _drivers = sorted(df["driver"].unique().tolist())
                _driver_label_map = {_strip_driver_code(d): d for d in _drivers}
                _stage_codes = sorted(df["stage_code"].dropna().unique().tolist())
                _df = df
    return _df


def preload():
    """Eagerly load data into memory to prevent cold-start latency."""
    _load()


def benefits() -> list[str]:
    _load()
    return _benefits


def drivers() -> list[str]:
    _load()
    return _drivers


def _match_driver(question: str) -> Optional[str]:
    """Users refer to drivers by their descriptive suffix (e.g. "Skin Feel
    IMD") rather than the full "CLT B9-10 Skin feel IMD" code, so match
    against stripped labels first, falling back to the full driver strings."""
    label = best_match(question, list(_driver_label_map.keys()), threshold=0.6)
    if label:
        return _driver_label_map[label]
    return best_match(question, _drivers, threshold=0.7)


def _ranked(benefit: str, top_n: int):
    df = _load()
    subset = df[df["benefit"] == benefit].drop_duplicates(subset=["driver"]).sort_values("driver_rank")
    return subset, subset.head(top_n)


def _which_benefits_driven_by(driver: str, top_n: int) -> dict:
    """Which benefits are influenced most by <driver>? (used by CLT-stage style queries)"""
    df = _load()
    subset = df[df["driver"] == driver].drop_duplicates(subset=["benefit"]).sort_values("driver_rank")
    if subset.empty:
        return {"ok": False, "module": "sd_driver", "error": "no_data", "driver": driver}
    top = subset.head(top_n)
    max_rank = subset["driver_rank"].max()
    scores = [round(float(max_rank + 1 - r), 1) for r in top["driver_rank"]]
    table = [{"Benefit": r.benefit, "Rank": int(r.driver_rank)} for r in top.itertuples()]
    lead = top.iloc[0]
    summary = f"'{driver}' most strongly drives '{lead['benefit']}' (rank {int(lead['driver_rank'])})."
    return {
        "ok": True, "module": "sd_driver", "title": f"Benefits driven by {driver}",
        "driver": driver,
        "chart": {"type": "bar", "labels": top["benefit"].tolist(), "values": scores, "value_label": "Relative strength (inverse rank)"},
        "table": table, "summary": summary,
    }


def _build_response(benefit: str, subset_all: pd.DataFrame, subset: pd.DataFrame, top_n: int,
                     category: Optional[str] = None, stage_code: Optional[str] = None, weakest: bool = False) -> dict:
    if subset.empty:
        return {"ok": False, "module": "sd_driver", "error": "no_data", "benefit": benefit}

    subset = subset.sort_values("driver_rank", ascending=not weakest)
    top = subset.head(top_n)
    max_rank = subset_all["driver_rank"].max()
    scores = [round(float(max_rank + 1 - r), 1) for r in top["driver_rank"]]
    table = [{"Test stage / driver": r.driver, "Rank": int(r.driver_rank)} for r in top.itertuples()]
    lead = top.iloc[0]
    which = "weakest" if weakest else "strongest"
    scope_bits = [b for b in (category, stage_code) if b]
    scope = f" ({', '.join(scope_bits)})" if scope_bits else ""
    summary = (
        f"For '{benefit}'{scope}, the {which} test stage / driver is '{lead['driver']}' "
        f"(rank {int(lead['driver_rank'])} of {int(max_rank)} drivers evaluated)."
    )
    title = ("Weakest test-stage drivers of " if weakest else "Top test-stage drivers of ") + benefit
    title += (f" — {category}" if category else "") + (f" ({stage_code})" if stage_code else "")
    return {
        "ok": True, "module": "sd_driver", "title": title,
        "benefit": benefit, "category": category, "stage_code": stage_code,
        "chart": {"type": "bar", "labels": top["driver"].tolist(), "values": scores, "value_label": "Relative strength (inverse rank)"},
        "table": table, "summary": summary,
    }


def _breakdown(benefit: str, subset_all: pd.DataFrame, top_n: int) -> dict:
    """Top test-stage drivers for `benefit`, split by benefit-stage category
    (Skin Feel / Fingertip / Skin Look / Fragrance) rather than ranked as
    one undifferentiated list of ten test stages."""
    per_cat_n = max(1, min(top_n, 3))
    max_rank = subset_all["driver_rank"].max()
    labels, values, table, lead_lines = [], [], [], []

    for cat in BENEFIT_STAGE_ORDER:
        cat_df = subset_all[subset_all["category"] == cat].sort_values("driver_rank")
        if cat_df.empty:
            continue
        cat_top = cat_df.head(per_cat_n)
        for r in cat_top.itertuples():
            score = round(float(max_rank + 1 - r.driver_rank), 1)
            labels.append(f"{cat}: {_strip_driver_code(r.driver)}")
            values.append(score)
            table.append({"Benefit stage": cat, "Test stage / driver": r.driver, "Rank": int(r.driver_rank)})
        lead = cat_top.iloc[0]
        lead_lines.append(f"{cat} → {_strip_driver_code(lead['driver'])} (rank {int(lead['driver_rank'])})")

    summary = f"For '{benefit}', strongest test stage by benefit-stage category: " + "; ".join(lead_lines) + "."
    return {
        "ok": True, "module": "sd_driver",
        "title": f"Top test-stage drivers of {benefit}, by benefit stage",
        "benefit": benefit, "breakdown": True,
        "chart": {"type": "bar", "labels": labels, "values": values, "value_label": "Relative strength (inverse rank)"},
        "table": table, "summary": summary,
    }


def _single(question: str, top_n: int) -> dict:
    df = _load()

    # "Which benefits are driven by <driver>?" style — check driver match first
    driver = _match_driver(question)
    if driver and ("benefit" in question.lower() or "driven by" in question.lower()):
        return _which_benefits_driven_by(driver, top_n)

    benefit = best_match(question, _benefits)
    if not benefit:
        return {"ok": False, "module": "sd_driver", "error": "no_benefit_match", "known": _benefits}

    subset_all = df[df["benefit"] == benefit].drop_duplicates(subset=["driver"])
    if subset_all.empty:
        return {"ok": False, "module": "sd_driver", "error": "no_data", "benefit": benefit}

    category = match_stage_category(question)
    stage_code = best_match(question, _stage_codes, threshold=0.7)

    if not category and not stage_code and wants_benefit_stage_breakdown(question):
        return _breakdown(benefit, subset_all, top_n)

    subset = subset_all
    if category:
        subset = subset[subset["category"] == category]
    if stage_code:
        subset = subset[subset["stage_code"] == stage_code]

    weakest = any_keyword(question, WEAKEST_KEYWORDS)
    return _build_response(benefit, subset_all, subset, top_n, category=category, stage_code=stage_code, weakest=weakest)


def _compare(b1_text: str, b2_text: str, top_n: int) -> dict:
    _load()
    b1 = best_match(b1_text, _benefits)
    b2 = best_match(b2_text, _benefits)
    if not b1 or not b2:
        return {"ok": False, "module": "sd_driver", "error": "no_benefit_match", "known": _benefits}

    full1, top1 = _ranked(b1, top_n)
    full2, top2 = _ranked(b2, top_n)
    max_rank = max(full1["driver_rank"].max(), full2["driver_rank"].max())
    r1 = full1.set_index("driver")["driver_rank"]
    r2 = full2.set_index("driver")["driver_rank"]

    drivers_union = list(dict.fromkeys(top1["driver"].tolist() + top2["driver"].tolist()))
    v1 = [round(float(max_rank + 1 - r1.get(d, max_rank + 1)), 1) for d in drivers_union]
    v2 = [round(float(max_rank + 1 - r2.get(d, max_rank + 1)), 1) for d in drivers_union]

    table = [{"Driver": d, f"{b1} rank": int(r1.get(d, -1)) if d in r1 else "—", f"{b2} rank": int(r2.get(d, -1)) if d in r2 else "—"} for d in drivers_union]
    shared = set(top1["driver"].head(3)) & set(top2["driver"].head(3))
    summary = (
        f"Comparing test-stage drivers: '{b1}' is led by '{top1.iloc[0]['driver']}' "
        f"(rank {int(top1.iloc[0]['driver_rank'])}); '{b2}' is led by '{top2.iloc[0]['driver']}' "
        f"(rank {int(top2.iloc[0]['driver_rank'])})."
        + (f" Shared top-3 driver(s): {', '.join(shared)}." if shared else " No overlap in top-3 drivers.")
    )
    return {
        "ok": True, "module": "sd_driver", "title": f"Test-stage drivers: {b1} vs {b2}",
        "benefit": b1, "compare_benefit": b2,
        "chart": {"type": "bar", "labels": drivers_union, "series": [{"name": b1, "values": v1}, {"name": b2, "values": v2}], "value_label": "Relative strength"},
        "table": table, "summary": summary,
    }


def structured(benefit: str, category: Optional[str] = None, stage_code: Optional[str] = None,
               breakdown: bool = False, weakest: bool = False, top_n: int = 8) -> dict:
    """Explicit-parameter counterpart to answer() — see sr_attribute.structured()."""
    df = _load()
    if benefit not in _benefits:
        return {"ok": False, "module": "sd_driver", "error": "no_benefit_match", "known": _benefits}
    subset_all = df[df["benefit"] == benefit].drop_duplicates(subset=["driver"])
    if subset_all.empty:
        return {"ok": False, "module": "sd_driver", "error": "no_data", "benefit": benefit}

    if breakdown:
        return _breakdown(benefit, subset_all, top_n)

    subset = subset_all
    if category and category in BENEFIT_STAGE_ORDER:
        subset = subset[subset["category"] == category]
    if stage_code and stage_code in _stage_codes:
        subset = subset[subset["stage_code"] == stage_code]

    return _build_response(benefit, subset_all, subset, top_n, category=category, stage_code=stage_code, weakest=weakest)


def filters() -> dict:
    _load()
    return {"benefits": _benefits, "categories": BENEFIT_STAGE_ORDER, "stage_codes": _stage_codes}


def answer(question: str, top_n: int = 8) -> dict:
    if is_compare_query(question):
        halves = split_compare_halves(question)
        if halves:
            result = _compare(halves[0], halves[1], min(top_n, 6))
            if result.get("ok"):
                return result
    return _single(question, top_n)
