"""
rw_relweight: "What drives Shine / Hairfall / Dry [shampoo] importance?" /
              "weakest drivers" / "consistent across studies" / "summarize learnings" /
              "compare drivers across haircare studies"
Source: rwa_driver_importance.csv (record_type = rwa_driver_importance, category_code = HRC)
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import pandas as pd

from .utils import best_match, is_compare_query, split_compare_halves

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "rwa_driver_importance.csv")

_df: Optional[pd.DataFrame] = None
_platforms: list[str] = []
_lock = threading.Lock()

WEAKEST_KEYWORDS = ["weakest", "least important", "lowest importance"]
CONSISTENT_KEYWORDS = ["consistent", "common across", "recur"]
SUMMARY_KEYWORDS = ["summarize", "summary", "learnings", "key learnings"]
STRONGEST_KEYWORDS = ["strongest"]


def _load():
    global _df, _platforms
    if _df is None:
        with _lock:
            if _df is None:
                df = pd.read_csv(DATA_PATH)
                _platforms = sorted(df["platform"].unique().tolist())
                _df = df
    return _df


def preload():
    """Eagerly load data into memory to prevent cold-start latency."""
    _load()


def platforms() -> list[str]:
    _load()
    return _platforms


def _agg(subset: pd.DataFrame):
    return (
        subset.groupby("driver")
        .agg(avg_importance=("importance_percent", "mean"), appearances=("importance_percent", "size"))
        .reset_index()
    )


def _ranked_result(platform: str, agg: pd.DataFrame, top_n: int, ascending: bool, label: str) -> dict:
    ranked = agg.sort_values("avg_importance", ascending=ascending).head(top_n)
    labels = ranked["driver"].tolist()
    values = [round(v, 1) for v in ranked["avg_importance"]]
    table = [{"Driver": r.driver, "Avg importance (%)": round(r.avg_importance, 1), "Studies": int(r.appearances)} for r in ranked.itertuples()]
    lead = ranked.iloc[0]
    summary = (
        f"For the '{platform}' hair care platform, the {label} driver is '{lead['driver']}' "
        f"at {round(lead['avg_importance'],1)}% average relative weight."
    )
    return {
        "ok": True, "module": "rw_relweight", "title": f"{label.capitalize()} drivers: {platform}",
        "platform": platform,
        "chart": {"type": "bar", "labels": labels, "values": values, "value_label": "Avg importance (%)"},
        "table": table, "summary": summary,
    }


def _consistency(platform: str, subset: pd.DataFrame, top_n: int) -> dict:
    n_studies = subset["study_name"].nunique()
    top3_per_study = (
        subset.sort_values("importance_percent", ascending=False)
        .groupby("study_name")
        .head(3)
    )
    counts = top3_per_study.groupby("driver")["study_name"].nunique().reset_index(name="studies_in_top3")
    counts = counts.sort_values("studies_in_top3", ascending=False).head(top_n)

    if counts.empty:
        return {"ok": False, "module": "rw_relweight", "error": "no_data", "platform": platform}

    table = [{"Driver": r.driver, "Studies where top-3": int(r.studies_in_top3), "Of studies": n_studies} for r in counts.itertuples()]
    lead = counts.iloc[0]
    summary = (
        f"Across {n_studies} '{platform}' studies, '{lead['driver']}' is the most consistently important driver, "
        f"appearing in the top 3 in {int(lead['studies_in_top3'])} of {n_studies} studies."
    )
    return {
        "ok": True, "module": "rw_relweight", "title": f"Most consistent drivers: {platform}",
        "platform": platform,
        "chart": {"type": "bar", "labels": counts["driver"].tolist(), "values": [int(v) for v in counts["studies_in_top3"]], "value_label": "Studies in top 3"},
        "table": table, "summary": summary,
    }


def _summarize_all(top_n: int) -> dict:
    df = _load()
    rows = []
    for platform in _platforms:
        agg = _agg(df[df["platform"] == platform]).sort_values("avg_importance", ascending=False)
        if agg.empty:
            continue
        top = agg.iloc[0]
        bottom = agg.iloc[-1]
        rows.append({"Platform": platform, "Top driver": top["driver"], "Top importance (%)": round(top["avg_importance"], 1),
                      "Weakest driver": bottom["driver"], "Weakest importance (%)": round(bottom["avg_importance"], 1)})
    labels = [r["Platform"] for r in rows]
    values = [r["Top importance (%)"] for r in rows]
    summary = "Key RWA learnings: " + "; ".join(f"{r['Platform']} is led by '{r['Top driver']}'" for r in rows) + "."
    return {
        "ok": True, "module": "rw_relweight", "title": "RWA summary across platforms",
        "chart": {"type": "bar", "labels": labels, "values": values, "value_label": "Top driver importance (%)"},
        "table": rows, "summary": summary,
    }


def _compare_studies(platform: str, top_n: int) -> dict:
    """Compare drivers across studies within one platform (grouped bar per study)."""
    df = _load()
    subset = df[df["platform"] == platform]
    studies = subset["study_name"].unique().tolist()
    if len(studies) < 2:
        return {"ok": False, "module": "rw_relweight", "error": "only_one_study", "platform": platform}

    overall_top_drivers = _agg(subset).sort_values("avg_importance", ascending=False).head(top_n)["driver"].tolist()
    series = []
    for study in studies:
        s = subset[subset["study_name"] == study].groupby("driver")["importance_percent"].mean()
        series.append({"name": study[:30], "values": [round(float(s.get(d, 0)), 1) for d in overall_top_drivers]})

    table = [{"Driver": d, **{s["name"]: s["values"][i] for s in series}} for i, d in enumerate(overall_top_drivers)]
    summary = f"Comparing driver importance for '{platform}' across {len(studies)} studies, for the {len(overall_top_drivers)} overall top drivers."
    return {
        "ok": True, "module": "rw_relweight", "title": f"Driver importance across studies: {platform}",
        "platform": platform,
        "chart": {"type": "bar", "labels": overall_top_drivers, "series": series, "value_label": "Importance (%)"},
        "table": table, "summary": summary,
    }


def structured(platform: Optional[str] = None, mode: str = "strongest", top_n: int = 8) -> dict:
    """Explicit-parameter counterpart to answer(). `mode`: strongest /
    weakest / consistency / compare_studies / summary (summary ignores
    `platform` and aggregates across all platforms)."""
    df = _load()
    if mode == "summary":
        return _summarize_all(top_n)
    if not platform or platform not in _platforms:
        return {"ok": False, "module": "rw_relweight", "error": "no_platform_match", "known": _platforms}

    subset = df[df["platform"] == platform]
    if subset.empty:
        return {"ok": False, "module": "rw_relweight", "error": "no_data", "platform": platform}

    if mode == "consistency":
        return _consistency(platform, subset, top_n)
    if mode == "compare_studies":
        return _compare_studies(platform, top_n)

    ascending = mode == "weakest"
    agg = _agg(subset)
    return _ranked_result(platform, agg, top_n, ascending=ascending, label="weakest" if ascending else "strongest")


def filters() -> dict:
    _load()
    return {"platforms": _platforms, "modes": ["strongest", "weakest", "consistency", "compare_studies", "summary"]}


def answer(question: str, top_n: int = 8) -> dict:
    df = _load()
    q = question.lower()

    if any(k in q for k in SUMMARY_KEYWORDS) and not best_match(question, _platforms, threshold=0.72):
        return _summarize_all(top_n)

    if is_compare_query(question) and "studies" in q:
        platform = best_match(question, _platforms, threshold=0.5) or (_platforms[0] if len(_platforms) == 1 else None)
        # try to find any platform mentioned even without high confidence
        if not platform:
            for p in _platforms:
                if p.lower() in q:
                    platform = p
                    break
        if platform:
            result = _compare_studies(platform, top_n)
            if result.get("ok"):
                return result

    platform = best_match(question, _platforms, threshold=0.6)
    if not platform:
        # No specific platform named — for strongest/weakest/consistent
        # DRIVER questions, fall back to aggregating across all platforms
        # combined. Require an explicit driver/RWA signal alongside
        # strongest/weakest so this doesn't fire on unrelated questions
        # that merely contain the word "strongest" or "weakest".
        has_driver_signal = "driver" in q or "rwa" in q
        if any(k in q for k in CONSISTENT_KEYWORDS) or (has_driver_signal and any(k in q for k in WEAKEST_KEYWORDS + STRONGEST_KEYWORDS)):
            all_platforms_label = "all hair care platforms"
            if any(k in q for k in CONSISTENT_KEYWORDS):
                return _consistency(all_platforms_label, df, top_n)
            agg = _agg(df)
            ascending = any(k in q for k in WEAKEST_KEYWORDS)
            return _ranked_result(all_platforms_label, agg, top_n, ascending=ascending, label="weakest" if ascending else "strongest")
        return {"ok": False, "module": "rw_relweight", "error": "no_platform_match", "known": _platforms}

    subset = df[df["platform"] == platform]
    if subset.empty:
        return {"ok": False, "module": "rw_relweight", "error": "no_data", "platform": platform}

    if any(k in q for k in CONSISTENT_KEYWORDS):
        return _consistency(platform, subset, top_n)

    agg = _agg(subset)
    if any(k in q for k in WEAKEST_KEYWORDS):
        return _ranked_result(platform, agg, top_n, ascending=True, label="weakest")

    return _ranked_result(platform, agg, top_n, ascending=False, label="strongest" if any(k in q for k in STRONGEST_KEYWORDS) else "top")
