"""
exec_summary: on-demand, more thorough narrative for any agent's result —
a headline, supporting points pulled from the table data, and (when
Gemini is configured) a business-facing closing note. Works across all
six agents without any agent-specific code, since it only reads the
generic result shape every module already returns (title/summary/table).

Never blocks the response: if no LLM client is configured, or the call
fails, falls back to a structured template built straight from the
table — the person still gets something useful.
"""
from __future__ import annotations

import logging
from typing import Optional

from .gemini_client import GEMINI_MODEL, _get_client

logger = logging.getLogger(__name__)


def _template_summary(result: dict) -> str:
    table = result.get("table") or []
    title = result.get("title", "Result")
    lines = [result.get("summary", "").strip()]
    if table:
        lines.append("")
        lines.append("Key figures:")
        for row in table[:5]:
            lines.append("- " + ", ".join(f"{k}: {v}" for k, v in row.items()))
    if result.get("suggestions"):
        lines.append("")
        lines.append("Worth exploring next: " + "; ".join(result["suggestions"][:3]))
    return "\n".join(l for l in lines if l is not None)


def generate(result: dict, question: str) -> str:
    client = _get_client()
    if client is None:
        return _template_summary(result)

    try:
        table_preview = result.get("table", [])[:8]
        prompt = (
            "Write a short executive summary (4-6 sentences, plain prose, no markdown headers or "
            "bullet lists) of the analysis result below, for a business stakeholder who will not "
            "see the underlying chart. Lead with the headline finding, then 2-3 supporting points "
            "grounded only in the data given, then one practical implication or next step. Never "
            f"invent a number that isn't present below.\n\nOriginal question: {question}\n"
            f"Title: {result.get('title')}\nCurrent summary: {result.get('summary')}\n"
            f"Table data (subset): {table_preview}"
        )
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = (resp.text or "").strip()
        return text or _template_summary(result)
    except Exception as e:
        logger.warning("Executive summary LLM generation failed: %s", e)
        return _template_summary(result)
