"""
context: minimal in-process conversation memory so short follow-up
questions ("what about the Fragrance stage?", "and BARRIER?") can inherit
the entities (benefit, product, platform, category...) resolved in the
previous turn, instead of every request having to be fully self-contained.

In-memory only, keyed by a client-generated session_id: state resets on
redeploy and isn't shared across Cloud Run instances/replicas. That's an
acceptable trade-off for a lightweight agent; a multi-instance deployment
would move this dict to Firestore/Memorystore keyed the same way.
"""
from __future__ import annotations

import threading
import time

_LOCK = threading.Lock()
_SESSIONS: dict[str, dict] = {}
_TTL_SECONDS = 60 * 60  # sessions idle for an hour are forgotten
_MAX_SESSIONS = 5000  # crude cap so a long-running instance can't leak memory forever

_SLOT_KEYS = (
    "benefit", "product", "platform", "category", "stage", "stage_code",
    "compare_benefit", "compare_product",
)

# How many prior (question, answer-summary) turns the LLM router gets to see
# verbatim, on top of the slot-based entity hint above. Kept short — this is
# real conversation memory (used for actual multi-turn reasoning by the LLM
# router), not just entity carry-over, so it costs prompt tokens per turn.
_HISTORY_MAX_TURNS = 6


def get_context(session_id: str | None) -> dict:
    """Known entities from the session's previous turn, or {} if there's
    no session, nothing stored yet, or it's gone stale."""
    if not session_id:
        return {}
    with _LOCK:
        entry = _SESSIONS.get(session_id)
        if not entry:
            return {}
        if time.time() - entry["_ts"] > _TTL_SECONDS:
            _SESSIONS.pop(session_id, None)
            return {}
        return {k: v for k, v in entry.items() if k not in ("_ts", "_history")}


def update_context(session_id: str | None, result: dict, question: str | None = None) -> None:
    """Replace (not merge) the session's stored entities with whatever this
    turn actually resolved — so a slot from two turns ago can't silently
    leak into a much later, unrelated follow-up.

    Separately, *append* (question, answer-summary) to a short rolling
    history so multi-turn callers (currently the LLM router) can see the
    actual conversation, not just the last turn's resolved entities.
    History is additive across turns within a session; the entity slots
    above stay "last turn only" on purpose (see the replace-not-merge note)."""
    if not session_id:
        return
    slots = {k: result[k] for k in _SLOT_KEYS if result.get(k)}
    slots["module"] = result.get("module")
    slots["_ts"] = time.time()
    with _LOCK:
        if session_id not in _SESSIONS and len(_SESSIONS) >= _MAX_SESSIONS:
            oldest = min(_SESSIONS, key=lambda k: _SESSIONS[k]["_ts"])
            _SESSIONS.pop(oldest, None)
        prior_history = _SESSIONS.get(session_id, {}).get("_history", [])
        if question and result.get("ok"):
            turn = {"question": question, "module": result.get("module"),
                     "summary": result.get("summary")}
            prior_history = (prior_history + [turn])[-_HISTORY_MAX_TURNS:]
        slots["_history"] = prior_history
        _SESSIONS[session_id] = slots


def get_history(session_id: str | None) -> list[dict]:
    """The session's last few (question, module, summary) turns, oldest
    first, or [] if there's no session or nothing stored yet."""
    if not session_id:
        return []
    with _LOCK:
        entry = _SESSIONS.get(session_id)
        if not entry or time.time() - entry.get("_ts", 0) > _TTL_SECONDS:
            return []
        return list(entry.get("_history", []))


def clear_context(session_id: str | None) -> None:
    if not session_id:
        return
    with _LOCK:
        _SESSIONS.pop(session_id, None)


def context_hint_text(ctx: dict) -> str:
    """Known entities formatted as trailing words a follow-up question can
    be augmented with, so each module's own fuzzy matcher can pick the
    missing entity back up without any module-specific plumbing.

    product + compare_product are joined with "and" specifically so a
    pronoun-only continuation ("why are they similar?") re-augmented with
    this hint still parses as a two-entity compare question, not just a
    bag of loose words."""
    bits = []
    if ctx.get("product") and ctx.get("compare_product"):
        bits.append(f"{ctx['product']} and {ctx['compare_product']}")
    elif ctx.get("product"):
        bits.append(str(ctx["product"]))
    for key in ("benefit", "platform", "category"):
        if ctx.get(key):
            bits.append(str(ctx[key]))
    return " ".join(bits)
