"""
llm_router: an optional LLM-driven "agent" front end.

Instead of the deterministic keyword/fuzzy router in router.py deciding
which analysis module to call, Gemini sees short descriptions of the six
analysis modules as callable tools, decides which one(s) the question
needs — possibly several, across a few rounds, when later calls depend on
what earlier ones returned — and writes the final natural-language answer
from what they actually returned. It can also ask the user a clarifying
question instead of guessing, when the question is genuinely ambiguous.

Hard safety rule: the chart, table, and every number shown to the user
always come straight from the same deterministic module.answer() calls
used elsewhere in this app. The LLM only (a) decides which module(s) to
call and with what sub-question, (b) decides whether to ask the user for
clarification instead, and (c) narrates the result in prose. It never
fills in a data number itself.

route() returns None — meaning "use the deterministic router instead" —
whenever:
  - USE_LLM_ROUTER is off (the default; this changes core routing
    behavior and costs an extra LLM round trip, so it's opt-in), or
  - no Gemini client is configured (no GEMINI_API_KEY / GCP project), or
  - the call fails for any reason, or
  - no tool call ever returns an ('ok': True) result and the model never
    asked for clarification either.

route() returns a dict with `needs_clarification: True` (and no chart/
table — there's nothing to show yet) when the model chose to ask the user
a question instead of guessing. Callers should surface that question to
the user rather than silently falling back to the deterministic router,
since a wrong silent guess is worse than one extra round-trip.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from . import router as deterministic_router
from .context import context_hint_text
from .gemini_client import GEMINI_MODEL, _get_client

logger = logging.getLogger(__name__)

USE_LLM_ROUTER = os.environ.get("USE_LLM_ROUTER", "True").lower() == "true"

# A "round" is one call to generate_content; the model can return several
# tool calls within a single round (handled in parallel below) and/or the
# loop can run several rounds in sequence when a later call depends on an
# earlier one's result. Raised from 3 -> 6 so genuinely multi-part
# questions ("compare the top driver of X, Y, and Z, then profile whichever
# product scores best on the winner") have room to resolve without being
# cut off mid-reasoning.
MAX_TOOL_ROUNDS = int(os.environ.get("LLM_ROUTER_MAX_ROUNDS", "6"))

# Safety valve independent of rounds: even with parallel calls per round,
# cap total tool invocations so a confused model can't loop indefinitely
# and run up Gemini + Cloud Run cost on one request.
MAX_TOTAL_TOOL_CALLS = int(os.environ.get("LLM_ROUTER_MAX_TOOL_CALLS", "8"))

_TOOL_DOCS = {
    "sr_attribute": (
        "Which sensory ATTRIBUTES (e.g. spreadability, absorption, fragrance intensity, "
        "gloss, greasiness) drive a given benefit. Use for 'what attributes drive X', 'top "
        "attributes for X', optionally narrowed to one benefit-stage category (Skin Feel, "
        "Fingertip, Skin Look, or Fragrance) or broken down 'by benefit stage', or to compare "
        "two benefits' attribute drivers."
    ),
    "sd_driver": (
        "Which TEST STAGE (one of ten CLT test-wave groups, e.g. 'CLT Fingertip', 'Fragrance "
        "10min') is the strongest driver of a benefit — coarser-grained than sr_attribute. Also "
        "answers 'which benefits does test stage X drive most', and can be filtered/broken down "
        "the same way as sr_attribute (by category or benefit stage)."
    ),
    "bh_blindhut": (
        "Blind-hut consumer liking SCORES per product, per benefit. Use for 'which products "
        "score highest on X', 'how does product Y score across benefits', or comparing two "
        "products' scores."
    ),
    "pc_perceptual": (
        "Perceptual map (PCA) positioning of products within a category: nearest neighbours, "
        "the closest or most different pair, clusters, or whitespace gaps."
    ),
    "rw_relweight": (
        "Relative-weight driver IMPORTANCE for hair-care platforms (shampoo, conditioner, "
        "scalp). Strongest/weakest driver, or how consistent a driver is across studies."
    ),
    "spider_profile": (
        "Full attribute PROFILE (radar) of one product or brand: its strongest/weakest "
        "attributes, what's distinctive about it, or comparing two profiles."
    ),
}

# A pseudo-tool: not a real analysis module, just the model's way of
# stopping to ask the user something instead of guessing. Handled specially
# in the tool-call loop below rather than routed through
# deterministic_router.MODULES.
_CLARIFY_TOOL_NAME = "ask_clarifying_question"

# Two more pseudo-tools: unlike the clarify tool, these don't end the turn —
# the model can call either (or both) alongside a normal data tool, in the
# same round, to express a formatting judgment about the question rather
# than just its data content. Keeping them as tools (not a second prompt)
# means this costs zero extra latency/API calls: the model decides
# everything it's going to decide in the same pass it already makes to
# pick which module(s) to call.
_CHART_TOOL_NAME = "set_chart_type"
_EXEC_SUMMARY_TOOL_NAME = "request_executive_summary"

# Only types that are ever safe to swap onto a bar-shaped result — see
# app/main.py::_apply_format_requests, which is the single place that
# actually performs (and validates) the swap. Scatter (perceptual map) and
# radar (attribute profile) are structurally fixed and never appear here;
# if the model tries to set one of those anyway, _apply_format_requests
# rejects it the same way it rejects a user's literal "...as a pie chart"
# request on incompatible data, and adds an explanatory note instead of
# silently failing.
_CHART_TYPE_OPTIONS = ["bar", "line", "pie", "doughnut"]


def _build_tools():
    from google.genai import types

    decls = [
        types.FunctionDeclaration(
            name=name,
            description=desc,
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "sub_question": {
                        "type": "string",
                        "description": (
                            "A complete, self-contained natural-language question for this "
                            "module. Name the benefit/product/platform/category explicitly — "
                            "no pronouns like 'it' or 'that'."
                        ),
                    }
                },
                "required": ["sub_question"],
            },
        )
        for name, desc in _TOOL_DOCS.items()
    ]

    decls.append(
        types.FunctionDeclaration(
            name=_CLARIFY_TOOL_NAME,
            description=(
                "Call this INSTEAD of any analysis tool when the question is genuinely "
                "ambiguous between two or more different meanings and guessing wrong would "
                "give the user a misleading chart (e.g. a product name that exists in more "
                "than one dataset, or a benefit that could mean the attribute-level or "
                "test-stage-level driver analysis, or a vague 'how are we doing' with no "
                "named benefit/product at all). Do NOT call this for questions that are just "
                "loosely worded but have one clearly intended meaning — resolve those "
                "yourself. This ends the turn with your question shown to the user instead "
                "of a chart, so use it sparingly."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "question_for_user": {
                        "type": "string",
                        "description": (
                            "One short, specific question that names the concrete options "
                            "the user should pick between (e.g. 'Did you mean the attribute-"
                            "level drivers of Moisturization, or the test-stage drivers?')."
                        ),
                    }
                },
                "required": ["question_for_user"],
            },
        )
    )
    decls.append(
        types.FunctionDeclaration(
            name=_CHART_TOOL_NAME,
            description=(
                "Optional. Call this AT MOST ONCE, in the same round as your analysis tool "
                "call(s), only if a different chart type would genuinely communicate this "
                "specific answer better than the module's default — e.g. the question is about "
                "a trend/progression ('line'), a share of a whole ('pie' or 'doughnut'), or a "
                "ranked comparison ('bar'). Skip this entirely for most questions; only use it "
                "when the question's own wording implies a specific chart shape, even if it "
                "never names a chart type outright (e.g. 'what share of total liking comes from "
                "Instant glow?' implies 'pie', not just a bar). Has no effect when the "
                "underlying result is a perceptual map or an attribute-profile radar — those "
                "always keep their native chart type regardless of what you set here."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "chart_type": {"type": "string", "enum": _CHART_TYPE_OPTIONS},
                },
                "required": ["chart_type"],
            },
        )
    )

    decls.append(
        types.FunctionDeclaration(
            name=_EXEC_SUMMARY_TOOL_NAME,
            description=(
                "Optional. Call this AT MOST ONCE, in the same round as your analysis tool "
                "call(s), if the person is asking for a fuller, leadership-facing write-up "
                "rather than a normal short answer — e.g. they mention a manager, leadership, "
                "stakeholders, a deck, or ask you to 'write this up' or 'summarize for the "
                "team', even if they never say the words 'executive summary'. Skip it for "
                "ordinary questions."
            ),
            parameters_json_schema={"type": "object", "properties": {}},
        )
    )
    return [types.Tool(function_declarations=decls)]


def _call_module_tool(name: str, sub_question: str, top_n: int) -> dict:
    module = deterministic_router.MODULES.get(name)
    if module is None:
        return {"ok": False, "error": f"unknown_module:{name}"}
    try:
        return module.answer(sub_question, top_n=top_n)
    except Exception as e:
        logger.warning("LLM router tool %s raised on %r: %s", name, sub_question, e)
        return {"ok": False, "error": "tool_exception"}


def _tool_result_for_llm(result: dict) -> dict:
    """A compact view of a tool result to feed back to the model — no need
    to round-trip the full chart payload."""
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "no_data")}
    return {
        "ok": True,
        "title": result.get("title"),
        "summary": result.get("summary"),
        "table_preview": result.get("table", [])[:6],
    }


def _finish(last_ok_result: dict, used_modules: list[str], narrative: Optional[str] = None,
            agent_chart_type: Optional[str] = None, agent_wants_exec_summary: bool = False) -> dict:
    """The chart/table/entities come from the last tool call that actually
    succeeded; the prose is the LLM's synthesis (or that tool's own
    templated summary, if the LLM never produced a closing message)."""
    out = dict(last_ok_result)
    if narrative:
        out["summary"] = narrative
    out["routed_by"] = "llm"
    out["tools_used"] = used_modules
    # These are judgments, not data — app/main.py runs them back through
    # the same _apply_format_requests() safety checks used for a user's
    # own literal "...as a pie chart" request, so an unsuitable choice
    # here (e.g. pie on a multi-series comparison) is caught the same way,
    # not applied blindly just because the agent asked for it.
    if agent_chart_type:
        out["agent_chart_type"] = agent_chart_type
    if agent_wants_exec_summary:
        out["agent_wants_exec_summary"] = True
    return out


def _needs_clarification(question_for_user: str) -> dict:
    return {
        "ok": False,
        "needs_clarification": True,
        "clarifying_question": question_for_user,
        "routed_by": "llm",
    }


def _history_block(history: list[dict]) -> str:
    """Prior turns as real conversation, not just the last turn's resolved
    entities — lets the model handle follow-ups ("and for Fragrance?",
    "what about the runner-up?") by reading what was actually asked and
    answered, rather than through slot-filling alone."""
    if not history:
        return ""
    lines = []
    for turn in history:
        q = turn.get("question")
        s = turn.get("summary")
        if q and s:
            lines.append(f'- Asked: "{q}" -> Answered ({turn.get("module")}): {s}')
    if not lines:
        return ""
    return "\nRecent conversation in this session, oldest first:\n" + "\n".join(lines)


def route(
    question: str,
    top_n: int = 8,
    context: Optional[dict] = None,
    history: Optional[list[dict]] = None,
) -> Optional[dict]:
    if not USE_LLM_ROUTER:
        return None
    client = _get_client()
    if client is None:
        # This is the single most common cause of "agentic mode isn't doing
        # anything" reports: USE_LLM_ROUTER=true only enables the feature —
        # it says nothing about whether Gemini credentials actually
        # resolved. Without this line, main.py's silent-and-safe fallback
        # to the deterministic router (by design, for the end user) also
        # meant zero server-side trace that agentic mode was even attempted.
        # GET /api/agent-status surfaces the same check on demand.
        logger.warning(
            "USE_LLM_ROUTER is on but no Gemini client is available (no "
            "GEMINI_API_KEY/GOOGLE_API_KEY and no GCP project/ADC detected) — "
            "falling back to the deterministic router for this request. "
            "Check GET /api/agent-status."
        )
        return None

    try:
        from google.genai import types
    except Exception as e:
        logger.warning("google-genai not available for LLM router: %s", e)
        return None

    context_note = ""
    if context:
        context_note = (
            f"\nFor reference, the user's previous turn was about: {context_hint_text(context)} "
            f"(module: {context.get('module')}). Only lean on this if the new question is clearly "
            "a continuation of that topic."
        )
    context_note += _history_block(history or [])

    instructions = (
        "You are the routing brain for a consumer sensory-research assistant with six analysis "
        "tools below, plus ask_clarifying_question, set_chart_type, and request_executive_summary. "
        "Call exactly one analysis tool for a simple question. Call more tools — in the same round "
        "if they're independent of each other, or in a following round if a later call needs an "
        "earlier result — when the question genuinely has multiple linked or independent parts. "
        "This includes questions that name TWO different things to look up even when phrased as one "
        "sentence: e.g. \"what's the top driver of X, and how does that same stage rank for Y\" needs "
        "sd_driver called twice; \"what drives X, and how does PRODUCT score on X\" needs a driver "
        "tool (sr_attribute or sd_driver) AND bh_blindhut — do not stop after the first half just "
        "because it produced a usable answer on its own; check whether the question asked for a "
        "second thing before finishing. If, after reading the question and the recent conversation "
        "below, you're genuinely unsure which dataset or which of two named entities is meant, call "
        "ask_clarifying_question instead of guessing — do not also call an analysis tool in that "
        "same turn. In the same round as your analysis tool call(s), also call set_chart_type and/or "
        "request_executive_summary if either genuinely applies to this question (see their own "
        "descriptions) — most questions need neither, so only call them when they clearly fit. "
        "Otherwise, once you have everything the question asked for, reply with a final plain-English "
        "answer, 2-5 sentences, no markdown, covering every part of the question — every number in it "
        f"must come from a tool result, never invented.{context_note}\n\nUser question: {question}"
    )

    contents = [types.Content(role="user", parts=[types.Part(text=instructions)])]
    config = types.GenerateContentConfig(
        tools=_build_tools(),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    last_ok_result: Optional[dict] = None
    used_modules: list[str] = []
    total_calls = 0
    agent_chart_type: Optional[str] = None
    agent_wants_exec_summary = False

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
        except Exception as e:
            logger.warning("LLM router generate_content failed: %s", e)
            return _finish(last_ok_result, used_modules, agent_chart_type=agent_chart_type,
                            agent_wants_exec_summary=agent_wants_exec_summary) if last_ok_result else None

        calls = resp.function_calls
        if not calls:
            final_text = (resp.text or "").strip()
            return _finish(last_ok_result, used_modules, narrative=final_text,
                            agent_chart_type=agent_chart_type,
                            agent_wants_exec_summary=agent_wants_exec_summary) if last_ok_result else None

        contents.append(resp.candidates[0].content)
        response_parts = []
        for fc in calls:
            if fc.name == _CLARIFY_TOOL_NAME:
                # The model chose to ask rather than guess: stop immediately
                # and surface its question. Any other calls in this same
                # round are ignored — the model was told not to mix the two,
                # but if it does anyway, asking wins over a possibly-wrong guess.
                q_for_user = (fc.args or {}).get("question_for_user") or (
                    "Could you clarify which product, benefit, or dataset you mean?"
                )
                return _needs_clarification(q_for_user)

            if fc.name == _CHART_TOOL_NAME:
                # A formatting judgment, not a data call — record it and give
                # the model a trivial ack so it doesn't treat this as a
                # missing tool result, then move on. No module is invoked.
                ct = (fc.args or {}).get("chart_type")
                if ct in _CHART_TYPE_OPTIONS:
                    agent_chart_type = ct
                response_parts.append(
                    types.Part(function_response=types.FunctionResponse(
                        name=fc.name, response={"result": {"ok": True, "noted": True}},
                    ))
                )
                continue

            if fc.name == _EXEC_SUMMARY_TOOL_NAME:
                agent_wants_exec_summary = True
                response_parts.append(
                    types.Part(function_response=types.FunctionResponse(
                        name=fc.name, response={"result": {"ok": True, "noted": True}},
                    ))
                )
                continue

            total_calls += 1
            sub_q = (fc.args or {}).get("sub_question") or question
            result = _call_module_tool(fc.name, sub_q, top_n)
            if result.get("ok"):
                last_ok_result = result
                used_modules.append(fc.name)
            response_parts.append(
                types.Part(function_response=types.FunctionResponse(
                    name=fc.name, response={"result": _tool_result_for_llm(result)},
                ))
            )
            if total_calls >= MAX_TOTAL_TOOL_CALLS:
                break
        contents.append(types.Content(role="user", parts=response_parts))
        if total_calls >= MAX_TOTAL_TOOL_CALLS:
            break

    return _finish(last_ok_result, used_modules, agent_chart_type=agent_chart_type,
                    agent_wants_exec_summary=agent_wants_exec_summary) if last_ok_result else None
