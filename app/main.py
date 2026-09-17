import logging
import os
import sys
import uuid

from flask import Flask, jsonify, request, send_from_directory

try:
    from . import context as session_context
    from . import llm_router
    from . import router as agent_router
    from .agents import list_agents
    from .example_queries import examples_for
    from .exec_summary import generate as generate_exec_summary
    from .gemini_client import summarize
    from .modules.utils import detect_requested_chart_type, wants_executive_summary, wants_table_only
except ImportError:
    # Handle direct execution (e.g. python app/main.py)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from app import context as session_context
    from app import llm_router
    from app import router as agent_router
    from app.agents import list_agents
    from app.example_queries import examples_for
    from app.exec_summary import generate as generate_exec_summary
    from app.gemini_client import summarize
    from app.modules.utils import detect_requested_chart_type, wants_executive_summary, wants_table_only

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")

# Preload all datasets upon server startup to eliminate cold-start latency in Cloud Run
try:
    agent_router.preload()
    logger.info("Successfully preloaded all dataset modules.")
except Exception as e:
    logger.warning("Dataset preloading encounter an issue (will load lazily on first query): %s", e)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/agent-status")
def agent_status():
    """Diagnostic endpoint for exactly the failure mode where USE_LLM_ROUTER=true
    but Gemini isn't actually reachable, so every request silently falls back
    to the deterministic router with no visible sign anything is wrong. Hit
    this directly to see, in one request, whether agentic routing can
    actually run right now — instead of inferring it from whether a
    particular answer happened to chain multiple tools.

    By default this only checks that a Gemini client *object* resolves —
    that's necessary but NOT sufficient, since llm_router.route() swallows
    any exception from the actual API call (auth/permission errors, wrong
    region, model access not granted, network egress blocked, etc.) and
    silently falls back, exactly like a missing credential would. Pass
    ?live=true to instead make one real, tiny call to Gemini and report
    whether it actually succeeds — the only way to be certain, short of
    reading server logs.
    """
    from . import gemini_client
    client = gemini_client._get_client()
    out = {
        "ok": True,
        "use_llm_router": llm_router.USE_LLM_ROUTER,
        "gemini_client_available": client is not None,
        "gemini_model": gemini_client.GEMINI_MODEL,
        "agentic_mode_active": llm_router.USE_LLM_ROUTER and client is not None,
        "note": (
            "agentic_mode_active must be true for cross-dataset chaining to work at all. "
            "If use_llm_router is true but gemini_client_available is false, set "
            "GEMINI_API_KEY (or GOOGLE_API_KEY), or GOOGLE_CLOUD_PROJECT with working ADC. "
            "If agentic_mode_active is true but real queries still don't chain, pass "
            "?live=true to this endpoint to test an actual Gemini call, not just client setup."
        ),
    }

    if request.args.get("live", "").lower() in ("1", "true", "yes") and client is not None:
        try:
            resp = client.models.generate_content(
                model=gemini_client.GEMINI_MODEL,
                contents="Reply with exactly the word: ok",
            )
            out["live_call_ok"] = True
            out["live_call_response"] = (resp.text or "").strip()[:200]
        except Exception as e:
            # This is the exact failure llm_router.route() swallows silently
            # on every real request — surfacing it here, once, on demand,
            # is what turns "agentic mode looks configured" into "agentic
            # mode actually works."
            out["live_call_ok"] = False
            out["live_call_error"] = f"{type(e).__name__}: {e}"

    return jsonify(out)


@app.get("/api/presets")
def presets():
    return jsonify({
        "presets": [
            "What attributes drive Moisturization?",
            "Compare Moisturization and Instant Glow.",
            "What drives Overall Opinion?",
            "Which products score highest on Instant glow?",
            "Show product clusters for SUNCARE",
            "What are the weakest RWA drivers?",
            "Compare Anessa and Ahc",
        ]
    })


def _apply_format_requests(result: dict, question: str = "", executive_summary: bool = False,
                            table_only: bool = False, requested_chart_type: str | None = None) -> str | None:
    """Apply chart-type swap / table-only / executive-summary formatting
    uniformly across both the free-text and structured endpoints. Mutates
    `result` in place (adding `suppress_chart` / `executive_summary` as
    needed, swapping `chart['type']` when safe) and returns a note string
    to append to the summary, or None.

    Honors an explicit chart-type request (e.g. "...as a line chart") when
    the underlying data is categorical (labels+values) — bar/line/pie/
    doughnut all share that shape so swapping is safe. Scatter (perceptual
    map) and radar (attribute profile) have a structurally different data
    shape, so a request for e.g. "pie chart" there is noted, not honored,
    since it would misrepresent the data. Multi-series comparisons also
    cannot cleanly render as pie/doughnut. "Table only" always wins over a
    chart-type request, since there's nothing to swap if nothing renders."""
    note = None
    if table_only:
        result["suppress_chart"] = True
    elif requested_chart_type:
        chart = result.get("chart", {})
        if chart.get("type"):
            is_single_series = "values" in chart and "series" not in chart
            if requested_chart_type in ("pie", "doughnut") and not is_single_series:
                note = (
                    f"A {requested_chart_type} chart requires a single series, but this "
                    f"query compares multiple series — displaying as {chart['type']} instead."
                )
            elif chart["type"] in ("bar", "line", "pie", "doughnut"):
                chart["type"] = requested_chart_type
            else:
                note = (
                    f"This data is a {chart['type']} by nature "
                    f"(product positions / multi-attribute profile), so a "
                    f"{requested_chart_type} chart wouldn't represent it accurately — "
                    f"showing it as a {chart['type']} instead."
                )

    if executive_summary:
        result["executive_summary"] = generate_exec_summary(result, question)

    return note


@app.get("/api/agents")
def get_agents():
    return jsonify({"ok": True, "agents": list_agents()})


@app.get("/api/examples/<module_name>")
def get_examples(module_name):
    if module_name not in agent_router.MODULES:
        return jsonify({"ok": False, "error": "unknown_module"}), 404
    return jsonify({"ok": True, "module": module_name, "examples": examples_for(module_name)})


@app.get("/api/filters/<module_name>")
def get_filters(module_name):
    module = agent_router.MODULES.get(module_name)
    if module is None or not hasattr(module, "filters"):
        return jsonify({"ok": False, "error": "unknown_module"}), 404
    return jsonify({"ok": True, "module": module_name, "filters": module.filters()})


@app.post("/api/structured/<module_name>")
def structured_query(module_name):
    module = agent_router.MODULES.get(module_name)
    if module is None or not hasattr(module, "structured"):
        return jsonify({"ok": False, "error": "unknown_module"}), 404

    body = request.get_json(silent=True) or {}
    control_keys = ("session_id", "executive_summary", "table_only", "chart_type")
    params = {k: v for k, v in body.items() if k not in control_keys}
    session_id = (body.get("session_id") or "").strip()[:128] or str(uuid.uuid4())

    try:
        result = module.structured(**params)
    except TypeError as e:
        return jsonify({"ok": False, "error": "bad_params", "detail": str(e), "session_id": session_id}), 400

    if result.get("ok"):
        note = _apply_format_requests(
            result,
            executive_summary=bool(body.get("executive_summary")),
            table_only=bool(body.get("table_only")),
            requested_chart_type=body.get("chart_type"),
        )
        if note:
            result["summary"] = f"{result.get('summary','')} {note}".strip()
        result["suggestions"] = agent_router.followups(module_name, result)
        session_context.update_context(session_id, result)
    result["session_id"] = session_id
    return jsonify(result)


MAX_QUESTION_CHARS = 500
MIN_TOP_N, MAX_TOP_N, DEFAULT_TOP_N = 1, 25, 8


def _parse_top_n(raw) -> int:
    """Clamp top_n to a sane range instead of trusting the client — an
    unbounded value here would let a request ask pandas/Chart.js to render
    an arbitrarily large table/chart, and (in agentic mode) inflate the
    tokens sent back to Gemini for every tool result."""
    try:
        n = int(raw) if raw not in (None, "") else DEFAULT_TOP_N
    except (TypeError, ValueError):
        return DEFAULT_TOP_N
    return max(MIN_TOP_N, min(MAX_TOP_N, n))


@app.route("/api/query", methods=["POST", "OPTIONS"])
def query():
    if request.method == "OPTIONS":
        return "", 204

    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    top_n = _parse_top_n(body.get("top_n"))
    session_id = (body.get("session_id") or "").strip()[:128] or str(uuid.uuid4())
    pinned_module = (body.get("module") or "").strip() or None
    hint_module = (body.get("hint_module") or "").strip() or None

    if not question:
        return jsonify({"ok": False, "error": "Missing 'question' in request body.", "session_id": session_id}), 400
    if len(question) > MAX_QUESTION_CHARS:
        return jsonify({
            "ok": False,
            "error": f"Question is too long (max {MAX_QUESTION_CHARS} characters).",
            "session_id": session_id,
        }), 400

    prior_context = session_context.get_context(session_id)
    prior_history = session_context.get_history(session_id)

    if pinned_module:
        # Tab-scoped mode: skip cross-dataset routing (and the LLM router,
        # which is also a cross-dataset decision) entirely — the person has
        # already told us which specialist to use.
        if pinned_module not in agent_router.MODULES:
            return jsonify({"ok": False, "error": "unknown_module", "session_id": session_id}), 400
        result = agent_router.answer_pinned(question, pinned_module, top_n=top_n, context=prior_context)
        llm_routed = False
    elif hint_module and hint_module in agent_router.MODULES:
        # Fast path for suggestion-chip follow-ups: the chip text was
        # generated FROM this exact module's own entity lists (see
        # router.py::_followups), so it is never actually ambiguous —
        # sending it through the full agentic tool-call loop again would
        # just add 1-2 extra sequential Gemini round trips for a question
        # we already know how to answer. This differs from `module`
        # (pinned_module) above in one way: it's a one-request hint, not a
        # sticky pin, so a freely-typed next question still goes through
        # full routing. If the fast path can't answer it for any reason
        # (stale/unexpected text), fall through to the normal agentic/
        # deterministic path below rather than surfacing a dead end.
        result = agent_router.answer_pinned(question, hint_module, top_n=top_n, context=prior_context)
        llm_routed = False
        if not result.get("ok"):
            result = llm_router.route(question, top_n=top_n, context=prior_context, history=prior_history)
            llm_routed = result is not None
            if not llm_routed:
                result = agent_router.answer_with_context(question, top_n=top_n, context=prior_context)
    else:
        result = llm_router.route(question, top_n=top_n, context=prior_context, history=prior_history)
        llm_routed = result is not None

        if llm_routed and result.get("needs_clarification"):
            # The LLM router chose to ask rather than guess. Surface that
            # question directly — do NOT silently fall back to the
            # deterministic router here, since that would hide the model's
            # explicit signal that it wasn't confident and hand back a
            # possibly-wrong guess instead of the question it actually
            # wanted answered.
            return jsonify({
                "ok": False,
                "needs_clarification": True,
                "error": result["clarifying_question"],
                "clarifying_question": result["clarifying_question"],
                "session_id": session_id,
            })

        if not llm_routed:
            result = agent_router.answer_with_context(question, top_n=top_n, context=prior_context)

    if not result.get("ok"):
        return jsonify({
            "ok": False,
            "error": "Could not match your question to a known benefit, product, platform, or category.",
            "detail": result,
            "session_id": session_id,
        })

    if llm_routed:
        result.setdefault("suggestions", agent_router.followups(result.get("module"), result))

    session_context.update_context(session_id, result, question=question)

    requested_type = detect_requested_chart_type(question)
    table_only = wants_table_only(question)
    exec_summary_wanted = wants_executive_summary(question)

    # Agentic mode can also infer either of these from the question's
    # semantics (not just literal phrasing) via the set_chart_type /
    # request_executive_summary tools in llm_router.py. The user's own
    # literal wording still wins if both are present — an explicit
    # "...as a pie chart" should never be second-guessed by the agent's
    # own judgment call.
    if llm_routed:
        requested_type = requested_type or result.get("agent_chart_type")
        exec_summary_wanted = exec_summary_wanted or bool(result.get("agent_wants_exec_summary"))

    note = _apply_format_requests(
        result, question=question, executive_summary=exec_summary_wanted,
        table_only=table_only, requested_chart_type=requested_type,
    )

    summary = result.get("summary", "")
    if not llm_routed:
        # For deterministic-router answers, an LLM pass turns the templated
        # summary into nicer prose. LLM-routed answers already carry the
        # LLM's own synthesized narrative, so skip the extra round trip.
        summary = summarize(question, result) or summary
    result["summary"] = summary + (f" {note}" if note else "")
    result["session_id"] = session_id
    return jsonify(result)


@app.post("/api/reset")
def reset_session():
    body = request.get_json(silent=True) or {}
    session_id = (body.get("session_id") or "").strip()
    session_context.clear_context(session_id)
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
