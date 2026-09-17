"""
Thin wrapper around Vertex AI Gemini used to turn a module's structured
result into a natural-language summary.

Optional: if GCP_PROJECT / Vertex AI isn't configured, or the call fails,
the caller falls back to the module's own templated `summary` field.
The numbers, chart, and table NEVER depend on this call succeeding.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

USE_LLM_SUMMARY = os.environ.get("USE_LLM_SUMMARY", "true").lower() == "true"
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Bounds every call this client makes (summary rewrite + the agent router,
# which reuses this client). Without a timeout, a slow/hanging Gemini
# response ties up a Gunicorn worker thread indefinitely — with only 8
# threads per the Dockerfile's `--threads 8`, a handful of stuck calls is
# enough to make the whole service stop answering new requests.
GEMINI_TIMEOUT_MS = int(os.environ.get("GEMINI_TIMEOUT_MS", "20000"))

_client = None


def _detect_project() -> Optional[str]:
    """Resolve the GCP project ID from environment variables or Cloud Run metadata/ADC."""
    proj = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if proj:
        return proj
    try:
        import google.auth

        _, proj = google.auth.default()
        if proj:
            return proj
    except Exception as e:
        logger.debug("Could not auto-detect GCP project via google.auth: %s", e)
    return None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not USE_LLM_SUMMARY:
        return None

    try:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            logger.info("Initializing Google GenAI client with API key using model %s", GEMINI_MODEL)
            _client = genai.Client(api_key=api_key, http_options=genai.types.HttpOptions(timeout=GEMINI_TIMEOUT_MS))
            return _client

        project = _detect_project()
        if not project:
            logger.info("No GCP project or GEMINI_API_KEY found; LLM summaries will use fallback template.")
            return None

        logger.info(
            "Initializing Vertex AI GenAI client (project=%s, location=%s, model=%s)",
            project,
            GCP_LOCATION,
            GEMINI_MODEL,
        )
        _client = genai.Client(vertexai=True, project=project, location=GCP_LOCATION,
                                http_options=genai.types.HttpOptions(timeout=GEMINI_TIMEOUT_MS))
        return _client
    except Exception as e:
        logger.warning("Failed to initialize Google GenAI client: %s", e)
        return None


def summarize(question: str, result: dict) -> Optional[str]:
    """Return an LLM-written summary of a module result, or None if unavailable/failed."""
    if not USE_LLM_SUMMARY:
        return None
    client = _get_client()
    if client is None:
        return None

    table_rows = "\n".join(
        "- " + ", ".join(f"{k}={v}" for k, v in row.items()) for row in result.get("table", [])[:10]
    )

    prompt = f"""You are a consumer research analyst. The user asked: "{question}"

This was answered using the "{result.get('title', result.get('module'))}" analysis.
Templated baseline summary: {result.get('summary', '')}

Supporting data (top rows):
{table_rows}

Write a concise 2-4 sentence business summary in plain English. Do not invent
numbers beyond what is given. No markdown headers."""

    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (resp.text or "").strip() or None
    except Exception as e:
        logger.warning("Failed to generate LLM summary using model %s: %s", GEMINI_MODEL, e)
        return None
