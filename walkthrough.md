# Walkthrough: Fixing the Flask Agent for Google Cloud Run

The codebase in [`flask-cloud-run`](file:///C:/Users/Praveen.Bhat/.gemini/antigravity/scratch/flask-cloud-run) has been fixed and enhanced to run reliably on Google Cloud Run, supporting both container image deployments and Cloud Buildpacks.

## Key Changes Made

### 1. Gemini Client Fixes & Auto-Detection
- **Valid Model**: Updated [`app/gemini_client.py`](file:///C:/Users/Praveen.Bhat/.gemini/antigravity/scratch/flask-cloud-run/app/gemini_client.py#L19) default model from the invalid `gemini-2.5-flash` (which failed with 404) to `gemini-2.0-flash`.
- **Cloud Run Project Discovery**: Added auto-detection via `google.auth.default()` to discover the GCP project ID from the Cloud Run metadata server when `GOOGLE_CLOUD_PROJECT` or `GCP_PROJECT` is not explicitly set as an environment variable.
- **API Key Support**: Added support for `GEMINI_API_KEY` / `GOOGLE_API_KEY` in addition to Vertex AI ADC.
- **Observability**: Added standard logging via `logging.getLogger(__name__)` so errors and client initialization are visible in Google Cloud Logging instead of being silently swallowed.

### 2. Thread Safety & Server Warmup
- **Thread Locking**: Added `threading.Lock()` double-checked locking in all 6 analysis modules:
  - [`app/modules/sr_attribute.py`](file:///C:/Users/Praveen.Bhat/.gemini/antigravity/scratch/flask-cloud-run/app/modules/sr_attribute.py#L25-L35)
  - [`app/modules/sd_driver.py`](file:///C:/Users/Praveen.Bhat/.gemini/antigravity/scratch/flask-cloud-run/app/modules/sd_driver.py#L35-L45)
  - [`app/modules/bh_blindhut.py`](file:///C:/Users/Praveen.Bhat/.gemini/antigravity/scratch/flask-cloud-run/app/modules/bh_blindhut.py#L27-L38)
  - [`app/modules/pc_perceptual.py`](file:///C:/Users/Praveen.Bhat/.gemini/antigravity/scratch/flask-cloud-run/app/modules/pc_perceptual.py#L40-L50)
  - [`app/modules/rw_relweight.py`](file:///C:/Users/Praveen.Bhat/.gemini/antigravity/scratch/flask-cloud-run/app/modules/rw_relweight.py#L30-L40)
  - [`app/modules/spider_profile.py`](file:///C:/Users/Praveen.Bhat/.gemini/antigravity/scratch/flask-cloud-run/app/modules/spider_profile.py#L35-L46)
- **Preloading**: Exposed `preload()` across modules and [`app/router.py`](file:///C:/Users/Praveen.Bhat/.gemini/antigravity/scratch/flask-cloud-run/app/router.py#L45-L50), and called it at server startup in [`app/main.py`](file:///C:/Users/Praveen.Bhat/.gemini/antigravity/scratch/flask-cloud-run/app/main.py#L20-L26) to eliminate cold-start query latency.

### 3. API Enhancements & Chart Safety
- **Safe Chart Swapping**: In [`app/main.py`](file:///C:/Users/Praveen.Bhat/.gemini/antigravity/scratch/flask-cloud-run/app/main.py#L80-L94), added validation preventing multi-series comparison queries from being switched to single-series chart types (`pie` or `doughnut`), avoiding frontend Chart.js crashes.
- **CORS Support**: Added global CORS headers and `OPTIONS` preflight handling so web apps and external clients can invoke the Cloud Run service.

### 4. Cloud Run & Container Configuration
- **Root Entrypoint**: Created [`main.py`](file:///C:/Users/Praveen.Bhat/.gemini/antigravity/scratch/flask-cloud-run/main.py) and [`Procfile`](file:///C:/Users/Praveen.Bhat/.gemini/antigravity/scratch/flask-cloud-run/Procfile) for seamless Google Cloud Buildpack deployments (`gcloud run deploy --source .`).
- **Optimized [`Dockerfile`](file:///C:/Users/Praveen.Bhat/.gemini/antigravity/scratch/flask-cloud-run/Dockerfile)**:
  - Added `ENV PYTHONUNBUFFERED=1` to ensure real-time log flushing to Cloud Logging.
  - Set Gunicorn command to `exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app` adhering to Cloud Run production recommendations (`--timeout 0` lets Cloud Run manage HTTP timeouts).
- **Flexible [`cloudbuild.yaml`](file:///C:/Users/Praveen.Bhat/.gemini/antigravity/scratch/flask-cloud-run/cloudbuild.yaml)**: Added `_TAG: latest` substitution so builds succeed both via Git push triggers and manual `gcloud builds submit`.
- **Dependencies**: Explicitly included `google-auth>=2.20.0` in [`requirements.txt`](file:///C:/Users/Praveen.Bhat/.gemini/antigravity/scratch/flask-cloud-run/requirements.txt).

---

## Verification Results

A comprehensive automated test suite (`verify_all.py`) verified:
- `test_01_root_app_same_as_package_app`: PASS
- `test_02_health_endpoint`: PASS (HTTP 200, status "ok", CORS header present)
- `test_03_presets_endpoint`: PASS (Returns all 7 preset queries)
- `test_04_all_presets_query`: PASS (All 7 presets successfully routed and answered)
- `test_05_options_cors`: PASS (OPTIONS request returns HTTP 204 with CORS allow headers)
- `test_06_single_series_chart_swap`: PASS (Swapping to line and pie for single series)
- `test_07_multi_series_chart_swap_safety`: PASS (Multi-series compare questions safely preserved as bar charts)
- `test_08_thread_safety_concurrent_queries`: PASS (18 concurrent queries across 8 worker threads execute without race conditions)
- `test_09_gemini_client_configuration`: PASS (Configured with `gemini-2.0-flash` and graceful fallback)

```
Ran 9 tests in 172.407s
OK
```
