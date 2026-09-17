# Sensory Agent

A multi-module sensory/consumer-research agent. One question, one endpoint,
one Cloud Run service — internally routed to whichever of six analysis
modules fits the question, each backed by its own dataset and chart type.

| Module | Source data | Question style | Chart |
|---|---|---|---|
| `sr_attribute` | sst_attribute_rank.csv | "What attributes drive Moisturization?" | bar |
| `sd_driver` | sst_driver_rank.csv | "What test stages drive Overall opinion?" | bar |
| `bh_blindhut` | blind_hut_benefit_score.csv | "Which products score highest on Instant glow?" / "How does <product> score?" | bar |
| `pc_perceptual` | pca_coordinate.csv | "Show the perceptual map for SUNCARE" | scatter |
| `rw_relweight` | rwa_driver_importance.csv | "What drives Shine importance?" (hair care) | bar |
| `spider_profile` | spider_attribute_score.csv | "Show the attribute profile for Anessa" | radar |

## How routing works

`app/router.py` classifies each question with keyword + fuzzy-matching
(`app/modules/utils.py`) — no LLM in the loop for routing, so it's fast,
free, and debuggable. It checks, in order: hair-care platform match (RW) →
explicit radar/profile language (spider) → explicit perceptual-map language
or category match (PC) → explicit test-stage language (SD) → product-scoring
language (BH) → named product matches → default to attribute-driver analysis
(SR), the most common ask. If the chosen module can't answer, it falls back
through the others before giving up.

Each module is self-contained: it owns its data loading, its ranking/
aggregation logic, and the shape of its chart. The numbers are always
deterministic pandas — an optional Vertex AI Gemini call (`app/gemini_client.py`)
only rewrites the templated summary into more natural prose; if it's not
configured or fails, the templated summary is used and nothing else changes.

## Project layout

```
main.py                Root entrypoint for Cloud Buildpacks and local runs
Procfile               Procfile for Cloud Run Buildpacks
app/
  main.py              Flask app, single /api/query endpoint
  router.py            Classifies a question -> picks a module
  gemini_client.py     Optional Vertex AI Gemini summary rewrite (gemini-2.0-flash)
  modules/
    utils.py           Shared fuzzy-match helpers
    sr_attribute.py    Attribute-level driver rank (bar)
    sd_driver.py       Test-stage driver rank (bar)
    bh_blindhut.py     Blind hut benefit scores (bar)
    pc_perceptual.py   PCA perceptual map (scatter)
    rw_relweight.py    Relative-weight driver importance, hair care (bar)
    spider_profile.py  Attribute profile (radar)
  static/index.html    Chat + multi-chart-type front-end
data/
  *.csv                Exported from the 6 source Excel files
Dockerfile
cloudbuild.yaml        Build -> Artifact Registry -> Cloud Run
requirements.txt
```

## API

- `GET  /` — chat UI
- `GET  /health` — health check
- `GET  /api/presets` — one example question per module
- `POST /api/query` — `{"question": "...", "top_n": 8}` ->
  `{ok, module, title, chart: {type, ...}, table, summary}`

`chart.type` is one of `bar`, `scatter`, `radar` — the front-end (and any
client you build against this API) branches on it.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m flask --app app.main run --port 8080
# open http://localhost:8080
```

Works fully with zero GCP config — falls back to templated summaries. Set
`GOOGLE_CLOUD_PROJECT` (and authenticate with
`gcloud auth application-default login`) to try the Gemini-written summaries
locally.

## Deploy to Cloud Run

### 0. One-time GCP setup

```bash
export PROJECT_ID=<your-gcp-project>
export REGION=europe-west1   # change if you deployed elsewhere

gcloud config set project $PROJECT_ID

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com

gcloud artifacts repositories create sst-agent-repo \
  --repository-format=docker \
  --location=$REGION
```

### 1. Push this repo to GitHub

```bash
cd sensory-agent
git init
git add .
git commit -m "Initial commit: sensory agent"
git branch -M main
git remote add origin https://github.com/<your-org>/sensory-agent.git
git push -u origin main
```

### Option A — Cloud Build trigger (auto-deploy on push)

In the Console: **Cloud Build > Triggers > Connect Repository**, pick this
GitHub repo, then create a trigger with config file `cloudbuild.yaml` and
branch `^main$`. If your region/repo name differ from the defaults
(`europe-west1` / `sst-agent-repo`), add substitution overrides `_REGION`
and `_REPO` on the trigger. Every push to `main` rebuilds and redeploys.

To enable the LLM tool-calling agent (see "Optional: LLM tool-calling
router" below) on this deploy path, add a substitution override
`_USE_LLM_ROUTER=true` on the trigger — do **not** set it with a one-off
`gcloud run services update` instead, since `cloudbuild.yaml`'s
`--set-env-vars` replaces the full environment variable list on every
push, and a manually-set var not present in this file would silently
disappear on the next merge to `main`.

### Option B — Deploy directly from source

```bash
gcloud run deploy sensory-agent \
  --source . \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GCP_LOCATION=$REGION
```

### Grant Vertex AI access (for the LLM summary rewrite)

```bash
SA=$(gcloud run services describe sensory-agent --region $REGION \
  --format='value(spec.template.spec.serviceAccountName)')

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA}" \
  --role="roles/aiplatform.user"
```

Skip this and set `USE_LLM_SUMMARY=false` if you don't want the LLM step —
the app works identically otherwise, just with the templated summary text.

### Optional: LLM tool-calling router (agentic mode)

By default, routing (which analysis module answers a question) is handled
by the deterministic keyword/fuzzy-match router in `app/router.py` — fast,
free, and fully predictable. Set `USE_LLM_ROUTER=true` (on top of the
Vertex AI access above, or a `GEMINI_API_KEY`) to let Gemini decide
instead, via function calling against the six analysis modules as tools.

This unlocks questions the deterministic router can't answer in one shot —
e.g. *"What's the top driver of Natural Product, and how does that same
stage rank for Moisturization?"* — where Gemini calls a tool, reads what
it returned, and calls a second tool using that result before writing a
final answer. It's capped at two tool calls per question.

Safety property: the chart, table, and every number shown to the user
always come straight from the same deterministic `module.answer()`
functions used everywhere else in the app — Gemini only chooses which
tool(s) to call and narrates the result. If no Gemini client is
configured, or the call fails, or no tool call ever succeeds, the app
transparently falls back to the deterministic router — the person asking
never sees an LLM-routing failure, only a normal answer or the usual
"couldn't match your question" message.

### If you see "Forbidden" after deploying

Your Cloud Run service is set to require authentication. Either:
- Console: service > **Security** tab > switch to "Allow unauthenticated
  invocations" (if your org policy allows public services), or
- Grant specific users the **Cloud Run Invoker** role under IAM if your org
  blocks public (`allUsers`) access — common on corporate GCP orgs.

## Adding a 7th module

1. Export the new dataset to `data/<name>.csv`.
2. Create `app/modules/<name>.py` following the existing modules' contract:
   `answer(question, top_n) -> {ok, module, title, chart, table, summary}`.
3. Register it in `MODULES` in `app/router.py` and add a routing rule.
4. Add its `chart.type` to the front-end's `renderChart()` if it's a new
   chart shape (bar/scatter/radar are already handled).
