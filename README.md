# Demand Insight Agent

A small, self-contained product opportunity screening service. It uses a versioned deterministic score to rank candidates, then runs a LangGraph workflow that selects products, scores them, and produces a concise summary. A model-written summary is optional; the score and verdict always come from inspectable rules.

All included catalog records are synthetic examples. This repository contains no customer data or connection to external commerce/ad platforms.

An empty discovery run returns "No products matched this run." without calling the model, even when model summaries are enabled. Credentials are unnecessary when there are no results to summarize.

## Architecture

```text
POST /runs
   └── LangGraph
       ├── select candidates from the bundled demo catalog
       ├── score candidates with deterministic, versioned rules
       └── summarize with a template or an optional LLM
```

The 100-point score combines demand (30), advertiser validation (20), gross margin (25), market headroom (15), and customer rating (10). Products below a 15% gross margin are rejected. The response includes each component, evidence-based confidence, and human-readable reasons. These example weights are not validated business advice.

Catalog advertiser and evidence counts require nonnegative integers. Booleans, strings, and floats are rejected rather than coerced into scoring or confidence inputs.
Catalog product IDs must contain non-whitespace text, just like shortlist request IDs. Invalid catalog IDs are rejected before assessments or model summaries are produced.
Catalog names and categories must also contain non-whitespace text, so results and summaries cannot contain blank product labels.
Catalog records are validated before shortlist filtering. Missing or malformed product IDs and non-object records return HTTP 422 in both modes before any model summary is requested.
The catalog must be a JSON list; other top-level values return HTTP 422 before selection or model calls. An empty list remains valid for an empty discovery run.
The entire catalog must also have unique product IDs; duplicates return HTTP 422 before shortlist filtering, scoring, or requesting a model summary.
Demand, margin, saturation, and rating signals require numbers; booleans and numeric strings are rejected before scoring or model summaries. Integer and floating-point JSON numbers remain valid within each signal's bounds.

## Run locally

```bash
python -m pip install -e ".[dev]"
uvicorn demand_insight_agent.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API interface.

Discover from the synthetic catalog:

```bash
curl -X POST http://127.0.0.1:8000/runs \
  -H 'content-type: application/json' \
  -d '{"mode":"discover","limit":3}'
```

Score selected catalog items:

```bash
curl -X POST http://127.0.0.1:8000/runs \
  -H 'content-type: application/json' \
  -d '{"mode":"shortlist","product_ids":["sample-001","sample-003"],"limit":2}'
```

Set `OPENAI_API_KEY` and `OPENAI_MODEL` and pass `"use_llm_summary": true` to request a model-written summary. Without that option, all functionality is local and does not make model calls.
Each model-summary request has a 20-second timeout per attempt; provider retries can extend the total run time.
Model provider failures return HTTP 503 with `Model summary unavailable.` and no partial assessments or provider response details. Deterministic runs and the health endpoint remain available.
Missing or unreadable catalog files return HTTP 503 with `Product catalog unavailable.` without exposing filesystem details or calling the model. The health endpoint remains available.
Catalog files with invalid UTF-8 or malformed JSON receive the same HTTP 503 response without exposing decoding or parsing details.
`use_llm_summary` requires a JSON boolean; strings and numbers return HTTP 422 before the graph runs.
An empty, whitespace-only, or reasoning-only model response returns HTTP 503 with `Model summary unavailable.`, just like other provider failures. The same request can be retried; deterministic runs remain available.

## API

- `GET /health` reports service health.
- `POST /runs` accepts `mode` (`discover` or `shortlist`), optional `product_ids`, result `limit`, and `use_llm_summary`.
- Unknown request fields return HTTP 422 before the graph runs, so misspelled options cannot silently use defaults.
- `limit` requires a JSON integer from 1 to 20; booleans, strings, and floats return HTTP 422 before the graph runs.
- Product IDs must contain non-whitespace text; malformed IDs return HTTP 422 before the graph runs.
- Each response includes a run ID, ranked assessments, a summary, and the scoring version.
- Equal scores are ordered by product ID before applying the result limit, so catalog order does not change tied selections.
- Deterministic summaries include each leading product's score and verdict, so a high score cannot hide rejection by the gross-margin gate.

## Design choices and next steps

- Scoring is deterministic and inspectable; an LLM cannot silently alter numeric scores.
- The graph makes execution stages and optional model use explicit, while keeping the demo easy to run.
- The bundled catalog is for demonstration only. A real deployment needs permissioned data, source timestamps, data-quality checks, calibrated score weights, an evaluation set, persistence, authentication, quotas, and observability.
- The API currently reads a small local catalog and is intended as a portfolio-sized starting point, not a production service.

## Interview topics

Explain why scoring is separated from generated summaries, how you would calibrate weights against outcomes, how stale evidence affects confidence, and what should be persisted to reproduce a run.
