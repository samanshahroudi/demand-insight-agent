# Demand Insight Agent

A small, self-contained product opportunity screening service. It uses a versioned deterministic score to rank candidates, then runs a LangGraph workflow that selects products, scores them, and produces a concise summary. A model-written summary is optional; the score and verdict always come from inspectable rules.

All included catalog records are synthetic examples. This repository contains no customer data or connection to external commerce/ad platforms.

## Architecture

```text
POST /runs
   └── LangGraph
       ├── select candidates from the bundled demo catalog
       ├── score candidates with deterministic, versioned rules
       └── summarize with a template or an optional LLM
```

The 100-point score combines demand (30), advertiser validation (20), gross margin (25), market headroom (15), and customer rating (10). Products below a 15% gross margin are rejected. The response includes each component, evidence-based confidence, and human-readable reasons. These example weights are not validated business advice.

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

## API

- `GET /health` reports service health.
- `POST /runs` accepts `mode` (`discover` or `shortlist`), optional `product_ids`, result `limit`, and `use_llm_summary`.
- Product IDs must contain non-whitespace text; malformed IDs return HTTP 422 before the graph runs.
- Each response includes a run ID, ranked assessments, a summary, and the scoring version.
- Deterministic summaries include each leading product's score and verdict, so a high score cannot hide rejection by the gross-margin gate.

## Design choices and next steps

- Scoring is deterministic and inspectable; an LLM cannot silently alter numeric scores.
- The graph makes execution stages and optional model use explicit, while keeping the demo easy to run.
- The bundled catalog is for demonstration only. A real deployment needs permissioned data, source timestamps, data-quality checks, calibrated score weights, an evaluation set, persistence, authentication, quotas, and observability.
- The API currently reads a small local catalog and is intended as a portfolio-sized starting point, not a production service.

## Interview topics

Explain why scoring is separated from generated summaries, how you would calibrate weights against outcomes, how stale evidence affects confidence, and what should be persisted to reproduce a run.
