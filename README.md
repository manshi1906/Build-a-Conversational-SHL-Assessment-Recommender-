# SHL Conversational Assessment Recommender

A stateless FastAPI agent that takes a hiring manager from a vague intent
(*"I'm hiring a Java developer"*) to a grounded shortlist of **real SHL
Individual Test Solutions**, through dialogue. It clarifies, recommends,
refines, compares, and refuses off-topic / injection attempts.

## Quick start

```bash
pip install -r requirements.txt
python scripts/seed_catalog.py          # bootstrap fallback catalog
python scripts/scrape_catalog.py        # (deploy env) full live catalog
export LLM_PROVIDER=groq
export LLM_API_KEY=<your_groq_key>      # free tier: console.groq.com
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Endpoints

`GET /health` → `{"status":"ok"}` (200)

`POST /chat`
```json
{ "messages": [ {"role":"user","content":"..."}, ... ] }
```
→
```json
{ "reply": "...",
  "recommendations": [ {"name":"...","url":"https://www.shl.com/...","test_type":"K"} ],
  "end_of_conversation": false }
```
`recommendations` is `[]` while clarifying/refusing, else **1–10** real
catalog items. Stateless: the full history is sent every call.

## Architecture

```
scrape_catalog.py ─┐
seed_catalog.py  ──┴─► data/catalog.json
                              │
                    retrieval.py  (BM25 + MiniLM hybrid, type coverage)
                              │
                    agent.py  (router → clarify/recommend/refine/compare/refuse
                               → grounded selection; turn-cap; schema guards)
                              │
                    main.py   (FastAPI, strict schema, startup-built retriever)
```

## Evaluate locally

```bash
python -m eval.run_eval            # Recall@10 + behaviour probes
python -m tests.test_agent_pipeline  # full state machine (mocked LLM)
```

## Deploy

- **Render**: `render.yaml` included (free tier). Set `LLM_API_KEY` in dashboard.
- **Docker / Fly / HF Spaces**: `Dockerfile` included; prefetches the
  embedding model so the first `/chat` is fast.

## Switching LLM provider

One env var, no code change: `LLM_PROVIDER=groq|openai|gemini`
(+ `LLM_API_KEY`, optional `LLM_MODEL`).

## Robustness

Every external dependency degrades instead of breaking: no LLM key → safe
deterministic path; no HuggingFace → BM25-only retrieval; LLM returns junk →
JSON repair + fallbacks. The schema is **always** valid (verified by
`eval/run_eval.py` even in fully-degraded mode).
