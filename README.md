# Clinical AI Guardian

A HIPAA-aware, production-hardened **clinical document intelligence platform** — the reference architecture for safe LLM applications over sensitive health data.

Ask clinical questions over documents. Every query passes through **PII redaction**, **hybrid retrieval**, an **agentic reasoning pipeline** with **human-in-the-loop** escalation, and **input/output guardrails** — with every decision written to a **tamper-evident audit log**.

```
                        ┌─────────────────────────────────────────────┐
                        │              FastAPI  (/v1/ask)              │
                        │  API key auth · rate limiting · SSE stream   │
                        └──────────────────┬──────────────────────────┘
                                           │ request_id
                        ┌──────────────────▼──────────────────────────┐
                        │              AGENT GRAPH                      │
                        │                                             │
   ┌─────────┐    ┌─────▼─────┐   ┌──────────┐   ┌──────────┐   ┌──────▼──────┐
   │  Query  │───▶│  TRIAGE   │──▶│ RETRIEVE │──▶│SYNTHESIZE│──▶│   VERIFY    │
   └─────────┘    │ guardrail │   │ hybrid   │   │ LLM +    │   │ groundedness│
                  │ + intent  │   │ RAG      │   │ citations│   │ PII scan    │
                  │ + PII     │   └────┬─────┘   └──────────┘   └──────┬──────┘
                  │ redact    │        │                             │ low score
                  └───────────┘        │                             ▼
                                       │                    ┌──────────────────┐
                                       │                    │  HUMAN-IN-LOOP   │
                                       │                    │ approve / reject │
                                       │                    └────────┬─────────┘
                                       │                             │
                        ┌──────────────▼──────────────┐   ┌──────────▼──────────┐
                        │   RESPOND: disclaimer,      │   │  AUDIT LOG (JSONL)  │
                        │   citations, final answer   │   │  hash-chained,      │
                        └─────────────────────────────┘   │  PII-free          │
                                                          └───────────────────┘

   RETRIEVE = dense embeddings + BM25 → RRF fusion → cross-encoder rerank
   PRIVACY  = regex + heuristic NER → [PERSON_1] placeholders + sealed map
```

## Why this project

Clinical LLM systems fail in predictable ways: prompt injection, hallucinated
dosages, PII leaking into logs and prompts, and no audit trail when something
goes wrong. This project implements the defenses as **testable, composable
modules** — not as prompt vibes.

## Quickstart (fully offline, no credentials)

```bash
git clone https://github.com/chandrahas748812-cmyk/clinical-ai-guardian
cd clinical-ai-guardian
pip install -r requirements.txt   # fastapi/uvicorn only needed to serve
python example.py                 # end-to-end demo on synthetic notes
pytest tests/ -q                  # 41 tests, all offline
```

`example.py` indexes two synthetic clinical notes, runs three queries —
a normal Q&A, a summarization, and a prompt-injection attack (blocked) —
and verifies the audit chain.

## API

```bash
# serve (needs fastapi + uvicorn)
uvicorn src.api.server:create_app --factory --port 8000
```

```bash
curl -X POST localhost:8000/v1/ask \
  -H "X-API-Key: demo-key-123" -H "Content-Type: application/json" \
  -d '{"query": "What dosage of metformin was prescribed?"}'
# → {"request_id": "...", "answer": "... [doc]", "citations": [...],
#     "groundedness": 0.92, "hitl": {"escalated": false, ...}}

# streaming
curl -N -X POST localhost:8000/v1/ask \
  -H "X-API-Key: demo-key-123" -H "Content-Type: application/json" \
  -d '{"query": "...", "stream": true}'   # Server-Sent Events

curl localhost:8000/metrics      # Prometheus exposition format
curl localhost:8000/health       # liveness
```

## Module tour

| Module | What it does |
|---|---|
| `src/retrieval/` | Fixed / semantic / hierarchical chunking; dense (sentence-transformers, mock fallback) + BM25 → RRF fusion → cross-encoder rerank; precision@k / recall@k / MRR |
| `src/privacy/` | Presidio-style PII detection (SSN, MRN, phone, email, dates, names); redaction to `[TYPE_N]` placeholders with a reversible, seal-hashed map |
| `src/agents/` | LangGraph state machine (triage → retrieve → synthesize → verify → respond) with pure-Python fallback; tool calling; HITL escalation |
| `src/guardrails/` | Input: injection/XSS/PII-harvesting patterns, length + topic checks. Output: groundedness scoring, PII-leak scan, citation requirement, medical disclaimer |
| `src/eval/` | Golden JSONL datasets, deterministic LLM-as-judge, hallucination detection, latency/cost tracking, CI regression gate |
| `src/api/` | FastAPI: SSE streaming, API-key auth, token-bucket rate limiting, Prometheus `/metrics`, background audit worker |
| `src/audit/` | Append-only JSONL log with sha256 hash chain; `verify()` detects tampering; PII values are scrubbed before write |

## HIPAA design notes — what's real vs. demo

**Real architectural patterns** (how production clinical AI is actually built):
- PII is redacted **before** any LLM call; raw values never leave the trust boundary
- The audit log stores redaction *seals* (hashes), never PII — you can prove
  what was redacted without storing it
- Hash-chained logs make silent modification detectable
- Human-in-the-loop defaults to **reject** on escalation (safe default)
- Every answer carries citations and a medical disclaimer

**Demo simplifications** (do not deploy as-is):
- PII detector is regex + heuristics, not a trained NER model (Presidio/spaCy)
  — expect lower recall on real notes; production needs human review sampling
- Mock LLM / mock embeddings are deterministic stand-ins, not real models
- API-key auth and in-memory rate limiting are demo-grade (use OAuth2 + Redis)
- No encryption at rest, no access-control lists, no BAA-covered infra
- Synthetic data only — never commit real PHI to a repo

## Eval results (offline, mock backends)

```
$ python -m pytest tests/ -q
41 passed in 0.08s

Golden set (configs/golden.jsonl): 5 cases — dosage QA, cardiac summary,
allergy extraction, prompt-injection block, lab lookup.
```

The regression gate (`src/eval/regression.py`) fails CI if pass rate,
groundedness, or latency regress past `configs/baseline.json`.

## Production gaps (honest)

1. **De-identification recall** — needs a trained clinical NER + human audit loop
2. **Real vector DB** — swap `DocumentStore` for pgvector/OpenSearch (interface is narrow by design)
3. **AuthN/Z** — OAuth2/OIDC, per-tenant isolation, audit-log encryption
4. **LLM backend** — Bedrock/Anthropic with VPC endpoints; the `src/llm.py`
   adapter interface is ready
5. **Red-teaming** — the injection patterns are a starting set, not exhaustive

## Layout

```
src/
  llm.py            LLM backend abstraction (mock ↔ real)
  retrieval/        chunking.py, hybrid.py, store.py, metrics.py
  privacy/          detector.py, redactor.py
  agents/           state.py, graph.py, tools.py
  guardrails/       input.py, output.py
  eval/             dataset.py, judge.py, regression.py
  api/              server.py
  audit/            logger.py
configs/            pipeline.yaml, privacy.yaml, guardrails.yaml,
                    baseline.json, golden.jsonl, api_keys.txt
data/               synthetic clinical notes (fictional)
tests/              41 tests
Dockerfile, docker-compose.yml
```

## License

MIT — synthetic demo data included; no real patient data anywhere in this repo.
