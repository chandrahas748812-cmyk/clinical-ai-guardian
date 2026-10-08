"""Clinical AI Guardian — HIPAA-aware clinical document intelligence platform.

A production-hardened reference architecture for clinical Q&A over sensitive
documents: hybrid retrieval, PII redaction with audit, agentic reasoning with
human-in-the-loop, input/output guardrails, offline evaluation, and a
production FastAPI with tamper-evident audit logging.

Every external service (LLM, embeddings, vector DB) sits behind a mock
fallback so the full pipeline — including example.py and the test suite —
runs offline with zero credentials.
"""

__version__ = "0.1.0"
