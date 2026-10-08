"""Production FastAPI: streaming Q&A, auth, rate limiting, observability.

Endpoints:
- POST /v1/ask            ask a clinical question (SSE stream or JSON)
- GET  /health            liveness
- GET  /ready             readiness (store indexed?)
- GET  /metrics           Prometheus-format metrics
- POST /v1/audit/verify   verify audit-log hash chain

Security model (demo-grade, documented in README):
- API key auth via X-API-Key header (keys in configs/api_keys.txt or env)
- In-memory token-bucket rate limiting per key
- Every request gets a request_id; all decisions go to the audit log
- PII is redacted before the LLM call; raw values never leave the trust boundary
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


# ----------------------------------------------------------------------------
# Pure components (no fastapi needed -> unit testable)
# ----------------------------------------------------------------------------

@dataclass
class RateLimiter:
    """Token-bucket rate limiter, per API key."""

    rate_per_min: int = 60
    _buckets: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    def allow(self, key: str, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        window = 60.0
        bucket = self._buckets[key]
        bucket[:] = [t for t in bucket if now - t < window]
        if len(bucket) >= self.rate_per_min:
            return False
        bucket.append(now)
        return True


class KeyAuth:
    """Simple API-key auth. Keys from file or CLINICAL_API_KEYS env (csv)."""

    def __init__(self, keys: Optional[List[str]] = None):
        if keys is None:
            keys = []
            env = os.environ.get("CLINICAL_API_KEYS", "")
            keys.extend(k.strip() for k in env.split(",") if k.strip())
            path = os.environ.get("CLINICAL_API_KEYS_FILE", "configs/api_keys.txt")
            if os.path.exists(path):
                with open(path) as f:
                    keys.extend(l.strip() for l in f if l.strip() and not l.startswith("#"))
        self._keys = set(keys) or {"demo-key-123"}

    def verify(self, provided: Optional[str]) -> bool:
        return bool(provided) and provided in self._keys


@dataclass
class Metrics:
    """In-memory counters rendered as Prometheus exposition format."""

    requests_total: int = 0
    requests_blocked: int = 0
    hitl_escalations: int = 0
    total_latency_ms: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0

    def observe(self, latency_ms: float, blocked: bool, hitl: bool,
                tokens_in: int, tokens_out: int) -> None:
        self.requests_total += 1
        self.total_latency_ms += latency_ms
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        if blocked:
            self.requests_blocked += 1
        if hitl:
            self.hitl_escalations += 1

    def render(self) -> str:
        avg_lat = self.total_latency_ms / max(self.requests_total, 1)
        lines = [
            "# HELP clinical_requests_total Total Q&A requests",
            "# TYPE clinical_requests_total counter",
            f"clinical_requests_total {self.requests_total}",
            "# HELP clinical_requests_blocked Blocked by guardrails",
            "# TYPE clinical_requests_blocked counter",
            f"clinical_requests_blocked {self.requests_blocked}",
            "# HELP clinical_hitl_escalations HITL escalations",
            "# TYPE clinical_hitl_escalations counter",
            f"clinical_hitl_escalations {self.hitl_escalations}",
            "# HELP clinical_latency_avg_ms Average end-to-end latency",
            "# TYPE clinical_latency_avg_ms gauge",
            f"clinical_latency_avg_ms {avg_lat:.1f}",
            "# HELP clinical_tokens_in_total Total input tokens",
            "# TYPE clinical_tokens_in_total counter",
            f"clinical_tokens_in_total {self.total_tokens_in}",
            "# HELP clinical_tokens_out_total Total output tokens",
            "# TYPE clinical_tokens_out_total counter",
            f"clinical_tokens_out_total {self.total_tokens_out}",
        ]
        return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# FastAPI app (import-guarded so the package imports without fastapi)
# ----------------------------------------------------------------------------

def create_app(deps: Dict[str, Any]):
    """Build the FastAPI application. Requires fastapi+uvicorn installed."""
    try:
        from fastapi import FastAPI, Header, HTTPException, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "fastapi is required to serve the API: pip install -r requirements.txt"
        ) from e

    from ..agents.graph import ClinicalAgent
    from ..audit.logger import AuditLogger

    auth = KeyAuth()
    limiter = RateLimiter(rate_per_min=int(os.environ.get("RATE_PER_MIN", "60")))
    metrics = Metrics()
    audit: AuditLogger = deps["audit"]
    agent = ClinicalAgent(deps)

    app = FastAPI(title="Clinical AI Guardian", version="0.1.0")

    def _check(request: Request, x_api_key: Optional[str] = Header(default=None)):
        if not auth.verify(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")
        key = x_api_key or "anonymous"
        if not limiter.allow(key):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        return key

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/ready")
    def ready():
        store = deps.get("store")
        indexed = bool(store and store.chunks)
        return {"ready": indexed, "chunks": len(store.chunks) if store else 0}

    @app.get("/metrics")
    def metrics_ep():
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(metrics.render())

    @app.post("/v1/audit/verify")
    def audit_verify():
        return audit.verify()

    async def _run_agent(query: str, request_id: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, agent.ask, query, request_id)

    @app.post("/v1/ask")
    async def ask(body: Dict[str, Any], request: Request,
                  x_api_key: Optional[str] = Header(default=None)):
        key = _check(request, x_api_key)
        query = (body.get("query") or "").strip()
        stream = bool(body.get("stream", False))
        request_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        audit.log("query", request_id, actor=f"api:{key[:6]}",
                  query_chars=len(query), stream=stream)
        state = await _run_agent(query, request_id)
        latency = (time.perf_counter() - t0) * 1000
        metrics.observe(latency, state.blocked, state.needs_hitl,
                        state.tokens_in, state.tokens_out)
        audit.log("response", request_id, actor="agent",
                  blocked=state.blocked, hitl=state.needs_hitl,
                  groundedness=state.groundedness,
                  citations=state.citations,
                  latency_ms=round(latency, 1))
        # Fire-and-forget redaction audit (background worker pattern)
        asyncio.create_task(_audit_redaction(audit, request_id, state))

        if state.blocked:
            raise HTTPException(status_code=422, detail=state.block_reason)
        payload = {
            "request_id": request_id,
            "answer": state.final_answer,
            "citations": state.citations,
            "groundedness": state.groundedness,
            "hitl": {"escalated": state.needs_hitl,
                     "decision": state.hitl_decision,
                     "reason": state.hitl_reason},
            "latency_ms": round(latency, 1),
        }
        if not stream:
            return JSONResponse(payload)

        async def _sse() -> AsyncIterator[str]:
            # Stream the answer in sentence chunks, then a final done event.
            text = state.final_answer or ""
            for sent in text.split(". "):
                chunk = json.dumps({"request_id": request_id, "delta": sent})
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0)
            yield f"data: {json.dumps({'request_id': request_id, 'done': True})}\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream")

    async def _audit_redaction(audit: AuditLogger, request_id: str, state: Any):
        audit.log("redaction", request_id, actor="privacy-pipeline",
                  seal=state.redaction_seal or "none")

    return app
