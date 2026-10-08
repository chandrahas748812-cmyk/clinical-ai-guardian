"""Tamper-evident audit logging for HIPAA compliance.

Every record is appended as one JSON line and includes:
- `prev_hash`: sha256 of the previous record (genesis uses "GENESIS")
- `record_hash`: sha256 over the canonical record + prev_hash

This hash chain makes silent modification/deletion detectable: `verify()`
recomputes the chain and reports the first broken link. The log stores
*redaction seals* (hashes), never raw PII.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional


def _canonical(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class AuditRecord:
    event: str                          # query | redaction | hitl | guardrail | response
    request_id: str
    actor: str = "system"
    details: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    prev_hash: str = "GENESIS"
    record_hash: str = ""

    def seal(self, prev_hash: str) -> "AuditRecord":
        self.prev_hash = prev_hash
        body = {
            "event": self.event, "request_id": self.request_id,
            "actor": self.actor, "details": self.details,
            "ts": self.ts, "record_id": self.record_id,
            "prev_hash": self.prev_hash,
        }
        self.record_hash = _sha256(_canonical(body))
        return self

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AuditRecord":
        return AuditRecord(**{k: v for k, v in d.items()
                              if k in AuditRecord.__dataclass_fields__})


class AuditLogger:
    """Append-only JSONL audit log with hash-chain integrity."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if not os.path.exists(path):
            open(path, "a").close()

    def _last_hash(self) -> str:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = [ln for ln in f if ln.strip()]
            if not lines:
                return "GENESIS"
            return json.loads(lines[-1]).get("record_hash", "GENESIS")
        except FileNotFoundError:
            return "GENESIS"

    def log(self, event: str, request_id: str, actor: str = "system",
            **details: Any) -> AuditRecord:
        # Strip anything that looks like raw PII values before persisting.
        safe_details = _scrub(details)
        rec = AuditRecord(event=event, request_id=request_id, actor=actor,
                          details=safe_details)
        with self._lock:
            rec.seal(self._last_hash())
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(_canonical(rec.to_dict()) + "\n")
        return rec

    def records(self) -> Iterator[AuditRecord]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield AuditRecord.from_dict(json.loads(line))

    def verify(self) -> Dict[str, Any]:
        """Recompute the hash chain. Returns {ok, checked, broken_at}."""
        prev = "GENESIS"
        checked = 0
        for rec in self.records():
            body = {
                "event": rec.event, "request_id": rec.request_id,
                "actor": rec.actor, "details": rec.details,
                "ts": rec.ts, "record_id": rec.record_id,
                "prev_hash": rec.prev_hash,
            }
            if rec.prev_hash != prev:
                return {"ok": False, "checked": checked,
                        "broken_at": rec.record_id, "reason": "prev_hash mismatch"}
            if rec.record_hash != _sha256(_canonical(body)):
                return {"ok": False, "checked": checked,
                        "broken_at": rec.record_id, "reason": "record_hash mismatch"}
            prev = rec.record_hash
            checked += 1
        return {"ok": True, "checked": checked, "broken_at": None}


_PII_HINT_KEYS = {"ssn", "phone", "email", "mrn", "dob", "name", "address"}


def _scrub(details: Dict[str, Any]) -> Dict[str, Any]:
    """Redact values whose keys look like PII before writing to the log."""
    out: Dict[str, Any] = {}
    for k, v in details.items():
        if any(h in k.lower() for h in _PII_HINT_KEYS) and isinstance(v, str):
            out[k] = f"<redacted:{_sha256(v)[:8]}>"
        else:
            out[k] = v
    return out
