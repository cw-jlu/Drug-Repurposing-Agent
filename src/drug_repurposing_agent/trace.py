"""Durable, local JSONL traces for research-agent and provider calls."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from uuid import uuid4
from typing import Callable, TypeVar


_SECRET_KEYS = {"authorization", "api_key", "apikey", "password", "secret",
                "access_token", "refresh_token", "credential", "credentials"}
_HIDDEN_KEYS = {"reasoning_content", "chain_of_thought", "hidden_reasoning"}
_SECRET_TEXT = re.compile(r"(?i)(bearer\s+[^\s\"']+|\bsk-[a-z0-9_-]{12,}\b)")
_ASSIGNMENT = re.compile(r"(?i)(DEEPSEEK_API_KEY|TYPESAFE_API_KEY)\s*=\s*[^\s\"']+")


def redact(value: object, secrets: tuple[str, ...] = ()) -> object:
    """Remove credential fields and credential-looking substrings recursively."""
    if isinstance(value, dict):
        return {str(redact(str(key), secrets)):
                ("[REDACTED]" if str(key).lower() in _SECRET_KEYS
                 else "[OMITTED: internal reasoning]" if str(key).lower() in _HIDDEN_KEYS
                 else redact(item, secrets)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        result = _ASSIGNMENT.sub(r"\1=[REDACTED]", _SECRET_TEXT.sub("[REDACTED]", value))
        for secret in secrets:
            if secret:
                result = result.replace(secret, "[REDACTED]")
        return result
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact(str(value), secrets)


class TraceRecorder:
    """Write each event immediately so a later failure does not erase the trace."""

    def __init__(self, kind: str, directory: Path | None = None,
                 secrets: tuple[str, ...] = ()):
        root = directory or Path(os.environ.get("DRUG_AGENT_TRACE_DIR", "artifacts/traces"))
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        safe_kind = re.sub(r"[^a-zA-Z0-9_-]", "_", kind)
        self.run_id = uuid4().hex
        self._secrets = secrets + tuple(os.environ.get(key, "") for key in
                                         ("DEEPSEEK_API_KEY", "TYPESAFE_API_KEY"))
        self.path = root / f"{stamp}_{safe_kind}_{self.run_id}.jsonl"
        self.emit("trace_started", kind=kind)

    def emit(self, stage: str, **details: object) -> dict[str, object]:
        event = redact({"schema_version": 1, "run_id": self.run_id,
                        "time": datetime.now(timezone.utc).isoformat(),
                        "stage": stage, **details}, self._secrets)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event

    def add_secret(self, secret: str) -> None:
        """Register a credential obtained after recorder construction."""
        if secret:
            self._secrets += (secret,)


_T = TypeVar("_T")


def traced_run(kind: str, action: Callable[[TraceRecorder], _T],
               directory: Path | None = None) -> _T:
    """Record entrypoint success/failure even when no result file is produced."""
    trace = TraceRecorder(kind, directory)
    try:
        result = action(trace)
        trace.emit("run_completed")
        print(f"Trace: {trace.path}")
        return result
    except Exception as exc:
        trace.emit("run_failed", error_type=type(exc).__name__, error=str(exc))
        print(f"Trace: {trace.path}")
        raise
