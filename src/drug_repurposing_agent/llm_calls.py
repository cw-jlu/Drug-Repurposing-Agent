"""Traced, retrying single-tool-call helper built on the existing DeepSeek client.

Every attempt goes through ``DeepSeekPlanner._post`` so each provider request gets
its own sealed ``TraceRecorder`` file. A fresh client object is created per call,
which keeps ``last_trace_path`` thread-safe under a small worker pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import threading
from time import sleep
from typing import Callable

from .deepseek import DeepSeekConfig, DeepSeekPlanner, DeepSeekPlannerError

BETA_BASE_URL = "https://api.deepseek.com/beta"
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens",
              "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")


class ToolCallError(RuntimeError):
    """Sanitized failure after all retries (never contains credentials)."""


def tool_payload(model: str, system: str, user: str, name: str, description: str,
                 parameters: dict, max_tokens: int = 600) -> dict:
    return {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "tools": [{"type": "function", "function": {
            "name": name, "strict": True, "description": description,
            "parameters": parameters}}],
        "tool_choice": "required",
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "stream": False,
    }


def extract_tool_arguments(response: dict, name: str) -> dict:
    try:
        choice = response["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("truncated response")
        calls = choice["message"]["tool_calls"]
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError("expected exactly one tool call")
        function = calls[0]["function"]
        if function["name"] != name:
            raise ValueError("unexpected function name")
        result = json.loads(function["arguments"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid provider tool-call structure") from exc
    if not isinstance(result, dict):
        raise ValueError("tool arguments must be an object")
    return result


@dataclass
class CallStats:
    """Thread-safe counters for attempts, failures and token usage."""
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    usage: dict = field(default_factory=lambda: {key: 0 for key in USAGE_KEYS})
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def start(self) -> None:
        with self._lock:
            self.attempts += 1

    def finish(self, ok: bool, usage: dict | None = None) -> None:
        with self._lock:
            if ok:
                self.successes += 1
            else:
                self.failures += 1
            for key, value in (usage or {}).items():
                if key in self.usage and isinstance(value, int):
                    self.usage[key] += value

    def as_dict(self) -> dict:
        with self._lock:
            return {"attempts": self.attempts, "successes": self.successes,
                    "failures": self.failures, "usage": dict(self.usage)}


def estimate_cost_usd(usage: dict) -> dict:
    """Same DeepSeek price table as ``evals/run_planner_eval.py`` (checked 2026-09-24)."""
    prompt = usage.get("prompt_tokens", 0)
    hit = usage.get("prompt_cache_hit_tokens", 0)
    miss = usage.get("prompt_cache_miss_tokens", 0)
    if hit + miss != prompt:
        miss = max(prompt - hit, 0)
    output = usage.get("completion_tokens", 0)
    tables = {"offpeak": {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.6},
              "peak": {"cache_hit": 0.006, "cache_miss": 0.3, "output": 1.2}}
    return {"currency": "USD", "pricing_checked_on": "2026-09-24",
            "price_source": "same table as evals/run_planner_eval.py estimate_cost",
            "per_million_tokens": tables,
            "estimated_cost_usd": {period: round((hit * p["cache_hit"] + miss * p["cache_miss"]
                                                  + output * p["output"]) / 1_000_000, 6)
                                   for period, p in tables.items()}}


Transport = Callable[[dict], dict]


class TracedToolCaller:
    """Call one strict function tool with retries; each attempt is separately traced."""

    def __init__(self, api_key: str | None, model: str = "deepseek-flash",
                 transport: Transport | None = None, max_retries: int = 3,
                 backoff_seconds: float = 2.0, timeout_seconds: float = 60.0,
                 stats: CallStats | None = None):
        if transport is None and not (api_key or "").strip():
            raise ValueError("API key required for live calls")
        self._api_key = api_key or "offline-test-transport"
        self.model = model
        self._transport = transport
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.timeout_seconds = timeout_seconds
        self.stats = stats or CallStats()

    def call(self, payload: dict, name: str,
             validate: Callable[[dict], dict] | None = None) -> dict:
        """Return {"arguments", "usage", "provider_trace_files", "attempts"}."""
        traces: list[str] = []
        last_error = "unknown"
        for attempt in range(self.max_retries + 1):
            client = DeepSeekPlanner(self._api_key, DeepSeekConfig(
                model=self.model, base_url=BETA_BASE_URL,
                timeout_seconds=self.timeout_seconds), transport=self._transport)
            self.stats.start()
            usage: dict = {}
            try:
                response = client._post(payload)
                usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
                arguments = extract_tool_arguments(response, name)
                if validate is not None:
                    arguments = validate(arguments)
                if client.last_trace_path:
                    traces.append(client.last_trace_path)
                if client._last_trace is not None:
                    client._last_trace.emit("tool_arguments_validated", tool=name)
                self.stats.finish(True, usage)
                return {"arguments": arguments, "usage": usage,
                        "model": str(response.get("model", self.model)),
                        "provider_trace_files": traces, "attempts": attempt + 1}
            except (DeepSeekPlannerError, ValueError, TypeError, KeyError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if client.last_trace_path:
                    traces.append(client.last_trace_path)
                if client._last_trace is not None:
                    client._last_trace.emit("attempt_rejected", tool=name, attempt=attempt + 1,
                                            error=last_error)
                self.stats.finish(False, usage)
                if attempt < self.max_retries:
                    sleep(self.backoff_seconds * (2 ** attempt))
        raise ToolCallError(f"{name} failed after {self.max_retries + 1} attempts: {last_error}")
