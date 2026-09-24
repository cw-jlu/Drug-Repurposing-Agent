"""DeepSeek-backed planner for the controlled Agent tool boundary."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from time import perf_counter
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .agent import AgentPlan, PlanningContext, ToolCall


class DeepSeekPlannerError(RuntimeError):
    """A sanitized planner failure that never includes credentials or raw responses."""


Transport = Callable[[dict[str, object]], dict[str, object]]


@dataclass(frozen=True)
class DeepSeekConfig:
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    timeout_seconds: float = 45.0


class DeepSeekPlanner:
    """Use DeepSeek function calling while retaining local validation and execution."""

    def __init__(self, api_key: str, config: DeepSeekConfig | None = None,
                 transport: Transport | None = None):
        if not api_key.strip():
            raise ValueError("DeepSeek API key is required")
        self._api_key = api_key
        self.config = config or DeepSeekConfig()
        if not self.config.base_url.startswith("https://"):
            raise ValueError("DeepSeek base_url must use HTTPS")
        self._transport = transport
        self.name = f"deepseek_tool_calling:{self.config.model}"
        self.last_metadata: dict[str, object] = {}

    @classmethod
    def from_env(cls, model: str | None = None) -> "DeepSeekPlanner":
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise ValueError("Set DEEPSEEK_API_KEY in the process environment")
        config = DeepSeekConfig(
            model=model or os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        return cls(key, config)

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        if self._transport is not None:
            return self._transport(payload)
        request = Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise DeepSeekPlannerError(f"DeepSeek HTTP error {exc.code}") from exc
        except URLError as exc:
            raise DeepSeekPlannerError("DeepSeek network request failed") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DeepSeekPlannerError("DeepSeek returned an invalid JSON response") from exc

    def plan(self, question: str, context: PlanningContext,
             tools: tuple[dict[str, object], ...]) -> AgentPlan:
        if not question.strip():
            raise ValueError("Question must not be empty")
        if not tools:
            raise ValueError("No tools are available in this mode")
        tool_payload = [
            {"type": "function", "function": schema}
            for schema in tools
        ]
        system = (
            "You route biomedical research requests to exactly one provided function. "
            "Call manual_review for unsupported or ambiguous requests, patient-specific "
            "treatment advice, destructive actions, clinical publication, or when no "
            "available function safely matches. Use rank_transcriptome for label-free "
            "disease-drug expression ranking. Use package_luad_case only when that function "
            "is provided and the request concerns the frozen LUAD/lung adenocarcinoma case. "
            "Never invent functions or parameters. Do not answer the biomedical question "
            "directly; return one function call."
        )
        user = json.dumps({
            "question": question,
            "mode": context.mode.value,
            "available_inputs": list(context.available_inputs),
        }, ensure_ascii=False)
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "tools": tool_payload,
            "tool_choice": "required",
            "thinking": {"type": "disabled"},
            "stream": False,
        }
        started = perf_counter()
        response = self._post(payload)
        latency_ms = round((perf_counter() - started) * 1000, 1)
        try:
            choice = response["choices"][0]
            message = choice["message"]
            raw_calls = message["tool_calls"]
            if not isinstance(raw_calls, list) or len(raw_calls) != 1:
                raise KeyError("expected exactly one tool call")
            raw_call = raw_calls[0]["function"]
            name = raw_call["name"]
            arguments = json.loads(raw_call.get("arguments") or "{}")
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise TypeError("invalid function call")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise DeepSeekPlannerError("DeepSeek returned an invalid tool call") from exc
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        self.last_metadata = {
            "provider": "deepseek",
            "model": str(response.get("model", self.config.model)),
            "latency_ms": latency_ms,
            "finish_reason": str(choice.get("finish_reason", "")),
            "usage": {
                key: int(value) for key, value in usage.items()
                if key in {"prompt_tokens", "completion_tokens", "total_tokens",
                           "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"}
                and isinstance(value, int)
            },
        }
        if isinstance(response.get("system_fingerprint"), str):
            self.last_metadata["system_fingerprint"] = response["system_fingerprint"]
        task_by_tool = {
            "rank_transcriptome": "transcriptomic_ranking",
            "package_luad_case": "luad_case",
            "manual_review": "manual_review",
        }
        return AgentPlan(
            task=task_by_tool.get(name, "unknown"),
            rationale="Selected by DeepSeek function calling; execution remains locally validated.",
            calls=(ToolCall(name, arguments),),
            planner=self.name,
        )
