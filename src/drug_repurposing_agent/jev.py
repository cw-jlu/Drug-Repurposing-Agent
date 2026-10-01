"""Optional TypeSafe Jev adapter with deterministic validation and abstention.

No API request is made without an explicit client call and an API key. Model
answers never override source, citation, identity, or publication checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

from .trace import TraceRecorder


API_URL = "https://api.typesafe.ai/v1/systemone"
# Jev is also served through OpenCode Zen with an OpenCode API key.
OPENCODE_ZEN_URL = "https://opencode.ai/zen/v1/systemone"
# Cloudflare in front of OpenCode rejects urllib's default User-Agent (error 1010).
USER_AGENT = "drug-repurposing-agent/1.0"
QUESTION_TYPES = {"choice", "score", "noul"}


def choice_question(instructions: str, choices: dict[str, str]) -> dict:
    if not instructions or len(choices) < 2:
        raise ValueError("Choice requires instructions and at least two choices")
    return {"type": "choice", "instructions": instructions, "criteria": choices}


def score_question(instructions: str, levels: list[str]) -> dict:
    if not instructions or len(levels) < 2:
        raise ValueError("Score requires instructions and ordered levels")
    return {"type": "score", "instructions": instructions, "criteria": levels}


def noul_question(instructions: str, yes: str, no: str) -> dict:
    if not instructions or not yes or not no:
        raise ValueError("Noul requires question and both criteria")
    return {"type": "noul", "instructions": instructions,
            "criteria": {"true": yes, "false": no}}


def _env_value(name: str) -> str:
    """Read a process variable or the ignored local .env without logging it."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    env_file = Path(".env")
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            if line.strip().startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _probability(value: object) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid probability") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("Probability outside [0, 1]")
    return number


def validate_answer(question: dict, answer: dict) -> None:
    kind = question.get("type")
    if kind not in QUESTION_TYPES or answer.get("type") != kind:
        raise ValueError("Jev answer type mismatch")
    if kind == "noul":
        _probability(answer.get("noul"))
        return
    _probability(answer.get("confidence"))
    probs = answer.get("probabilities")
    if not isinstance(probs, dict):
        raise ValueError("Missing probability distribution")
    expected = (set(question["criteria"]) if kind == "choice"
                else {str(i) for i in range(len(question["criteria"]))})
    if set(probs) != expected:
        raise ValueError("Probability choices differ from request")
    values = {key: _probability(value) for key, value in probs.items()}
    if not math.isclose(sum(values.values()), 1.0, abs_tol=0.02):
        raise ValueError("Probabilities do not sum to one")
    if kind == "choice":
        selected = answer.get("choice")
        if selected not in expected or values[selected] < max(values.values()) - 1e-9:
            raise ValueError("Choice is inconsistent with probabilities")
    else:
        score = float(answer.get("score", float("nan")))
        weighted = sum(int(level) * value for level, value in values.items())
        if not math.isfinite(score) or abs(score - weighted) > 0.05:
            raise ValueError("Score is inconsistent with distribution")


class JevClient:
    def __init__(self, api_key: str, model: str = "jev-latest", timeout: int = 30,
                 api_url: str = API_URL, provider: str = "typesafe_jev"):
        if not api_key or not model:
            raise ValueError("API key and model are required")
        if not api_url.startswith("https://"):
            raise ValueError("Jev api_url must use HTTPS")
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        self.api_url = api_url
        self.provider = provider
        self.last_trace_path: str | None = None

    @classmethod
    def from_env(cls, model: str | None = None) -> "JevClient":
        """TypeSafe key first; otherwise an OpenCode key routed through OpenCode Zen."""
        key = _env_value("TYPESAFE_API_KEY")
        if key:
            return cls(key, model or "jev-latest")
        key = _env_value("OPENCODE_API_KEY")
        if key:
            return cls(key, model or "jev-1.13", api_url=OPENCODE_ZEN_URL, provider="opencode_zen_jev")
        raise RuntimeError("TYPESAFE_API_KEY or OPENCODE_API_KEY is unavailable")

    def ask(self, state: str | dict | list, questions: dict[str, dict]) -> dict:
        if not questions or any(q.get("type") not in QUESTION_TYPES for q in questions.values()):
            raise ValueError("At least one supported question is required")
        payload = {"state": state, "model": self.model, "questions": questions}
        trace = TraceRecorder("jev", secrets=(self._api_key,))
        self.last_trace_path = str(trace.path)
        trace.emit("model_request", provider=self.provider, api_url=self.api_url, payload=payload)
        started = perf_counter()
        try:
            result = self._ask_and_validate(payload, questions, trace)
            trace.emit("model_response_validated", model=result.get("model"),
                       usage=result.get("usage"),
                       latency_ms=round((perf_counter() - started) * 1000, 1))
            return result
        except Exception as exc:
            trace.emit("model_error", error_type=type(exc).__name__,
                       latency_ms=round((perf_counter() - started) * 1000, 1))
            raise

    def _ask_and_validate(self, payload: dict, questions: dict[str, dict],
                          trace: TraceRecorder) -> dict:
        request = Request(self.api_url, data=json.dumps(payload).encode("utf-8"),
                          headers={"Authorization": f"Bearer {self._api_key}",
                                   "Content-Type": "application/json",
                                   "User-Agent": USER_AGENT}, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        trace.emit("model_response", response=result)
        answers = result.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise ValueError("Jev response lacks requested answers")
        for name, question in questions.items():
            validate_answer(question, answers[name])
        if result.get("model") is None or not isinstance(result.get("usage"), dict):
            raise ValueError("Jev response lacks model or usage")
        return result

    def available_models(self) -> list[dict]:
        trace = TraceRecorder("jev_models", secrets=(self._api_key,))
        self.last_trace_path = str(trace.path)
        trace.emit("model_catalog_request", provider="typesafe_jev")
        try:
            models = self._available_models()
            trace.emit("model_catalog_response", models=models)
            return models
        except Exception as exc:
            trace.emit("model_catalog_error", error_type=type(exc).__name__)
            raise

    def _available_models(self) -> list[dict]:
        request = Request("https://api.typesafe.ai/v1/models",
                          headers={"Authorization": f"Bearer {self._api_key}"}, method="GET")
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        models = result.get("models")
        if not isinstance(models, list) or any(not isinstance(item, dict) or
                                                not isinstance(item.get("name"), str)
                                                for item in models):
            raise ValueError("Invalid Jev model list")
        return models


@dataclass(frozen=True)
class DecisionOutcome:
    action: str
    source: str
    reason: str
    confidence: float | None


def gate_choice(question: dict, answer: dict | None, *, high_risk: bool = False) -> DecisionOutcome:
    """Return only allowed actions; fail closed when unavailable or uncertain."""
    if question.get("type") != "choice":
        raise ValueError("Choice gate requires a choice question")
    if answer is None:
        return DecisionOutcome("manual_review", "rule_fallback", "provider_unavailable", None)
    validate_answer(question, answer)
    selected = answer["choice"]
    confidence = _probability(answer["confidence"])
    if selected == "publish_clinical":
        return DecisionOutcome("manual_review", "hard_rule", "clinical_publication_requires_review",
                               confidence)
    threshold = 0.9 if high_risk else 0.8
    if confidence < threshold:
        return DecisionOutcome("manual_review", "confidence_gate", "low_confidence",
                               confidence)
    return DecisionOutcome(selected, "jev", "passed_gate", confidence)
