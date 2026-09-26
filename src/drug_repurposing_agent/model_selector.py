"""Prescore LLM choice among frozen TRANSCRIPT methods, without test metrics."""

from __future__ import annotations

import json
from typing import Callable


METHODS = ("B0p", "B1k", "B1", "B2")
SPLITS = ("random_simple", "weakly_correlated")
Transport = Callable[[dict[str, object]], dict[str, object]]


def validate_config(config: dict) -> None:
    if not isinstance(config, dict) or set(config) != {"version", "purpose", "dataset", "methods", "splits"}:
        raise ValueError("Unexpected selector input fields")
    if config["version"] != "component_selector_v1_prescore":
        raise ValueError("Unexpected selector version")
    if set(config["methods"]) != set(METHODS) or set(config["splits"]) != set(SPLITS):
        raise ValueError("Method or split inventory changed")
    if set(config["dataset"]) != {"name", "drugs", "diseases", "known_positive_pairs",
                                   "explicit_negative_pairs", "unknown_pairs", "feature_type"}:
        raise ValueError("Dataset inventory changed or includes forbidden metrics")


def select_methods(config: dict, transport: Transport, model: str = "deepseek-flash") -> dict:
    validate_config(config)
    schema = {"type": "object", "properties": {
        split: {"type": "object", "properties": {
            "method": {"type": "string", "enum": list(METHODS)},
            "reason": {"type": "string"}},
            "required": ["method", "reason"], "additionalProperties": False}
        for split in SPLITS},
        "required": list(SPLITS), "additionalProperties": False}
    payload: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "You are selecting a frozen research prediction method BEFORE seeing the "
                "component ablation scores. Call submit_method_choice exactly once. "
                "Choose one of B0p, B1k, B1, B2 separately for each split. "
                "Use only supplied dataset/split/method descriptions; do not assume any "
                "unseen holdout performance. Keep each reason under 150 characters. "
                "Explain uncertainty; this is not treatment advice.")},
            {"role": "user", "content": json.dumps(config, ensure_ascii=False)},
        ],
        "tools": [{"type": "function", "function": {
            "name": "submit_method_choice", "strict": True,
            "description": "Record one prescore method choice for each split.",
            "parameters": schema}}],
        "tool_choice": "required", "thinking": {"type": "disabled"},
        "max_tokens": 800, "stream": False,
    }
    response = transport(payload)
    try:
        if response["choices"][0].get("finish_reason") == "length":
            raise ValueError("Truncated method choice")
        calls = response["choices"][0]["message"]["tool_calls"]
        if len(calls) != 1 or calls[0]["function"]["name"] != "submit_method_choice":
            raise ValueError("Expected exactly one method-choice tool call")
        choices = json.loads(calls[0]["function"]["arguments"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid method-choice tool response") from exc
    if not isinstance(choices, dict) or set(choices) != set(SPLITS):
        raise ValueError("Missing or extra split choice")
    for split in SPLITS:
        choice = choices[split]
        if not isinstance(choice, dict) or set(choice) != {"method", "reason"}:
            raise ValueError("Invalid split choice")
        if choice["method"] not in METHODS or not isinstance(choice["reason"], str):
            raise ValueError("Unknown method or invalid reason")
        if not 1 <= len(choice["reason"].strip()) <= 2000:
            raise ValueError("Reason must be nonempty and concise")
    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    return {
        "status": "prescore_method_selection_not_validated",
        "model": str(response.get("model", model)),
        "choices": choices,
        "provider_usage": {key: int(value) for key, value in usage.items()
                           if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                           and isinstance(value, int)},
        "interpretation": "Choice made without outer-test metrics; benchmark scores must be checked separately.",
    }
