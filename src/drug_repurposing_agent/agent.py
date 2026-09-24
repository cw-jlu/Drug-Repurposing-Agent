"""Controlled natural-language planner and tool executor.

The planner never receives benchmark labels or raw matrices.  It sees only a
small inventory of available inputs and an allow-listed set of tool schemas.
This keeps natural-language routing separate from deterministic scientific
computation and makes a future LLM planner replaceable and testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4

from .luad_case import build_luad_case
from .workflow import Mode, run_expression_workflow


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, object]
    modes: tuple[Mode, ...]

    def public_schema(self) -> dict[str, object]:
        """Return the provider-neutral function schema exposed to a planner."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


TOOLS = {
    "rank_transcriptome": ToolSpec(
        name="rank_transcriptome",
        description=("Validate aligned disease/drug expression matrices and produce "
                     "label-free reversal, connectivity, and RRF rankings."),
        parameters={
            "type": "object",
            "properties": {"top_k": {"type": "integer", "minimum": 1, "maximum": 1000}},
            "additionalProperties": False,
        },
        modes=(Mode.BENCHMARK_STRICT, Mode.RESEARCH_OPEN),
    ),
    "package_luad_case": ToolSpec(
        name="package_luad_case",
        description=("Validate and package the frozen LUAD expression screen, identity "
                     "audit, controls, evidence ledgers, provenance, and trace."),
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        modes=(Mode.RESEARCH_OPEN,),
    ),
    "manual_review": ToolSpec(
        name="manual_review",
        description="Stop safely when the requested task is unsupported or underspecified.",
        parameters={
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
            "additionalProperties": False,
        },
        modes=(Mode.BENCHMARK_STRICT, Mode.RESEARCH_OPEN),
    ),
}


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentPlan:
    task: str
    rationale: str
    calls: tuple[ToolCall, ...]
    planner: str

    @classmethod
    def from_dict(cls, value: dict[str, object], planner: str) -> "AgentPlan":
        if not isinstance(value, dict):
            raise ValueError("Planner output must be a JSON object")
        raw_calls = value.get("calls")
        if not isinstance(raw_calls, list) or not raw_calls:
            raise ValueError("Planner output requires a nonempty calls list")
        calls = []
        for raw in raw_calls:
            if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
                raise ValueError("Each tool call requires a string name")
            arguments = raw.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object")
            calls.append(ToolCall(raw["name"], arguments))
        return cls(
            task=str(value.get("task", "unspecified")),
            rationale=str(value.get("rationale", "")),
            calls=tuple(calls),
            planner=planner,
        )


@dataclass(frozen=True)
class AgentInputs:
    """Paths controlled by the executor, never invented by the planner."""
    items: Path | None = None
    users: Path | None = None
    screen_dir: Path | None = None
    disease_manifest: Path | None = None
    screen_manifest: Path | None = None

    def inventory(self) -> tuple[str, ...]:
        return tuple(name for name, value in asdict(self).items() if value is not None)


@dataclass(frozen=True)
class PlanningContext:
    mode: Mode
    available_inputs: tuple[str, ...]


class Planner(Protocol):
    name: str

    def plan(self, question: str, context: PlanningContext,
             tools: tuple[dict[str, object], ...]) -> AgentPlan: ...


class RulePlanner:
    """Deterministic fallback used when no external LLM planner is configured."""

    name = "rule_fallback_v1"
    _luad_terms = ("luad", "lung adenocarcinoma", "肺腺癌")
    _drug_terms = ("drug", "repurpos", "candidate", "药物", "重定位", "候选")

    def __init__(self, top_k: int = 100):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 1000:
            raise ValueError("top_k must be an integer in [1, 1000]")
        self.top_k = top_k

    def plan(self, question: str, context: PlanningContext,
             tools: tuple[dict[str, object], ...]) -> AgentPlan:
        del tools
        normalized = question.strip().lower()
        if not normalized:
            raise ValueError("Question must not be empty")
        if any(term in normalized for term in self._luad_terms):
            return AgentPlan(
                task="luad_case",
                rationale="The request names lung adenocarcinoma/LUAD.",
                calls=(ToolCall("package_luad_case"),),
                planner=self.name,
            )
        if any(term in normalized for term in self._drug_terms):
            return AgentPlan(
                task="transcriptomic_ranking",
                rationale="The request asks for drug-repurposing candidate ranking.",
                calls=(ToolCall("rank_transcriptome", {"top_k": self.top_k}),),
                planner=self.name,
            )
        return AgentPlan(
            task="unsupported",
            rationale="No supported biomedical workflow can be selected safely.",
            calls=(ToolCall("manual_review", {"reason": "unsupported_or_ambiguous_request"}),),
            planner=self.name,
        )


class StructuredPlanner:
    """Adapter boundary for an LLM that returns provider-neutral tool calls.

    ``invoke`` receives a JSON-serializable request containing the user question,
    safe context, and function schemas.  Provider credentials and network logic
    stay outside the scientific package, while returned calls undergo the same
    allow-list and argument validation as the deterministic fallback.
    """

    name = "structured_external_planner"

    def __init__(self, invoke: Callable[[dict[str, object]], dict[str, object]]):
        self.invoke = invoke

    def plan(self, question: str, context: PlanningContext,
             tools: tuple[dict[str, object], ...]) -> AgentPlan:
        raw = self.invoke({
            "question": question,
            "context": {"mode": context.mode.value,
                        "available_inputs": list(context.available_inputs)},
            "tools": list(tools),
            "required_output": {"task": "string", "rationale": "string",
                                "calls": [{"name": "string", "arguments": {}}]},
        })
        return AgentPlan.from_dict(raw, self.name)


def _validate_call(call: ToolCall, mode: Mode) -> None:
    spec = TOOLS.get(call.name)
    if spec is None:
        raise ValueError(f"Tool is not allow-listed: {call.name}")
    if mode not in spec.modes:
        raise ValueError(f"Tool {call.name} is not permitted in {mode.value}")
    allowed = set(spec.parameters.get("properties", {}))
    unknown = set(call.arguments) - allowed
    if unknown:
        raise ValueError(f"Unexpected arguments for {call.name}: {sorted(unknown)}")
    if call.name == "rank_transcriptome":
        top_k = call.arguments.get("top_k", 100)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 1000:
            raise ValueError("rank_transcriptome.top_k must be an integer in [1, 1000]")
    if call.name == "manual_review" and not str(call.arguments.get("reason", "")).strip():
        raise ValueError("manual_review.reason is required")


def _missing_inputs(call: ToolCall, inputs: AgentInputs) -> list[str]:
    required = {
        "rank_transcriptome": ("items", "users"),
        "package_luad_case": ("screen_dir", "disease_manifest", "screen_manifest"),
        "manual_review": (),
    }[call.name]
    return [name for name in required if getattr(inputs, name) is None]


def run_agent_task(question: str, inputs: AgentInputs, output: Path,
                   mode: Mode = Mode.RESEARCH_OPEN,
                   planner: Planner | None = None) -> dict[str, object]:
    """Plan and execute one bounded research task with a complete audit trace."""
    run_id = uuid4().hex
    trace: list[dict[str, object]] = []

    def event(stage: str, **details: object) -> None:
        trace.append({"time": datetime.now(timezone.utc).isoformat(),
                      "stage": stage, **details})

    output.mkdir(parents=True, exist_ok=True)
    active_planner = planner or RulePlanner()
    context = PlanningContext(mode, inputs.inventory())
    public_tools = tuple(spec.public_schema() for spec in TOOLS.values() if mode in spec.modes)
    report: dict[str, object] = {
        "run_id": run_id,
        "question": question,
        "mode": mode.value,
        "status": "planning",
        "planner": active_planner.name,
        "available_inputs": list(context.available_inputs),
        "tool_results": [],
        "trace": trace,
    }

    def save() -> None:
        (output / "agent_run.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    event("request_received", question_length=len(question), mode=mode.value)
    try:
        plan = active_planner.plan(question, context, public_tools)
        for call in plan.calls:
            _validate_call(call, mode)
        report["plan"] = {
            "task": plan.task,
            "rationale": plan.rationale,
            "calls": [asdict(call) for call in plan.calls],
        }
        metadata = getattr(active_planner, "last_metadata", None)
        if isinstance(metadata, dict) and metadata:
            report["planner_metadata"] = metadata
        event("plan_validated", task=plan.task, calls=[call.name for call in plan.calls])
    except Exception as exc:
        report.update(status="blocked", error={"type": type(exc).__name__, "message": str(exc)})
        event("plan_blocked", error_type=type(exc).__name__)
        save()
        return report

    for index, call in enumerate(plan.calls, start=1):
        missing = _missing_inputs(call, inputs)
        if missing:
            report.update(status="needs_input", missing_inputs=missing)
            event("tool_waiting_for_input", tool=call.name, missing=missing)
            save()
            return report
        event("tool_started", tool=call.name, sequence=index)
        try:
            if call.name == "rank_transcriptome":
                tool_output = output / "expression_ranking"
                result = run_expression_workflow(
                    inputs.items, inputs.users, tool_output, mode,
                    int(call.arguments.get("top_k", 100)),
                )
                summary = {"tool": call.name, "status": result["status"],
                           "manifest": str(tool_output / "manifest.json"), "qc": result["qc"]}
            elif call.name == "package_luad_case":
                tool_output = output / "luad_case"
                result = build_luad_case(inputs.screen_dir, inputs.disease_manifest,
                                         inputs.screen_manifest, tool_output)
                summary = {"tool": call.name, "status": result["status"],
                           "report": str(tool_output / "case_report.json"),
                           "candidates": len(result["candidates"])}
            else:
                summary = {"tool": call.name, "status": "manual_review_required",
                           "reason": call.arguments["reason"]}
                report["tool_results"].append(summary)
                report["status"] = "needs_review"
                event("manual_review_required", reason=call.arguments["reason"])
                save()
                return report
            report["tool_results"].append(summary)
            event("tool_completed", tool=call.name, sequence=index, status=summary["status"])
        except Exception as exc:
            report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
            event("tool_failed", tool=call.name, sequence=index, error_type=type(exc).__name__)
            save()
            return report

    report["status"] = "completed"
    event("run_completed", tool_count=len(plan.calls))
    save()
    return report
