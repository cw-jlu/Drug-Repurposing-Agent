"""Controlled natural-language planner and tool executor.

The planner never receives benchmark labels or raw matrices.  It sees only a
small inventory of available inputs and an allow-listed set of tool schemas.
This keeps natural-language routing separate from deterministic scientific
computation and makes a future LLM planner replaceable and testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Callable, Protocol

from .luad_case import build_luad_case
from .trace import TraceRecorder, redact
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

    name = "rule_fallback_v2"
    _luad_terms = ("luad", "lung adenocarcinoma", "肺腺癌")
    _rank_terms = (
        "drug", "repurpos", "candidate", "药物", "重定位", "候选",
        "transcriptom", "expression", "signature", "spearman", "rrf",
        "表达谱", "表达矩阵", "转录组", "反向匹配", "反转分数",
    )
    _unsafe_terms = (
        "患者", "处方", "剂量", "疗程", "治愈", "已证实疗法", "最有效",
        "patient", "prescribe", "dose", "clinically proven", "patient benefit",
        "treatment advice", "delete", "remove failed", "overwrite", "删除", "覆盖",
        "modify the labels", "修改标签", "api key", "bypass", "忽略工具白名单",
        "pretend manual review", "无需医生审核", "skip review", "自动推荐",
        "train a new neural network", "训练一个新的神经网络",
    )
    _nonexecution_terms = (
        "不要执行", "不要运行", "without repackaging", "do not run",
        "explain what", "summarize", "列出", "局限", "statistically", "统计",
        "检查文件是否存在",
    )

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
        if any(term in normalized for term in self._unsafe_terms + self._nonexecution_terms):
            return AgentPlan(
                task="manual_review",
                rationale="The request is unsafe, non-executable, or requires human interpretation.",
                calls=(ToolCall("manual_review", {"reason": "safety_or_scope_boundary"}),),
                planner=self.name,
            )
        if any(term in normalized for term in self._luad_terms):
            required = {"screen_dir", "disease_manifest", "screen_manifest"}
            if context.mode != Mode.RESEARCH_OPEN or not required.issubset(context.available_inputs):
                return AgentPlan(
                    task="manual_review",
                    rationale="LUAD packaging is unavailable in this mode or lacks required inputs.",
                    calls=(ToolCall("manual_review", {"reason": "luad_mode_or_inputs"}),),
                    planner=self.name,
                )
            return AgentPlan(
                task="luad_case",
                rationale="The request names lung adenocarcinoma/LUAD.",
                calls=(ToolCall("package_luad_case"),),
                planner=self.name,
            )
        if any(term in normalized for term in self._rank_terms):
            if not {"items", "users"}.issubset(context.available_inputs):
                return AgentPlan(
                    task="manual_review",
                    rationale="Expression ranking requires both item and user matrices.",
                    calls=(ToolCall("manual_review", {"reason": "missing_expression_inputs"}),),
                    planner=self.name,
                )
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
        self.last_trace_path: str | None = None

    def plan(self, question: str, context: PlanningContext,
             tools: tuple[dict[str, object], ...]) -> AgentPlan:
        request = {
            "question": question,
            "context": {"mode": context.mode.value,
                        "available_inputs": list(context.available_inputs)},
            "tools": list(tools),
            "required_output": {"task": "string", "rationale": "string",
                                "calls": [{"name": "string", "arguments": {}}]},
        }
        recorder = TraceRecorder("structured_planner")
        self.last_trace_path = str(recorder.path)
        recorder.emit("model_request", payload=request)
        try:
            raw = self.invoke(request)
            recorder.emit("model_response", response=raw)
            plan = AgentPlan.from_dict(raw, self.name)
            recorder.emit("tool_call_parsed", calls=[asdict(call) for call in plan.calls])
            return plan
        except Exception as exc:
            recorder.emit("model_or_parse_error", error_type=type(exc).__name__,
                          error=str(exc))
            raise


def validate_tool_call(call: ToolCall, mode: Mode) -> None:
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
    trace: list[dict[str, object]] = []
    output.mkdir(parents=True, exist_ok=True)
    recorder = TraceRecorder("agent", output / "traces")
    run_id = recorder.run_id

    def event(stage: str, **details: object) -> None:
        recorded = recorder.emit(stage, **details)
        trace.append({key: value for key, value in recorded.items()
                      if key not in {"schema_version", "run_id"}})

    active_planner = planner or RulePlanner()
    planner_secret = getattr(active_planner, "_api_key", "")
    if isinstance(planner_secret, str):
        recorder.add_secret(planner_secret)
    else:
        planner_secret = ""
    context = PlanningContext(mode, inputs.inventory())
    public_tools = tuple(spec.public_schema() for spec in TOOLS.values() if mode in spec.modes)
    report: dict[str, object] = {
        "run_id": run_id,
        "question": redact(question, (planner_secret,)),
        "mode": mode.value,
        "status": "planning",
        "planner": active_planner.name,
        "available_inputs": list(context.available_inputs),
        "tool_results": [],
        "trace": trace,
        "trace_file": str(recorder.path),
    }

    def save() -> None:
        (output / "agent_run.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    event("request_received", question=question, mode=mode.value,
          available_inputs=list(context.available_inputs), planner=active_planner.name)
    try:
        plan = active_planner.plan(question, context, public_tools)
        event("plan_proposed", task=plan.task, calls=[asdict(call) for call in plan.calls],
              provider_trace=getattr(active_planner, "last_trace_path", None))
        for call in plan.calls:
            validate_tool_call(call, mode)
        report["plan"] = {
            "task": plan.task,
            "rationale": redact(plan.rationale, (planner_secret,)),
            "calls": redact([asdict(call) for call in plan.calls], (planner_secret,)),
        }
        metadata = getattr(active_planner, "last_metadata", None)
        if isinstance(metadata, dict) and metadata:
            report["planner_metadata"] = metadata
        event("plan_validated", task=plan.task, calls=[asdict(call) for call in plan.calls],
              planner_metadata=metadata if isinstance(metadata, dict) else None)
    except Exception as exc:
        report.update(status="blocked", error={"type": type(exc).__name__,
                                               "message": redact(str(exc), (planner_secret,))})
        event("plan_blocked", error_type=type(exc).__name__, error=str(exc),
              provider_trace=getattr(active_planner, "last_trace_path", None))
        save()
        return report

    for index, call in enumerate(plan.calls, start=1):
        missing = _missing_inputs(call, inputs)
        if missing:
            report.update(status="needs_input", missing_inputs=missing)
            event("tool_waiting_for_input", tool=call.name, missing=missing)
            save()
            return report
        event("tool_started", tool=call.name, arguments=call.arguments, sequence=index)
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
            report.update(status="failed", error={"type": type(exc).__name__,
                                                  "message": redact(str(exc), (planner_secret,))})
            event("tool_failed", tool=call.name, sequence=index,
                  error_type=type(exc).__name__, error=str(exc))
            save()
            return report

    report["status"] = "completed"
    event("run_completed", tool_count=len(plan.calls))
    save()
    return report
