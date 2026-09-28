"""Preserve the editable v4 deck while aligning v5 with official NS-AUC."""

from __future__ import annotations

from pathlib import Path
import zipfile

import win32com.client

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run
from scripts.update_defense_deck_com import set_notes, set_text, update_chart


SOURCE = Path("deliverables/药物重定位Agent_答辩稿_v4.pptx")
OUTPUT = Path("deliverables/药物重定位Agent_答辩稿_v5.pptx")
PREVIEWS = Path("artifacts/slide_build_v5")


def _main(trace: TraceRecorder) -> None:
    source, output, previews = SOURCE.resolve(), OUTPUT.resolve(), PREVIEWS.resolve()
    if output.exists():
        raise ValueError("v5 deck already exists; refusing to overwrite the final version")
    trace.emit("source_verified", path=str(source), sha256=sha256_file(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    previews.mkdir(parents=True, exist_ok=True)

    app = win32com.client.DispatchEx("PowerPoint.Application")
    app.DisplayAlerts = 1
    presentation = None
    try:
        presentation = app.Presentations.Open(str(source), WithWindow=False)
        if presentation.Slides.Count != 9:
            raise RuntimeError("Expected the nine-slide source deck")
        slide5 = presentation.Slides(5)
        update_chart(slide5, {
            (2, 1): "B1 表达反转", (2, 2): 48.16,
            (2, 3): "B1 表达反转", (2, 4): 54.17,
            (3, 1): "B0p 训练流行度", (3, 2): 50.00,
            (3, 3): "B0p 训练流行度", (3, 4): 50.00,
            (4, 1): "B2 固定融合", (4, 2): 52.22,
            (4, 3): "B2 固定融合", (4, 4): 50.19,
            (5, 1): "LogisticMF 论文结果", (5, 2): 67.04,
            (5, 3): "LogisticMF 论文结果", (5, 4): 65.75,
        })
        set_text(slide5, 4, "官方 NS-AUC × 100；100 次相同种子、五折选模。弱相关外层测试集重复")
        set_notes(slide5, "TRANSCRIPT v2.0.0 与 RECeSS 官方 Runner，100 个同序种子、20% 测试、五折选模。B1、B0p、B2 为本项目官方 Runner 输出；LogisticMF 为作者发表的同协议结果，非本地 100 次重跑。数值为 NS-AUC × 100，不与早期全局 AUC 混比。弱相关拆分 100 个种子共享同一个外层测试集。来源：docs/benchmark_results.md 与 benchmark/results/recess_official_b2_vs_11.json。")
        trace.emit("benchmark_slide_updated", slide=5, metric="official NS-AUC",
                   models=["B1", "B0p", "B2", "LogisticMF"])

        slide8 = presentation.Slides(8)
        set_text(slide8, 5, "冻结的 100 例 v3：规则 90/100，DeepSeek 98/100")
        set_text(slide8, 6, "旧 DeepSeek 结果无原始响应 trace，不能补造内部原因")
        set_text(slide8, 7, "58 项测试通过；新规则 v3 trace 逐条评分 90/100")
        set_notes(slide8, "v3 holdout 在任何旧 v3 调用前以提交 4611c88 冻结。旧结果：benchmark/results/planner_eval_v3_rule.json、planner_eval_v3_deepseek_flash.json。新增规则回归运行在同一题集上为 90/100，并经 evals/grade_planner_traces.py 检查工具名、参数、安全边界和 trace 完整性；这是回归检查而非第二份独立 holdout。98/100 是工具选择，不是药物疗效或 NS-AUC。")
        trace.emit("agent_slide_updated", slide=8, historical_deepseek=98,
                   new_rule_trace_grade=90, tests=58)

        slide9 = presentation.Slides(9)
        set_text(slide9, 3, "官方 NS-AUC：B2 不及论文强基线")
        set_text(slide9, 4, "Top-10 签名可追溯，疗效证据仍不足")
        set_text(slide9, 5, "五个方法选择分区已冻结，尚无新模型成绩")
        set_text(slide9, 6, "湿实验无资源；剂量、亚型与组合仅为方案")
        set_notes(slide9, "B2 与 11 个作者模型在同一官方协议下比较，随机/弱相关 NS-AUC 为 0.5222/0.5019，分列 9/12 与 8/12。一次 DeepSeek 预选只涉及两个拆分，不能证明泛化。方法选择 v2 的五个互不重叠疾病分区已于 8b14325 冻结；它们仍来自 TRANSCRIPT，不是外部数据集。由于当前无新 API 密钥，还没有本轮大模型选择或分区结果。当前无实体化合物、细胞模型及仪器，湿实验未执行。候选不构成临床建议。")
        trace.emit("conclusion_slide_updated", slide=9, wetlab_completed=False,
                   method_selection_outcome_available=False)

        presentation.SaveAs(str(output), 24)
        presentation.Close()
        presentation = app.Presentations.Open(str(output), WithWindow=False)
        for slide in presentation.Slides:
            slide.Export(str(previews / f"slide-{slide.SlideIndex}.png"), "PNG", 1600, 900)
    finally:
        if presentation is not None:
            presentation.Close()
        app.Quit()
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt PPTX member: {bad}")
        chart_count = sum(name.startswith("ppt/charts/chart") and name.endswith(".xml")
                          for name in archive.namelist())
        if chart_count < 2:
            raise RuntimeError("Native editable charts were lost")
    trace.emit("deck_saved", path=str(output), sha256=sha256_file(output),
               slide_count=9, native_chart_count=chart_count)
    print(f"Updated {output}; rendered 9 slides to {previews}")


def main() -> None:
    traced_run("defense_deck_v5", _main)


if __name__ == "__main__":
    main()
