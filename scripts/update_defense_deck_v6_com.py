"""Preserve editable v5 slides while correcting the now-obsolete v2 conclusion."""

from __future__ import annotations

from pathlib import Path
import zipfile

import win32com.client

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run
from scripts.update_defense_deck_com import set_notes, set_text


SOURCE = Path("deliverables/药物重定位Agent_答辩稿_v5.pptx")
OUTPUT = Path("deliverables/药物重定位Agent_答辩稿_v6.pptx")
PREVIEWS = Path("artifacts/slide_build_v6")


def _main(trace: TraceRecorder) -> None:
    source, output, previews = SOURCE.resolve(), OUTPUT.resolve(), PREVIEWS.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    previews.mkdir(parents=True, exist_ok=True)
    if output.exists():
        old_sha256 = sha256_file(output)
        archive = previews / f"previous-v6-{old_sha256[:12]}.pptx"
        if archive.exists():
            raise ValueError("v6 and its previous archive both exist; refusing replacement")
        output.rename(archive)
        trace.emit("previous_draft_archived", path=str(archive), sha256=old_sha256)
    trace.emit("source_verified", path=str(source), sha256=sha256_file(source))
    app = win32com.client.DispatchEx("PowerPoint.Application")
    app.DisplayAlerts = 1
    presentation = None
    try:
        presentation = app.Presentations.Open(str(source), WithWindow=False)
        if presentation.Slides.Count != 9:
            raise RuntimeError("Expected nine source slides")
        slide5 = presentation.Slides(5)
        set_notes(slide5, "TRANSCRIPT v2.0.0、官方 Runner 100 同序种子、五折选模。B2 官方随机/弱相关 NS-AUC 0.5222/0.5019，分别列 9/12 和 8/12；不能与全局 AUC 混比。B1k 官方得分很低；在各复算的一个冻结种子上，分数和保存值精确相符，大量正负对同分，官方严格比较同分记零；反向分数更差。该诊断不替换 100 次官方数值。弱相关 100 种子复用同一个外层测试集。来源：docs/benchmark_results.md、docs/b1k_metric_audit.md。")
        slide8 = presentation.Slides(8)
        set_text(slide8, 7, "完整测试通过；新规则 v3 trace 逐条评分 90/100")
        set_notes(slide8, "历史冻结 v3 规划/工具调用 holdout：规则 90/100，DeepSeek 98/100；旧 DeepSeek 原始 provider trace 不可补造。新规则逐条 trace 回归同为 90/100，并非第二份独立 holdout。另对历史 LUAD 研究行动分诊单例做身份、引文范围和弃权检查，三项行动均通过；这不是新的 Agent holdout、药效或临床准确率。来源：docs/planner_eval_results.md、artifacts/reports/luad_research_triage_grade.json。")
        slide9 = presentation.Slides(9)
        set_text(slide9, 5, "五分区修订：选法较 B2 低 0.110 NS-AUC")
        set_notes(slide9, "五分区 15 条选择在结果计算前冻结，provider-visible trace 15/15 核对。原始官方协议因第二分区仅一条明确负例无法分层而失败；单独标注的事后正例/未知修订实验完成五分区、四方法、20 外层种子和五折选模。DeepSeek 所选方法平均 NS-AUC 0.40653，固定 B2 0.51684，分区配对均差 −0.11031，0/5 分区为正。修订结果不是原预注册协议，也不属独立外部数据。下一步需要独立数据以及外层训练集内的交叉验证证据；已看过的五分区不能再作为新 holdout。LUAD Top-10 仅为研究假设，未做湿实验。来源：docs/method_selection_partition_v2.md、docs/method_selection_v3_protocol.md、artifacts/reports/method_selection_partition_v2_positive_only_score.json。")
        trace.emit("slides_updated", slides=[5, 8, 9], amended_delta=-0.11031,
                   new_holdout_completed=False, wetlab_completed=False)
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
        if bad:
            raise RuntimeError(f"Corrupt PPTX member: {bad}")
        chart_count = sum(name.startswith("ppt/charts/chart") and name.endswith(".xml")
                          for name in archive.namelist())
        if chart_count < 2:
            raise RuntimeError("Native editable charts were lost")
    trace.emit("deck_saved", path=str(output), sha256=sha256_file(output),
               slide_count=9, native_chart_count=chart_count)
    print(f"Updated {output}; rendered 9 slides to {previews}")


if __name__ == "__main__":
    traced_run("defense_deck_v6", _main)
