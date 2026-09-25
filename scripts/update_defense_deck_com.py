"""Update the existing editable defense deck through Microsoft PowerPoint COM."""

from __future__ import annotations

from pathlib import Path
import zipfile

import win32com.client


SOURCE = Path("deliverables/药物重定位Agent_答辩稿_v3.pptx")
OUTPUT = Path("deliverables/药物重定位Agent_答辩稿_v4.pptx")
PREVIEWS = Path("artifacts/slide_build_v4")


def set_text(slide, shape_id: int, value: str) -> None:
    for shape in slide.Shapes:
        if shape.Id == shape_id:
            shape.TextFrame.TextRange.Text = value
            return
    raise RuntimeError(f"Shape {shape_id} missing on slide {slide.SlideIndex}")


def set_notes(slide, value: str) -> None:
    for shape in slide.NotesPage.Shapes:
        if shape.HasTextFrame and "Notes Placeholder" in shape.Name:
            shape.TextFrame.TextRange.Text = value
            return
    raise RuntimeError(f"Notes placeholder missing on slide {slide.SlideIndex}")


def update_chart(slide, edits: dict[tuple[int, int], object]) -> None:
    charts = [shape.Chart for shape in slide.Shapes if shape.HasChart]
    if len(charts) != 1:
        raise RuntimeError(f"Expected one chart on slide {slide.SlideIndex}")
    chart = charts[0]
    chart.ChartData.Activate()
    workbook = chart.ChartData.Workbook
    excel = workbook.Application
    try:
        sheet = workbook.Worksheets(1)
        for (row, column), value in edits.items():
            sheet.Cells(row, column).Value = value
        chart.Refresh()
    finally:
        workbook.Close(True)
        excel.Quit()


def main() -> None:
    source = SOURCE.resolve()
    output = OUTPUT.resolve()
    previews = PREVIEWS.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    previews.mkdir(parents=True, exist_ok=True)

    app = win32com.client.DispatchEx("PowerPoint.Application")
    app.DisplayAlerts = 1
    presentation = None
    try:
        presentation = app.Presentations.Open(str(source), WithWindow=False)
        if presentation.Slides.Count != 9:
            raise RuntimeError("Expected nine source slides")

        slide3 = presentation.Slides(3)
        set_text(slide3, 8, "961 个重合 landmark 基因，Top-10 精确签名 ID 已恢复")
        set_notes(slide3, "GEO GSE32863: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32863\nGSE92742: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742\nEH3226 构建方法与签名 ID 重建见 data/manifests/eh3226-luad.json、data/manifests/luad-top10-signatures-v1.json 和 configs/luad_top10_signature_ids.csv。")

        slide5 = presentation.Slides(5)
        update_chart(slide5, {
            (4, 1): "B2 嵌套", (4, 2): 73.71, (4, 3): "B2 嵌套", (4, 4): 47.02,
            (5, 1): "LogisticMF 嵌套", (5, 2): 78.26,
            (5, 3): "LogisticMF 嵌套", (5, 4): 64.74,
        })
        set_text(slide5, 4, "全局 AUC × 100，B2 使用 3 个外层种子，LogisticMF 使用 5 个外层种子")
        set_notes(slide5, "数据：TRANSCRIPT v2.0.0，https://zenodo.org/records/7982976\n拆分与全局 AUC：stanscofi 2.0.1。B2 使用三折内层调参与 3 个外层种子；LogisticMF 使用三折内层调参与 5 个外层种子。完整均值、标准差、NDCG 和泄漏审计见 docs/benchmark_results.md 与 benchmark/results/nested_cv_official_summary.json。未知 0 按全局 AUC 约定计入非正类，不代表临床失败。")

        slide7 = presentation.Slides(7)
        set_text(slide7, 5, "逐签名身份与证据缺口")
        set_text(slide7, 4, "10/10 源签名 ID 已恢复，十个候选仍均为“证据不足”")
        set_notes(slide7, "逐签名交叉表：configs/luad_top10_signature_ids.csv。重建依据是 GSE92742 校验固定的 sig_info/pert_info 和 EH3226 发布的源顺序去重规则。图表显示恢复精确 GSE 签名后仍存在的 Broad Repurposing Hub 跨来源核对状态。候选级支持与反对证据见 docs/luad_top10_evidence_matrix.md。")

        slide8 = presentation.Slides(8)
        set_text(slide8, 4, "工具、参数、输入和模式权限必须通过本地验证")
        set_text(slide8, 5, "100 例独立 v3：规则 90/100，DeepSeek 98/100")
        set_text(slide8, 6, "两处 DeepSeek 失败保留，未回改题集或 Planner")
        set_text(slide8, 7, "32 项测试通过，Planner 不接收原始矩阵或测试标签")
        set_notes(slide8, "v3 holdout 文件在任何调用前以提交 4611c88 冻结，SHA-256 为 0803e9bb8bfa87b61bd5832b89a76dde8cf40ead9be350015d7173f6d28dda5f。结果见 docs/planner_eval_results.md、benchmark/results/planner_eval_v3_rule.json 和 planner_eval_v3_deepseek_flash.json。一次 DeepSeek 运行不能估计随机波动。")

        slide9 = presentation.Slides(9)
        set_text(slide9, 4, "Top-10 的精确签名 ID 已恢复，疗效证据仍不足")
        set_text(slide9, 5, "五种子嵌套 CV 与 100 例独立 holdout 已完成")
        set_text(slide9, 6, "待执行剂量、亚型、NR3C1 依赖和联合用药实验")
        set_notes(slide9, "当前状态见 docs/project_status.md。体外验证的剂量范围、细胞亚型、NR3C1 机制对照、联合用药矩阵和预设门槛见 docs/luad_experimental_validation_protocol.md。工作区没有湿实验资源，因此没有把方案写成已完成验证。TypeSafe Jev 仍缺少凭据。任何候选都不构成临床治疗建议。")

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
    print(f"Updated {output}")
    print(f"Rendered 9 slides to {previews}")


if __name__ == "__main__":
    main()
