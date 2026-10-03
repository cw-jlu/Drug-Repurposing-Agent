"""Record a chained receipt for the current report/deck and their visual QA files."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import fitz

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


def _main(trace: TraceRecorder) -> None:
    report = Path("deliverables/药物重定位Agent_课程设计报告_v5.docx")
    deck = Path("deliverables/药物重定位Agent_答辩稿_v7.pptx")
    pdf = Path("artifacts/report_render_v5/药物重定位Agent_课程设计报告_v5.pdf")
    pages = sorted(Path("artifacts/report_render_v5").glob("page-*.png"))
    slides = sorted(Path("artifacts/slide_build_v7").glob("slide-*.png"))
    if len(pages) != 13 or len(slides) != 12 or len(fitz.open(pdf)) != 13:
        raise RuntimeError("Rendered page/slide count differs from expected layout")
    with zipfile.ZipFile(report) as archive:
        if archive.testzip():
            raise RuntimeError("DOCX package corrupt")
    with zipfile.ZipFile(deck) as archive:
        if archive.testzip():
            raise RuntimeError("PPTX package corrupt")
        names = archive.namelist()
        charts = sum(name.startswith("ppt/charts/chart") and name.endswith(".xml")
                     for name in names)
        if charts != 4:
            raise RuntimeError("Expected four native editable charts")
    result = {"report_sha256": sha256_file(report), "deck_sha256": sha256_file(deck),
              "report_pages": len(pages), "deck_slides": len(slides),
              "native_editable_charts": charts,
              "visual_review": "All thirteen rendered report pages (Word) and all twelve rendered slides (PowerPoint) inspected after pre-registering five diseases (section 5.3 table, slide 12 limitations row); report pages 11-12 and slide 12 re-checked.",
              "page_png_sha256": {str(path): sha256_file(path) for path in pages},
              "slide_png_sha256": {str(path): sha256_file(path) for path in slides}}
    output = Path("artifacts/reports/current_deliverables_qa.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("qa_recorded", output=str(output), output_sha256=sha256_file(output),
               report_sha256=result["report_sha256"], deck_sha256=result["deck_sha256"],
               pages=len(pages), slides=len(slides), native_charts=charts)
    print(f"Verified report {len(pages)} pages and deck {len(slides)} slides; trace: {trace.path}")


if __name__ == "__main__":
    traced_run("current_deliverables_qa", _main)
