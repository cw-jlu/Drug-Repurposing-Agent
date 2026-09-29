"""Windows fallback renderer for DOCX visual QA when bundled LibreOffice is absent."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz
import win32com.client

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


def _render(args: argparse.Namespace, trace: TraceRecorder) -> None:
    source = args.input.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / f"{source.stem}.pdf"
    trace.emit("input_checked", source=str(source), source_sha256=sha256_file(source),
               output_dir=str(output))

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    document = None
    try:
        document = word.Documents.Open(str(source), ReadOnly=True)
        document.ExportAsFixedFormat(str(pdf), 17)
    finally:
        if document is not None:
            document.Close(False)
        word.Quit()
    trace.emit("word_exported", pdf=str(pdf), pdf_sha256=sha256_file(pdf))

    with fitz.open(pdf) as rendered:
        expected = set()
        for index, page in enumerate(rendered, start=1):
            destination = output / f"page-{index}.png"
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            pixmap.save(destination)
            expected.add(destination.name)
            trace.emit("page_rendered", page=index, output=str(destination),
                       output_sha256=sha256_file(destination))
        count = len(rendered)
    for stale in output.glob("page-*.png"):
        if stale.name not in expected and stale.stem[5:].isdigit():
            stale.unlink()
            trace.emit("stale_generated_page_removed", output=str(stale))
    print(f"Rendered {count} pages to {output}; trace: {trace.path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    traced_run("course_report_word_render", lambda trace: _render(args, trace),
               args.output_dir / "traces")


if __name__ == "__main__":
    main()
