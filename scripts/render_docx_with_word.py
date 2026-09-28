"""Windows fallback renderer for DOCX visual QA when bundled LibreOffice is absent."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz
import win32com.client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / f"{source.stem}.pdf"

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

    rendered = fitz.open(pdf)
    for index, page in enumerate(rendered, start=1):
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        pixmap.save(output / f"page-{index}.png")
    print(f"Rendered {len(rendered)} pages to {output}")


if __name__ == "__main__":
    main()
