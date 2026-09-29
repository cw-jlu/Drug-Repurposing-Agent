"""Render every report page and record PDF integrity/layout diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

import fitz
from PIL import Image, ImageOps, ImageDraw

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


MANIFEST = Path("configs/report_latex_manifest.json")
PDF_DIR = Path("deliverables/latex")
BUILD = Path("artifacts/latex_build")
QA = Path("artifacts/latex_qa")
OVERFULL = re.compile(r"Overfull \\hbox \(([0-9.]+)pt too wide\)")


def _main(trace: TraceRecorder) -> None:
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["reports"]
    QA.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in entries:
        stem = entry["stem"]
        source, tex, pdf = (Path(entry["source"]), PDF_DIR / f"{stem}.tex",
                            PDF_DIR / f"{stem}.pdf")
        if not all(path.is_file() for path in (source, tex, pdf)):
            raise FileNotFoundError(f"Missing report source/output: {stem}")
        with fitz.open(pdf) as document:
            pages = len(document)
            rotations = [page.rotation for page in document]
            text_lengths = [len(page.get_text().strip()) for page in document]
        if pages < 1 or any(length < 30 for length in text_lengths):
            raise ValueError(f"Blank/unreadable page in {stem}: {text_lengths}")
        page_dir = QA / stem / trace.run_id
        page_dir.mkdir(parents=True, exist_ok=True)
        process = subprocess.run(["pdftoppm", "-png", "-scale-to", "850", str(pdf),
                                  str(page_dir / "page")], capture_output=True,
                                 text=True, check=False)
        if process.returncode:
            raise RuntimeError(f"Poppler render failed for {stem}: {process.stderr[-400:]}")
        images = sorted(page_dir.glob("page-*.png"))
        if len(images) != pages:
            raise ValueError(f"Rendered page count mismatch for {stem}: {len(images)} / {pages}")
        tiles = []
        for number, image_path in enumerate(images, start=1):
            with Image.open(image_path) as picture:
                picture.load()
                canvas = Image.new("RGB", (880, 900), "#e8ecef")
                fitted = ImageOps.contain(picture.convert("RGB"), (850, 850))
                canvas.paste(fitted, ((880 - fitted.width) // 2, 8))
                ImageDraw.Draw(canvas).text((14, 868), f"{stem} · page {number}/{pages}",
                                            fill="#243746")
                tiles.append(canvas)
        columns = 2
        rows = (len(tiles) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * 880, rows * 900), "#e8ecef")
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % columns) * 880, (i // columns) * 900))
        sheet_path = QA / f"{stem}_contact.png"
        sheet.save(sheet_path)
        log = BUILD / stem / f"{stem}.log"
        log_text = log.read_text(encoding="utf-8", errors="replace")
        warnings = [float(value) for value in OVERFULL.findall(log_text)]
        missing_glyphs = log_text.count("Missing character:")
        row = {"stem": stem, "source_sha256": sha256_file(source),
               "tex_sha256": sha256_file(tex), "pdf_sha256": sha256_file(pdf),
               "pages": pages, "rotations": rotations,
               "largest_overfull_pt": max(warnings, default=0),
               "overfull_count": len(warnings), "missing_glyphs": missing_glyphs,
               "contact_sheet": str(sheet_path),
               "contact_sheet_sha256": sha256_file(sheet_path)}
        results.append(row)
        trace.emit("report_qa_rendered", **row)
        print(f"{stem}: {pages} pages; overfull max {row['largest_overfull_pt']:.1f}pt; missing glyphs {missing_glyphs}")
    receipt = {"status": "all_pages_rendered_for_visual_review", "report_count": len(results),
               "total_pages": sum(row["pages"] for row in results),
               "reports": results, "trace": str(trace.path)}
    target = QA / f"qa_receipt_{trace.run_id}.json"
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("qa_receipt_saved", output=str(target), output_sha256=sha256_file(target))
    print(f"Receipt: {target}")


if __name__ == "__main__":
    traced_run("report_latex_qa", _main, QA / "traces")
