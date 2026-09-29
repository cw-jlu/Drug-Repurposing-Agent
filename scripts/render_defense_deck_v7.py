"""Render all current defense slides in PowerPoint with a chained QA trace."""

from __future__ import annotations

from pathlib import Path
import win32com.client
from PIL import Image

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


DECK = Path("deliverables/药物重定位Agent_答辩稿_v7.pptx")
OUTPUT = Path("artifacts/slide_build_v7")


def _main(trace: TraceRecorder) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = DECK.resolve()
    trace.emit("deck_opening", source=str(source), source_sha256=sha256_file(DECK))
    app = win32com.client.DispatchEx("PowerPoint.Application")
    presentation = None
    try:
        presentation = app.Presentations.Open(str(source), ReadOnly=True, WithWindow=False)
        count = presentation.Slides.Count
        if count != 11:
            raise ValueError(f"Expected 11 slides, got {count}")
        for slide in presentation.Slides:
            target = (OUTPUT / f"slide-{slide.SlideIndex:02d}.png").resolve()
            slide.Export(str(target), "PNG", 1600, 900)
            trace.emit("slide_rendered", number=slide.SlideIndex, output=str(target),
                       output_sha256=sha256_file(target))
    finally:
        if presentation is not None:
            presentation.Close()
        app.Quit()
    sheet = Image.new("RGB", (1600, 2700), "white")
    for index in range(11):
        with Image.open(OUTPUT / f"slide-{index + 1:02d}.png") as slide_image:
            thumb = slide_image.convert("RGB").resize((800, 450))
            sheet.paste(thumb, ((index % 2) * 800, (index // 2) * 450))
    sheet_path = OUTPUT / "sheet.png"
    sheet.save(sheet_path)
    trace.emit("contact_sheet_saved", output=str(sheet_path),
               output_sha256=sha256_file(sheet_path))
    trace.emit("render_completed", slides=11)
    print(f"Rendered 11 slides; trace: {trace.path}")


if __name__ == "__main__":
    traced_run("defense_deck_v7_render", _main, OUTPUT / "traces")
