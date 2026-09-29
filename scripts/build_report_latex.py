"""Build reproducible XeLaTeX/PDF counterparts for current Markdown reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

import fitz

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


MANIFEST = Path("configs/report_latex_manifest.json")
OUTPUT = Path("deliverables/latex")
BUILD = Path("artifacts/latex_build")
PANDOC = Path("artifacts/tools/pandoc-3.11/pandoc.exe")
TABLE = re.compile(r"(\\begin\{longtable\}\[\]\{@\{\}([lcr]+)@\{\}\})(.*?)(\\end\{longtable\})",
                   re.DOTALL)
WIDE_COLUMN_FRACTIONS = {
    ("multi_agent_review", 8): (0.05, 0.17, 0.10, 0.10, 0.13, 0.10, 0.14, 0.11),
}


def _cells(line: str) -> list[str]:
    return [cell.strip().replace("\\|", "|") for cell in
            re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def _readable_wide_tables(source: str) -> tuple[str, int]:
    """Reflow dense GFM tables as labeled records without dropping any cell."""
    lines = source.splitlines()
    output: list[str] = []
    index = 0
    converted = 0
    while index < len(lines):
        if (index + 1 < len(lines) and lines[index].lstrip().startswith("|") and
                lines[index + 1].lstrip().startswith("|")):
            header = _cells(lines[index])
            rule = _cells(lines[index + 1])
            if (len(header) > 5 and len(header) == len(rule) and
                    all(re.fullmatch(r":?-+:?", cell) for cell in rule)):
                rows = []
                index += 2
                while index < len(lines) and lines[index].lstrip().startswith("|"):
                    row = _cells(lines[index])
                    if len(row) != len(header):
                        raise ValueError(f"Wide Markdown table row has {len(row)} cells, expected {len(header)}")
                    rows.append(row)
                    index += 1
                for row in rows:
                    output.append(f"**{header[0]} {row[0]} · {row[1]}**")
                    output.append("")
                    for label, value in zip(header[2:], row[2:]):
                        output.append(f"- **{label}：** {value}")
                    output.append("")
                converted += 1
                continue
        output.append(lines[index])
        index += 1
    return "\n".join(output) + "\n", converted


def _fit_tables_and_symbols(tex: Path) -> int:
    """Give Pandoc tables bounded, wrapping columns rather than clipped natural widths."""
    source = tex.read_text(encoding="utf-8")

    def break_inline_code(document: str) -> str:
        marker = "\\texttt{"
        pieces = []
        cursor = 0
        while True:
            start = document.find(marker, cursor)
            if start < 0:
                pieces.append(document[cursor:])
                break
            pieces.append(document[cursor:start])
            position = start + len(marker)
            depth = 1
            while position < len(document) and depth:
                if document[position] == "{" and document[position - 1] != "\\":
                    depth += 1
                elif document[position] == "}" and document[position - 1] != "\\":
                    depth -= 1
                position += 1
            if depth:
                raise ValueError("Unclosed Pandoc inline code command")
            content = document[start + len(marker):position - 1]
            content = content.replace("\\_", "\\_\\allowbreak{}")
            content = content.replace("/", "/\\allowbreak{}")
            content = content.replace(".", ".\\allowbreak{}")
            content = re.sub(r"[A-Za-z0-9]{24,}",
                             lambda match: "\\allowbreak{}".join(
                                 match.group()[i:i + 14]
                                 for i in range(0, len(match.group()), 14)), content)
            pieces.append(marker + content + "}")
            cursor = position
        return "".join(pieces)

    def replace(match: re.Match[str]) -> str:
        count = len(match.group(2))
        body = match.group(3).replace("\\_", "\\_\\allowbreak{}")
        if count >= 6:
            fractions = WIDE_COLUMN_FRACTIONS.get((tex.stem, count), (0.90 / count,) * count)
            if len(fractions) != count or abs(sum(fractions) - 0.90) > 0.001:
                raise ValueError("Invalid wide-table column allocation")
            columns = "".join(f">{{\\raggedright\\arraybackslash}}p{{{fraction:.4f}\\textheight}}"
                              for fraction in fractions)
            return ("\\begin{landscape}\\scriptsize\\sloppy\n"
                    "\\begin{longtable}[]{@{}" + columns + "@{}}" +
                    body + match.group(4) + "\n\\end{landscape}")
        fraction = 0.90 / count
        column = f">{{\\raggedright\\arraybackslash}}p{{{fraction:.4f}\\textwidth}}"
        return "\\begin{longtable}[]{@{}" + column * count + "@{}}" + body + match.group(4)

    edited, tables = TABLE.subn(replace, source)
    edited = break_inline_code(edited)
    edited = edited.replace("\\begin{verbatim}",
                            "\\begin{Verbatim}[breaklines=true,breakanywhere=true]")
    edited = edited.replace("\\end{verbatim}", "\\end{Verbatim}")
    edited = re.sub(
        r"(?m)^(\\DefineVerbatimEnvironment\{Highlighting\}\{Verbatim\}\{.*)\}$",
        r"\1,breaklines=true,breakanywhere=true}", edited)
    edited = edited.replace("\\begin{document}",
                            "\\usepackage{pdflscape,fvextra}\n"
                            "\\setlength{\\tabcolsep}{3pt}\n\\begin{document}", 1)
    tex.write_text(edited, encoding="utf-8", newline="\n")
    return tables


def _run(command: list[str], trace: TraceRecorder, stage: str) -> None:
    result = subprocess.run(command, text=True, capture_output=True, encoding="utf-8",
                            errors="replace", check=False)
    trace.emit(stage, command=[Path(command[0]).name, *command[1:]],
               exit_code=result.returncode,
               stdout_tail=result.stdout[-3000:], stderr_tail=result.stderr[-3000:])
    if result.returncode:
        raise RuntimeError(f"{stage} failed (exit {result.returncode}): {result.stderr[-800:] or result.stdout[-800:]}")


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="Optional manifest stems for a repair build")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    reports = manifest["reports"]
    stems = [entry["stem"] for entry in reports]
    if len(stems) != len(set(stems)) or not reports:
        raise ValueError("Report manifest has duplicate stems or is empty")
    if args.only:
        unknown = set(args.only) - set(stems)
        if unknown:
            raise ValueError(f"Unknown report stems: {sorted(unknown)}")
        reports = [entry for entry in reports if entry["stem"] in args.only]
    if not PANDOC.is_file() or not shutil.which("xelatex"):
        raise FileNotFoundError("Verified Pandoc 3.11 and XeLaTeX are required")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    trace.emit("manifest_loaded", manifest_sha256=sha256_file(MANIFEST),
               report_count=len(reports), pandoc_sha256=sha256_file(PANDOC))
    rows = []
    for entry in reports:
        source = Path(entry["source"])
        stem = entry["stem"]
        if not source.is_file():
            raise FileNotFoundError(source)
        tex = OUTPUT / f"{stem}.tex"
        pdf = OUTPUT / f"{stem}.pdf"
        temporary = BUILD / stem
        temporary.mkdir(parents=True, exist_ok=True)
        trace.emit("report_started", source=str(source), source_sha256=sha256_file(source),
                   tex=str(tex), pdf=str(pdf))
        normalized, wide_tables = _readable_wide_tables(source.read_text(encoding="utf-8"))
        markdown_input = source
        if wide_tables:
            markdown_input = temporary / "normalized.md"
            markdown_input.write_text(normalized, encoding="utf-8", newline="\n")
            trace.emit("wide_tables_reflowed", table_count=wide_tables,
                       normalized_sha256=sha256_file(markdown_input))
        _run([str(PANDOC), str(markdown_input), "--from=gfm", "--to=latex", "--standalone",
              "--resource-path=.;docs", "--variable=documentclass:ctexart",
              "--variable=geometry:margin=2cm", "--variable=fontsize:10pt",
              "--variable=mainfont:DejaVu Sans",
              "--variable=colorlinks:true", "--output", str(tex)], trace, "pandoc_finished")
        tables = _fit_tables_and_symbols(tex)
        trace.emit("latex_layout_normalized", tables=tables,
                   tex_sha256=sha256_file(tex))
        _run(["xelatex", "-halt-on-error", "-interaction=nonstopmode",
              f"-output-directory={temporary.as_posix()}", tex.as_posix()],
             trace, "xelatex_finished")
        built_pdf = temporary / f"{stem}.pdf"
        if not built_pdf.exists():
            raise FileNotFoundError(built_pdf)
        shutil.copy2(built_pdf, pdf)
        with fitz.open(pdf) as document:
            page_count = len(document)
            empty_pages = [i + 1 for i, page in enumerate(document) if not page.get_text().strip()]
        if page_count < 1 or empty_pages:
            raise ValueError(f"Blank or image-only pages in {pdf}: {empty_pages}")
        row = {"source": str(source), "source_sha256": sha256_file(source),
               "tex": str(tex), "tex_sha256": sha256_file(tex),
               "pdf": str(pdf), "pdf_sha256": sha256_file(pdf), "pages": page_count}
        rows.append(row)
        trace.emit("report_completed", **row)
        print(f"{stem}: {page_count} pages")
    receipt = {"status": "built_and_text_checked_not_visual_qa",
               "manifest_sha256": sha256_file(MANIFEST), "reports": rows,
               "trace": str(trace.path)}
    path = BUILD / f"build_receipt_{trace.run_id}.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("build_receipt_saved", output=str(path), output_sha256=sha256_file(path))
    print(f"Receipt: {path}")


if __name__ == "__main__":
    traced_run("report_latex_build", _main, BUILD / "traces")
