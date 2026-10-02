"""Shared figure style for every figure under docs/figures/ (paper-like).

One palette, one font and one size scale for all generators
(scripts/make_figures.py, scripts/make_architecture_figures.py). Colours match
the defense deck (``const C`` in scripts/build_defense_deck_v7.mjs). Meaning is
fixed across figures: tumour-up = RED, tumour-down = BLUE; B2 = RED, B3 = BLUE,
B4 = TEAL; published RECeSS models = GRAY. Figures carry no headline title:
the caption in the report/deck says what the figure shows; multi-panel figures
get bold panel letters (a, b).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

NAVY, INK, INK2, MUTED = "#112B3C", "#183042", "#4A5A64", "#8A979F"
GRID, LIGHT, GRAY = "#E3E7E9", "#D5DBDE", "#B9C2C7"
TEAL, RED, BLUE, GOLD = "#087E78", "#9E493D", "#2F6690", "#B9862E"
UP, DOWN = RED, BLUE
METHOD_COLORS = {"B2": RED, "B3": BLUE, "B4": TEAL}
PUBLISHED = GRAY
# Box fills for diagrams (LLM / plain code / deterministic tool / output-stop).
FILL = {"llm": "#E3F1EF", "code": "#E8EDF2", "tool": "#F4EFE6", "out": "#F2E7E4", "user": NAVY}
# Diverging map centred on 0.5: RED = not reversed, TEAL = strongly reversed.
REVERSAL_CMAP = LinearSegmentedColormap.from_list("reversal", [RED, "#F6F4F1", TEAL])
DPI = 300


def apply() -> None:
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    cjk = [n for n in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
                       "PingFang SC") if n in available]
    if not cjk:
        raise SystemExit("No CJK font found; install Microsoft YaHei/SimHei/Noto Sans CJK.")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": cjk + ["DejaVu Sans"],
        "axes.unicode_minus": False, "svg.fonttype": "path",
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 8, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5, "legend.frameon": False,
        "axes.edgecolor": INK2, "axes.linewidth": 0.8, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def panel(ax, letter: str, x: float = -0.02, y: float = 1.0) -> None:
    """Bold panel letter at the top-left corner, outside the axes."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK,
            ha="right", va="bottom")


def grid(ax, axis: str = "both") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
