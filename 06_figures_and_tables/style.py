"""Shared plotting style: one place that fixes fonts, sizes and colour roles.

Every figure script imports :func:`apply_style` and :data:`PALETTE`, so a
journal's figure requirements can be changed once and propagate to all panels.
Colour roles are semantic, not decorative:

    NEG   augmentation does not add value (confidence interval crosses zero)
    POS   a genuine improvement
    NEUTRAL  reference / descriptive elements
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

PALETTE = {
    "NEG": "#4C72B0",
    "POS": "#C44E52",
    "NEUTRAL": "#8C8C8C",
    "GRID": "#DDDDDD",
    "ACCENT": "#55A868",
}

DPI = 300
FONT_SIZE = 9


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 1,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": PALETTE["GRID"],
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    })


def save(fig, path) -> str:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)
