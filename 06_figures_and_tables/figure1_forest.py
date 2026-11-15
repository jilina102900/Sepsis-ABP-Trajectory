"""Figure 1: forest plot of the incremental value of ABP features.

Each row is one analysis of the same question -- does adding arterial waveform
information improve prediction beyond what is already available? -- and each
point is a paired ΔAUC with a patient-level bootstrap interval.  The three rows
are deliberately different in sample and model family, and the figure is read as
a set: if all three intervals straddle zero, no version of the analysis finds
incremental value.

Inputs are produced by the ``03_prediction_models`` scripts.

Usage
-----
    python 06_figures_and_tables/figure1_forest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_processing"))
import config  # noqa: E402
from style import PALETTE, apply_style, save  # noqa: E402

SOURCES = [
    ("Primary LightGBM (n = 480)", "table2_delta_auc.csv"),
    ("Ridge logistic (n = 480)", "table3_ridge_delta.csv"),
    ("NEWS2 subset (n = 319)", "table3_news2_delta.csv"),
]


def main() -> None:
    apply_style()
    rows = []
    for label, fname in SOURCES:
        path = config.require(config.out_path(fname))
        d = pd.read_csv(path).iloc[0]
        rows.append({"label": label, "delta": float(d["delta_auc"]),
                     "lo": float(d["ci_lo"]), "hi": float(d["ci_hi"])})

    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    for i, r in enumerate(rows):
        colour = PALETTE["NEG"] if (r["lo"] < 0 < r["hi"]) else PALETTE["POS"]
        ax.plot([r["lo"], r["hi"]], [i, i], color=colour, lw=1.4, zorder=2)
        ax.plot(r["delta"], i, "o", color=colour, ms=5, zorder=3)
        ax.annotate(f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]",
                    xy=(r["hi"], i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=colour)

    ax.axvline(0, color=PALETTE["NEUTRAL"], lw=0.9, ls="--", zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("ΔAUC (EHR + ABP − EHR)")
    ax.set_xlim(min(r["lo"] for r in rows) - 0.02, max(r["hi"] for r in rows) + 0.10)
    ax.grid(axis="y", visible=False)
    ax.set_title("Incremental discrimination of arterial waveform features")
    save(fig, config.out_path("figure1_delta_auc_forest.png"))
    print(f"written: {config.out_path('figure1_delta_auc_forest.png')}")


if __name__ == "__main__":
    main()
