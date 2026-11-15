"""Figure 3: method-level minimal detectable effect, both cohorts.

The heatmap is read along two axes at once.  Rows are the aggregation level used
to produce a slope; columns are the observation horizon used to fit it.  Moving
right (longer horizon) lowers the limit for both cohorts, and moving from patient
to beat level raises it -- a beat-level slope is noisier, so more of the signal is
buried.

Only cells marked ``recomputed`` in the source table were re-estimated here; cells
marked ``reported`` are the manuscript's values for aggregation levels whose
per-unit arrays are not part of the released tables.

Usage
-----
    python 06_figures_and_tables/figure3_heatmap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_processing"))
import config  # noqa: E402
from style import PALETTE, apply_style, save  # noqa: E402


def main() -> None:
    apply_style()
    df = pd.read_csv(config.require(config.out_path("table_s2_mde_method_level.csv")))

    cohort_col, gran_col = df.columns[0], df.columns[1]
    win_cols = [c for c in df.columns if "min" in str(c)]
    horizons = [int(str(c).split()[0]) for c in win_cols]
    df = df.rename(columns={cohort_col: "cohort", gran_col: "granularity"})

    # rows top-to-bottom: finest aggregation first within each cohort
    order = ["Window-level", "Segment-level", "Track-level", "ICU-level", "Case-level"]
    df["granularity"] = pd.Categorical(df["granularity"], categories=order, ordered=True)
    df["cohort"] = pd.Categorical(df["cohort"], categories=["MIMIC", "VitalDB"],
                                 ordered=True)
    df = df.sort_values(["cohort", "granularity"])

    mat = df[win_cols].to_numpy(float)
    labels = [f"{c} · {g}" for c, g in zip(df["cohort"], df["granularity"])]

    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(win_cols)))
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Observation horizon (min)")
    ax.grid(False)

    vmax = np.nanmax(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}" if val < 10 else f"{val:.1f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if val < 0.65 * vmax else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("MDE (%/h)")
    ax.set_title("Minimal detectable slope difference by method", fontsize=9)
    save(fig, config.out_path("figure3_mde_heatmap.png"))
    print(f"written: {config.out_path('figure3_mde_heatmap.png')}")


if __name__ == "__main__":
    main()
