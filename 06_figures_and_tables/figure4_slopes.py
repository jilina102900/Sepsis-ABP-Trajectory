"""Figure 4: distribution of the tau-proxy slope in MIMIC-III.

Two panels share one story.  The left panel shows the group comparison that the
paper reports -- medians that differ by about 1.9 %/h with heavily overlapping
distributions.  The right panel explains why that difference is hard to detect:
the distribution has a long tail, so the spread that matters for a rank test is
one to two orders of magnitude larger than the median difference.

Slopes are clipped at +/-100 %/h for display only; the statistics reported in the
text are computed on the unclipped values.

Usage
-----
    python 06_figures_and_tables/figure4_slopes.py
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

CLIP = 100.0


def main() -> None:
    apply_style()
    df = pd.read_csv(config.require(config.data_path("mimic_slope_441.csv")))
    v = df["tau_proxy_slope_pct_h"].to_numpy(float)
    e = df["event_6h"].to_numpy(int) == 1
    keep = np.isfinite(v)
    v, e = v[keep], e[keep]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))

    ax = axes[0]
    groups = [np.clip(v[e], -CLIP, CLIP), np.clip(v[~e], -CLIP, CLIP)]
    bp = ax.boxplot(groups, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops={"color": "black", "lw": 1.2})
    for patch, colour in zip(bp["boxes"], [PALETTE["POS"], PALETTE["NEG"]]):
        patch.set_facecolor(colour)
        patch.set_alpha(0.65)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Sepsis\n(n = {e.sum()})", f"No sepsis\n(n = {(~e).sum()})"])
    ax.set_ylabel("τ-proxy slope (%/h)")
    ax.axhline(0, color=PALETTE["NEUTRAL"], lw=0.8, ls=":")
    diff = np.median(v[e]) - np.median(v[~e])
    ax.set_title(f"Median difference = {diff:+.2f} %/h")

    ax = axes[1]
    bins = np.linspace(-CLIP, CLIP, 60)
    ax.hist(v, bins=bins, color=PALETTE["NEUTRAL"], alpha=0.55, label="All stays")
    ax.hist(v[e], bins=bins, color=PALETTE["POS"], alpha=0.55, label="Sepsis")
    ax.set_yscale("log")
    ax.set_xlabel("τ-proxy slope (%/h)")
    ax.set_ylabel("Stays per bin (log scale)")
    sd = v.std(ddof=1)
    ax.set_title(f"SD = {sd:.0f} %/h, range clipped to ±{CLIP:.0f}")
    ax.legend(frameon=False, loc="upper right")

    fig.subplots_adjust(wspace=0.32)
    save(fig, config.out_path("figure4_slope_distribution.png"))
    print(f"written: {config.out_path('figure4_slope_distribution.png')}")


if __name__ == "__main__":
    main()
