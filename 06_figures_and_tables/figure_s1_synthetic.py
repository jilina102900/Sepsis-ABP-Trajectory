"""Figure S1: synthetic recovery of tau under the two fitting paths.

A synthetic waveform with a known tau is inverted, and the estimate is compared
with the truth across a noise grid.  The point of the panel is that the free-P_inf
path recovers tau well inside the range the study relies on, and that the error
grows with noise rather than with tau itself.

The numbers in the panel title are read from the validation table produced by
``02_waveform_modeling/synthetic_validation.py``; nothing is typed in by hand.

Usage
-----
    python 06_figures_and_tables/figure_s1_synthetic.py
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
    path = config.out_path("synthetic_validation.csv")
    if not path.exists():
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_waveform_modeling"))
        import synthetic_validation as sv
        sv.main()
    df = pd.read_csv(path)
    df = df[(df["route"] == "freePinf") & np.isfinite(df["tau_est_s"])].copy()
    df["rel_error_pct"] = df["rel_error"] * 100.0
    # average across seeds for each (true tau, noise) cell
    agg = (df.groupby(["tau_true_s", "noise_mmhg"])
             .agg(tau_est_s=("tau_est_s", "mean"),
                  rel_error_pct=("rel_error_pct", "mean"))
             .reset_index())

    fixed = agg[agg["noise_mmhg"] == 1.0].sort_values("tau_true_s")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))

    ax = axes[0]
    ax.plot(fixed["tau_true_s"], fixed["tau_est_s"], "o-", color=PALETTE["NEG"],
            label="Estimated τ")
    ax.plot(fixed["tau_true_s"], fixed["tau_true_s"], "--",
            color=PALETTE["NEUTRAL"], label="Identity")
    ax.fill_between(fixed["tau_true_s"], fixed["tau_true_s"] * 0.95,
                    fixed["tau_true_s"] * 1.05, color=PALETTE["NEG"], alpha=0.12,
                    lw=0, label="±5 % band")
    ax.set_xlabel("True τ (s)")
    ax.set_ylabel("Estimated τ (s)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Recovery at fixed noise (1 mmHg)")

    ax = axes[1]
    for tau in sorted(agg["tau_true_s"].unique()):
        sub = agg[agg["tau_true_s"] == tau].sort_values("noise_mmhg")
        ax.plot(sub["noise_mmhg"], sub["rel_error_pct"], "o-", ms=3,
                label=f"τ = {tau:.2f} s")
    ax.axhline(5, color=PALETTE["POS"], ls="--", lw=1.0)
    ax.set_xlabel("Additive noise (mmHg)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Error grows with noise, not with τ")
    ax.legend(frameon=False, fontsize=7)

    if len(fixed):
        row = fixed.iloc[(fixed["tau_true_s"] - 0.2).abs().argmin()]
        title = (f"True τ = {row['tau_true_s']:.3f} s, estimated "
                 f"{row['tau_est_s']:.3f} s (error {row['rel_error_pct']:.1f} %, "
                 f"noise 1 mmHg)")
    else:
        title = "Synthetic recovery"
    fig.suptitle(title, fontsize=9)
    fig.subplots_adjust(wspace=0.32, top=0.84)
    save(fig, config.out_path("figure_s1_synthetic_validation.png"))
    print(f"written: {config.out_path('figure_s1_synthetic_validation.png')}")


if __name__ == "__main__":
    main()
