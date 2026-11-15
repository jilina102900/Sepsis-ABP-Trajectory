"""Figure 2: empirical power curve and the detectable effect.

The curve answers one question: with the slope distribution this cohort actually
has, what effect would have been needed for the study to detect it 80 % of the
time?  Two vertical lines sit on the same axis -- the detection limit and the
effect that was observed -- so the distance between them is the reason the
comparison is underpowered.

Usage
-----
    python 06_figures_and_tables/figure2_power.py
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
    curve = pd.read_csv(config.require(config.out_path("mimic_power_curve.csv")))
    boot = config.out_path("mimic_mde_bootstrap.csv")
    mde_vals = pd.read_csv(boot)["mde_boot"].dropna() if boot.exists() else pd.Series(dtype=float)

    slopes = pd.read_csv(config.require(config.data_path("mimic_slope_441.csv")))
    v = slopes["tau_proxy_slope_pct_h"].dropna().to_numpy(float)
    ev = slopes.loc[slopes["tau_proxy_slope_pct_h"].notna(), "event_6h"].to_numpy(int) == 1
    observed = float(np.median(v[ev]) - np.median(v[~ev]))

    mde = float(curve.loc[curve["power"] >= config.MDE_TARGET_POWER, "shift_pct_h"].min())

    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.plot(curve["shift_pct_h"], curve["power"], color=PALETTE["NEG"], lw=1.4,
            label="Empirical power")
    ax.axhline(config.MDE_TARGET_POWER, color=PALETTE["NEUTRAL"], ls=":", lw=0.9)
    ax.text(12.0, config.MDE_TARGET_POWER + 0.015, "80 % power",
            color=PALETTE["NEUTRAL"], fontsize=8)

    ax.axvline(mde, color=PALETTE["POS"], ls="--", lw=1.0, label=f"MDE = {mde:.1f} %/h")
    ax.axvline(abs(observed), color=PALETTE["ACCENT"], ls="--", lw=1.0,
               label=f"Observed |effect| = {abs(observed):.2f} %/h")

    if len(mde_vals):
        lo, hi = np.percentile(mde_vals, [2.5, 97.5])
        ax.axvspan(lo, hi, color=PALETTE["POS"], alpha=0.10,
                   label=f"MDE 95 % CI [{lo:.1f}, {hi:.1f}]")

    ax.set_xlabel("Slope difference (%/h)")
    ax.set_ylabel("Power")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, min(float(curve["shift_pct_h"].max()), 20))
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7,
              handlelength=1.8, labelspacing=0.3, edgecolor="none")
    ax.set_title("Detectable versus observed slope difference", fontsize=9)
    save(fig, config.out_path("figure2_power_curve.png"))
    print(f"written: {config.out_path('figure2_power_curve.png')}")


if __name__ == "__main__":
    main()
