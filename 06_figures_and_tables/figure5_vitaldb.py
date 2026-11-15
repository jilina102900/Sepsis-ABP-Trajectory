"""Figure 5: VitalDB trajectory quality.

Left panel
    Density of the case-level tau drift, split by whether intra-operative
    hypotension occurred.  The two densities overlap almost completely, which is
    the visual form of the null result.

Right panel
    Tau level against the fit quality of the tau trajectory.  A case with no
    computable trajectory fit is treated as failing the threshold, because the
    point of the panel is how often the trajectory is *informative*: a case that
    yields no R² at all is exactly a case whose drift estimate cannot be trusted.
    Every percentage is computed here from the shipped table and printed next to
    the one drawn in the panel, so the figure and the text cannot drift apart.

Three denominators are reported:

* all cases in the cohort                         -> the headline percentage
* cases with a computable fit only                -> the conditional percentage
* cases whose fit is computable *and* R² < 0.1     -> the numerator

Usage
-----
    python 06_figures_and_tables/figure5_vitaldb.py
    python 06_figures_and_tables/figure5_vitaldb.py --features path/to/table.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_processing"))
import config  # noqa: E402
from style import PALETTE, apply_style, save  # noqa: E402

R2_THRESHOLD = 0.1


def kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE with Silverman's bandwidth (no seaborn dependency)."""
    v = values[np.isfinite(values)]
    if v.size < 3:
        return np.zeros_like(grid)
    bw = 1.06 * v.std(ddof=1) * v.size ** (-1 / 5)
    bw = bw if bw > 0 else 1e-6
    kernel = np.exp(-0.5 * ((grid[:, None] - v[None, :]) / bw) ** 2)
    return kernel.sum(axis=1) / (v.size * bw * np.sqrt(2 * np.pi))


def main() -> None:
    apply_style()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features",
                    default=str(config.data_path("vitaldb_prediction_set_1043.csv")))
    args = ap.parse_args()

    df = pd.read_csv(config.require(Path(args.features)))
    if "event" not in df.columns:
        df["event"] = df["first_ioh_in_pred"]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    ax = axes[0]
    v = df["tau_slope"].to_numpy(float)
    e = df["event"].to_numpy(int) == 1
    lo, hi = np.nanpercentile(v, [0.5, 99.5])
    grid = np.linspace(lo, hi, 200)
    ax.plot(grid, kde(v[e], grid), color=PALETTE["POS"], label=f"IOH (n = {int(e.sum())})")
    ax.plot(grid, kde(v[~e], grid), color=PALETTE["NEG"],
            label=f"No IOH (n = {int((~e).sum())})")
    ax.set_xlabel("τ drift (%/min)")
    ax.set_ylabel("Density")
    ax.set_title("Case-level τ drift by IOH status")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.78))

    ax = axes[1]
    x = df["tau_level"].to_numpy(float)
    r2 = df["tau_r2_5min"].to_numpy(float)
    fit = np.isfinite(x) & np.isfinite(r2)
    ax.scatter(x[fit], r2[fit], s=2.5, alpha=0.35, color=PALETTE["NEUTRAL"], lw=0)
    ax.axhline(R2_THRESHOLD, color=PALETTE["POS"], ls="--", lw=1.0)

    n_all = len(df)
    low = int(np.sum(r2 < R2_THRESHOLD))
    no_fit = int(np.isnan(r2).sum())
    share_all = (low + no_fit) / n_all * 100
    share_fit = low / int(fit.sum()) * 100

    ax.set_xlabel("τ level (s)")
    ax.set_ylabel("Trajectory-fit R²")
    ax.set_title(f"{share_all:.1f}% of cases: R² < {R2_THRESHOLD}")
    ax.annotate(f"{low} with R² < {R2_THRESHOLD}  +  {no_fit} without a\n"
                f"computable fit  =  {low + no_fit}/{n_all}",
                xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=7.5, color=PALETTE["NEUTRAL"], va="bottom")

    fig.subplots_adjust(wspace=0.34)
    save(fig, config.out_path("figure5_vitaldb_trajectory.png"))

    print(f"cohort: {n_all} cases")
    print(f"R² < {R2_THRESHOLD} with a computable fit: {low} cases")
    print(f"no computable trajectory fit: {no_fit} cases")
    print(f"headline share over the whole cohort: {low + no_fit}/{n_all} "
          f"= {share_all:.1f}%")
    print(f"conditional share among the {int(fit.sum())} fitted trajectories: "
          f"{share_fit:.1f}%")
    print(f"written: {config.out_path('figure5_vitaldb_trajectory.png')}")


if __name__ == "__main__":
    main()
