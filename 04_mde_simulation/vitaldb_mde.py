"""VitalDB detection limit and the short/long observation split.

Two questions are answered here.

1. **How large is the VitalDB detection limit?**  The empirical procedure used for
   MIMIC is applied to the *case-level* tau-slope distribution of the 1,043-case
   landmark cohort.  A detection limit quoted without its aggregation level and its
   unit is not interpretable, so the number of units, the median slope, its standard
   deviation, and both unit forms are reported alongside.

2. **Does observation length change it?**  Cases are split at the median number of
   evaluable pre-landmark minutes into a short and a long group.  A short record
   gives fewer points per slope fit, so the short group is expected to have the
   wider slope distribution and the coarser detection limit.

Unit provenance
---------------
``tau_slope`` is the Theil-Sen trend of the beat-level decay constant against time,
divided by the median decay constant:

    pt      = peak_times / 60.0            # seconds -> minutes
    s_beat  = theilslopes(taus, pt)[0] / tau_median * 100

so the column is a *relative* change **per minute** (``%/min``).  The manuscript
reports the VitalDB detection limit in ``%/h``, and the conversion is a factor of 60
inside the unit, not inside the simulation: the simulation is run on the column as
measured, and the detected shift is multiplied by ``--per-hour-factor`` (default 60)
for reporting.  A median |slope| of 3.20 %/min at a median tau of 0.249 s is a drift
of 8.0 ms/min (0.48 s/h) on a record whose length is a few minutes, which is the
range these traces show.

Usage
-----
    python 04_mde_simulation/vitaldb_mde.py
    python 04_mde_simulation/vitaldb_mde.py --n-sim 5000 --n-boot 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_processing"))
import config  # noqa: E402
from empirical_mde import empirical_mde  # noqa: E402


def unit_audit(df: pd.DataFrame, column: str, tau_column: str | None,
               per_hour: float) -> None:
    """Print the implied absolute drift so the slope unit can be verified."""
    if not tau_column or tau_column not in df.columns:
        return
    slope = df[column].abs().median()
    tau = df[tau_column].median()
    abs_pm = slope * tau / 100
    print(f"unit check: median |slope| = {slope:.3f} %/min, "
          f"median tau = {tau:.3f} s")
    print(f"            implied drift = {abs_pm * 1000:.1f} ms/min "
          f"= {abs_pm * per_hour * 1000:.0f} ms/h "
          f"= {abs_pm * per_hour:.3f} s/h if sustained for an hour")
    print(f"            reported unit: %/h = %/min x {per_hour:g}")


def run_block(label: str, df: pd.DataFrame, column: str, n_sim: int, n_boot: int,
              n_sim_boot: int, shifts: np.ndarray, per_hour: float) -> dict:
    v = df[column].to_numpy(float)
    m = np.isfinite(v)
    values, positive = v[m], df["event"].to_numpy(int)[m] == 1
    if values.size < 50 or positive.sum() < 10:
        return {"block": label, "n_units": int(values.size),
                "median_pct_min": np.nan, "sd_pct_min": np.nan,
                "mde_pct_min": np.nan, "mde_pct_h": np.nan,
                "mde_lo_pct_h": np.nan, "mde_hi_pct_h": np.nan,
                "note": "too few units to estimate"}
    res = empirical_mde(values, positive, n_boot=n_boot, n_sim=n_sim,
                        n_sim_boot=n_sim_boot, shifts=shifts)
    # Power at a 1 %/min shift, read off the same grid; this is the shift the
    # manuscript quotes when it contrasts the short and the long group.
    power_1 = float(np.interp(1.0, shifts, res["power"]))
    return {"block": label, "n_units": int(values.size),
            "median_pct_min": float(np.median(values)),
            "sd_pct_min": float(values.std(ddof=1)),
            "mde_pct_min": res["mde"],
            "mde_pct_h": res["mde"] * per_hour,
            "mde_lo_pct_h": res["mde_lo"] * per_hour,
            "mde_hi_pct_h": res["mde_hi"] * per_hour,
            "type_one": res["type_one"], "power_at_1pct_min": power_1,
            "note": f"{res['mde_boot'].size} bootstrap replicates"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features",
                    default=str(config.data_path("vitaldb_prediction_set_1043.csv")),
                    help="one row per case; needs the slope column and the outcome")
    ap.add_argument("--column", default="tau_slope",
                    help="case-level relative tau slope, in %%/min")
    ap.add_argument("--tau-column", default="tau_median")
    ap.add_argument("--split-column", default="pre_landmark_valid_win",
                    help="observation length used for the short/long split; falls "
                         "back to n_windows_5min when absent")
    ap.add_argument("--per-hour-factor", type=float, default=60.0,
                    help="minutes per hour; converts the reported MDE from %%/min "
                         "to %%/h. This is a unit conversion, not a model setting")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiplicative rescaling of --column, for a column stored "
                         "in another unit; 1.0 leaves the column as measured")
    ap.add_argument("--n-sim", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--n-sim-boot", type=int, default=100)
    ap.add_argument("--grid-step", type=float, default=0.05)
    ap.add_argument("--grid-max", type=float, default=1.5,
                    help="largest imposed shift on the grid, in the column's unit")
    args = ap.parse_args()

    df = pd.read_csv(config.require(Path(args.features)))
    if "event" not in df.columns:
        for alias in ("first_ioh_in_pred", "label", "y", "outcome"):
            if alias in df.columns:
                df["event"] = df[alias]
                break
    df = df.dropna(subset=["event"]).copy()
    df["event"] = df["event"].astype(int)
    if args.scale != 1.0:
        df[args.column] = df[args.column].to_numpy(float) * float(args.scale)

    print(f"units: {len(df)}  events: {int(df['event'].sum())}  column: {args.column}"
          + (f"  scale: {args.scale:g}" if args.scale != 1.0 else ""))
    unit_audit(df, args.column, args.tau_column, args.per_hour_factor)

    shifts = np.arange(0.0, args.grid_max + 1e-9, args.grid_step)
    blocks = [run_block("full", df, args.column, args.n_sim, args.n_boot,
                        args.n_sim_boot, shifts, args.per_hour_factor)]

    split_col = args.split_column if args.split_column in df.columns else (
        "n_windows_5min" if "n_windows_5min" in df.columns else None)
    if split_col:
        med = float(df[split_col].median())
        short = df[df[split_col] <= med]
        long = df[df[split_col] > med]
        blocks.append(run_block(f"short (<={med:.0f})", short, args.column,
                                args.n_sim, args.n_boot, args.n_sim_boot, shifts,
                                args.per_hour_factor))
        blocks.append(run_block(f"long (>{med:.0f})", long, args.column,
                                args.n_sim, args.n_boot, args.n_sim_boot, shifts,
                                args.per_hour_factor))
        print(f"split on {split_col} at median {med:.1f}")

    table = pd.DataFrame(blocks)
    table.to_csv(config.out_path("vitaldb_mde.csv"), index=False)
    print("\n=== VitalDB detection limit (case-level) ===")
    print(table.round(3).to_string(index=False))
    full = table.iloc[0]
    print(f"\nreported: {full['mde_pct_h']:.1f} %/h "
          f"[{full['mde_lo_pct_h']:.1f}, {full['mde_hi_pct_h']:.1f}] "
          f"= {full['mde_pct_min']:.2f} %/min x {args.per_hour_factor:g}")
    print(f"written: {config.out_path('vitaldb_mde.csv')}")


if __name__ == "__main__":
    main()
