# -*- coding: utf-8 -*-
"""Aggregate window-level trajectories into patient-level drift statistics.

The window-level trajectory produced by the extraction scripts has one row per
estimation window.  This module turns it into the quantities the manuscript
actually reports:

* window-level relative slope (``%/min``) of tau, and of the tau-proxy ``R``;
* segment-level slope (median over the windows of one continuous track);
* patient-level slope (median over all eligible windows inside the 0-4 h
  landmark window) -- the number entered into every between-group comparison;
* ``last-observed`` value (the final eligible window before the landmark);
* trajectory-fit ``R^2`` (quality of the linear fit behind the slope).

Relative slope definition
-------------------------
``100 * (d y / d t) / median(y)`` in ``%/min``, i.e. the slope normalised by the
patient's own median level.  This is the quantity whose heavy tails motivate the
non-parametric detection-limit analysis (Results, "long-tailed slope
distribution").

Usage
-----
    python 02_waveform_modeling/slope_aggregation.py \
        --traj-dir outputs/mimic_trajectories \
        --out data/mimic_patient_slopes.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_processing"))
import config  # noqa: E402


# ---------------------------------------------------------------------------
# Core estimators
# ---------------------------------------------------------------------------
def _ols(t: np.ndarray, y: np.ndarray):
    """Ordinary least squares slope, intercept and R^2."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size < 3 or np.ptp(t) == 0:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(t, y, 1)
    resid = y - (slope * t + intercept)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), r2


def _theil_sen(t: np.ndarray, y: np.ndarray):
    """Median of pairwise slopes; robust to heavy-tailed outliers."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    n = t.size
    if n < 3:
        return np.nan
    if n > 400:                       # keep the pairwise cost bounded
        idx = np.linspace(0, n - 1, 400).round().astype(int)
        t, y = t[idx], y[idx]
        n = t.size
    i, j = np.triu_indices(n, k=1)
    dt = t[j] - t[i]
    keep = np.abs(dt) > 1e-9
    return float(np.median((y[j][keep] - y[i][keep]) / dt[keep]))


def relative_slope(t: Sequence[float], y: Sequence[float], unit: str = "pct_min",
                   method: str = "ols"):
    """Relative slope in ``%/min`` (or ``%/h`` when ``unit='pct_h'``).

    Returns ``(slope, r2)``.
    """
    t_arr, y_arr = np.asarray(t, float), np.asarray(y, float)
    level = np.nanmedian(y_arr)
    if not np.isfinite(level) or level <= 0:
        return np.nan, np.nan
    if method == "theil_sen":
        slope = _theil_sen(t_arr, y_arr)
        r2 = np.nan
    else:
        slope, _, r2 = _ols(t_arr, y_arr)
    if not np.isfinite(slope):
        return np.nan, r2
    rel = 100.0 * slope / level
    return (rel * 60.0 if unit == "pct_h" else rel), r2


# ---------------------------------------------------------------------------
# Windowing helpers
# ---------------------------------------------------------------------------
def window_slopes(traj: pd.DataFrame, value_col: str,
                  window_min: int, step_min: int,
                  horizon_min: float = 240.0,
                  min_points: int = 5,
                  min_fraction: float = 0.6) -> pd.DataFrame:
    """Sliding-window relative slopes on one trajectory.

    A window is kept only if it holds at least ``min_points`` quality-passing
    samples covering at least ``min_fraction`` of its nominal length -- shorter
    coverage makes the slope numerically unstable and inflates the tail.
    """
    ok = traj[traj["quality"] == "ok"].sort_values("time_min")
    ok = ok[ok["time_min"] <= horizon_min]
    rows = []
    for start in np.arange(0.0, max(horizon_min - window_min, 0) + 1e-9, step_min):
        sel = ok[(ok["time_min"] >= start) & (ok["time_min"] < start + window_min)]
        if len(sel) < min_points:
            continue
        span = float(sel["time_min"].max() - sel["time_min"].min())
        if span < min_fraction * window_min:
            continue
        slope, r2 = relative_slope(sel["time_min"], sel[value_col], method="ols")
        rows.append({"window_start": float(start), "slope_pct_min": slope,
                     "r2": r2, "n_points": len(sel)})
    return pd.DataFrame(rows)


def patient_summary(traj: pd.DataFrame, horizon_min: float = 240.0,
                    window_min: int = 15, step_min: int = 5,
                    min_windows: int = 1) -> dict:
    """Patient-level drift summary used as the unit of analysis."""
    ok = traj[(traj["quality"] == "ok") & (traj["time_min"] <= horizon_min)]
    out = {"n_ok_windows": int(len(ok))}
    if ok.empty:
        return out

    tau_slope, tau_r2 = relative_slope(ok["time_min"], ok["tau"])
    r_slope, r_r2 = relative_slope(ok["time_min"], ok["R"])
    out.update({
        "tau_slope_pct_h": tau_slope * 60.0 if np.isfinite(tau_slope) else np.nan,
        "tau_proxy_slope_pct_h": r_slope * 60.0 if np.isfinite(r_slope) else np.nan,
        "tau_fit_r2": tau_r2,
        "tau_median_s": float(np.nanmedian(ok["tau"])),
        "tau_last_s": float(ok["tau"].dropna().iloc[-1]) if ok["tau"].notna().any() else np.nan,
        "r_median": float(np.nanmedian(ok["R"])),
        "r_last": float(ok["R"].dropna().iloc[-1]) if ok["R"].notna().any() else np.nan,
        "MAP_mean": float(np.nanmean(ok["MAP"])),
        "MAP_min": float(np.nanmin(ok["MAP"])),
        "HR_mean": float(np.nanmean(ok["HR"])),
        "PP_mean": float(np.nanmean(ok["PP"])),
    })

    win = window_slopes(traj, "tau", window_min, step_min, horizon_min)
    qualified = win.dropna(subset=["slope_pct_min"])
    out["n_windows"] = int(len(qualified))
    if len(qualified) >= min_windows:
        out["tau_window_slope_median_pct_h"] = float(qualified["slope_pct_min"].median() * 60.0)
    else:
        out["tau_window_slope_median_pct_h"] = np.nan
    return out


def aggregate_directory(traj_dir: Path, pattern: str = "*.csv",
                        id_cols: Iterable[str] = ("subject_id", "icustay_id", "caseid"),
                        horizon_min: float = 240.0) -> pd.DataFrame:
    """Run :func:`patient_summary` over a folder of per-track trajectories."""
    rows = []
    for path in sorted(Path(traj_dir).glob(pattern)):
        if path.name.startswith("failures"):
            continue
        traj = pd.read_csv(path)
        if traj.empty or "quality" not in traj.columns:
            continue
        row = patient_summary(traj, horizon_min=horizon_min)
        for col in id_cols:
            if col in traj.columns:
                row[col] = traj[col].iloc[0]
        row["source_file"] = path.name
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj-dir", required=True, help="folder of per-track trajectory CSVs")
    ap.add_argument("--out", default=str(config.out_path("patient_slopes.csv")))
    ap.add_argument("--horizon-min", type=float, default=240.0)
    args = ap.parse_args()

    df = aggregate_directory(Path(args.traj_dir), horizon_min=args.horizon_min)
    df.to_csv(args.out, index=False)
    print(f"patients: {len(df)}")
    if "tau_proxy_slope_pct_h" in df.columns:
        s = df["tau_proxy_slope_pct_h"].dropna()
        print(f"tau-proxy slope %/h: median={s.median():.2f} "
              f"IQR=[{s.quantile(.25):.2f}, {s.quantile(.75):.2f}] "
              f"SD={s.std():.2f}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
