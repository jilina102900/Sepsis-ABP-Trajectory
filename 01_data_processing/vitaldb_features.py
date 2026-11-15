# -*- coding: utf-8 -*-
"""Case-level feature table for the VitalDB models M1 / M2 / M3.

Feature blocks (Methods):

    M1  clinical    pre-operative variables from ``cases.csv`` (14 columns)
    M2  M1 + routine ABP summaries measured in the 30-min baseline block
    M3  M2 + physical waveform features (tau level, tau drift, tau-proxy)

Only pre-operative clinical variables enter M1: intra-operative cumulative
quantities (blood loss, urine output, fluids, vasopressor dose) are recorded after
the prediction landmark and are therefore excluded by construction.

The baseline block is the first ``VITAL_LANDMARK_MIN`` minutes of the anaesthetic
record; the model is evaluated with 5-min estimation windows, so the median number
of eligible windows in this block is small (reported median: 4) and the median
number of accepted beats per case is large (reported median: 405).

Usage
-----
    python 01_data_processing/vitaldb_features.py \
        --vitaldb-root /path/to/vitaldb --out data/vitaldb_case_features.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abp_summary  # noqa: E402
import config  # noqa: E402

INTRAOP_POSTOP_VARS = [
    "intraop_ebl", "intraop_uo", "intraop_rbc", "intraop_ffp", "intraop_crystalloid",
    "intraop_colloid", "intraop_ppf", "intraop_mdz", "intraop_ftn", "intraop_rocu",
    "intraop_vecu", "intraop_eph", "intraop_phe", "intraop_epi", "intraop_ca",
]


def _relative_slope(t: np.ndarray, y: np.ndarray):
    """Relative slope in %/min plus R^2, normalised by the median level."""
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    level = np.median(y) if y.size else np.nan
    if t.size < 3 or not np.isfinite(level) or level <= 0 or np.ptp(t) == 0:
        return np.nan, np.nan
    slope, intercept = np.polyfit(t, y, 1)
    resid = y - (slope * t + intercept)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
    return 100.0 * slope / level, r2


def case_features(traj: pd.DataFrame, baseline_min: float = config.VITAL_LANDMARK_MIN) -> dict:
    """Physical + conventional features for one case."""
    ok = traj[traj["quality"] == "ok"].sort_values("time_min")
    base = ok[ok["time_min"] <= baseline_min]
    out = {"n_ok_windows": int(len(ok)), "n_eval_windows": int(len(base))}
    out["n_beats_accepted_median"] = float(np.nanmedian(ok["n_beats_accepted"])) \
        if "n_beats_accepted" in ok else np.nan

    out.update(abp_summary.conventional_abp(base))

    tau_slope, tau_r2 = _relative_slope(ok["time_min"].to_numpy(float),
                                        ok["tau"].to_numpy(float))
    r_slope, _ = _relative_slope(ok["time_min"].to_numpy(float),
                                 ok["R"].to_numpy(float))
    out.update({
        "tau_base_mean": float(np.nanmean(base["tau"])) if base["tau"].notna().any() else np.nan,
        "tau_last": float(ok["tau"].dropna().iloc[-1]) if ok["tau"].notna().any() else np.nan,
        "tau_slope_pct_h": tau_slope * 60.0 if np.isfinite(tau_slope) else np.nan,
        "tau_slope_rel_pct_h": tau_slope * 60.0 if np.isfinite(tau_slope) else np.nan,
        "tau_fit_r2": tau_r2,
        "R_base_mean": float(np.nanmean(base["R"])) if base["R"].notna().any() else np.nan,
        "R_slope_pct_h": r_slope * 60.0 if np.isfinite(r_slope) else np.nan,
    })
    return out


def build(traj: pd.DataFrame, cases: pd.DataFrame,
          baseline_min: float = config.VITAL_LANDMARK_MIN) -> pd.DataFrame:
    """Assemble the case-level table used by all three VitalDB models."""
    rows = []
    for caseid, grp in traj.groupby("caseid"):
        row = case_features(grp, baseline_min=baseline_min)
        row["caseid"] = caseid
        rows.append(row)
    feats = pd.DataFrame(rows)

    clin = cases[["caseid", "subjectid"] + [c for c in config.VITAL_CLIN_COLS if c in cases]]
    clin = clin.copy()
    clin["sex"] = (clin["sex"].astype(str).str.upper() == "M").astype(float)
    feats = feats.merge(clin, on="caseid", how="left")

    # case-level outcome: any sustained hypotension episode after the landmark
    ev = (traj.groupby("caseid")["hypo_event"].any().rename("event").reset_index())
    feats = feats.merge(ev, on="caseid", how="left")
    feats["event"] = feats["event"].fillna(False).astype(int)
    return feats.sort_values("caseid").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vitaldb-root", default=config.VITALDB_ROOT)
    ap.add_argument("--trajectories",
                    default=str(config.data_path("vitaldb_trajectories.csv.gz")))
    ap.add_argument("--out", default=str(config.data_path("vitaldb_case_features.csv")))
    args = ap.parse_args()
    if not args.vitaldb_root:
        raise SystemExit("set --vitaldb-root or APREPO_VITALDB_ROOT")

    traj = pd.read_csv(config.require(Path(args.trajectories)))
    cases = pd.read_csv(config.require(Path(args.vitaldb_root) / "cases.csv"))
    feats = build(traj, cases)
    feats.to_csv(args.out, index=False)

    print(f"cases: {len(feats)}  events: {int(feats['event'].sum())} "
          f"({feats['event'].mean()*100:.1f}%)")
    print(f"median eligible windows in the 30-min baseline: "
          f"{feats['n_eval_windows'].median():.0f}")
    print(f"median accepted beats per case: {feats['n_beats_accepted_median'].median():.0f}")
    print(f"tau median: {feats['tau_base_mean'].median():.3f} s")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
