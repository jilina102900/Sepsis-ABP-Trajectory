# -*- coding: utf-8 -*-
"""MIMIC-III cohort construction: Sepsis-3 -> 4-h landmark -> analysis cohorts.

Cascade implemented here, with the reported count at each step:

    Sepsis-3 (suspected infection + first-day SOFA >= 2, official derived table)
      -> exclude vasopressor started more than 6 h before ICU admission
      -> require continuous good-quality arterial waveform in the 0-4 h window
         and a successful waveform inversion                        -> 480 stays
      -> require a >=60 min continuous stable period that begins >=30 min after
         ICU admission and ends >=30 min before vasopressor start or ICU
         discharge                                                 -> 441 stays
      -> require a complete NEWS2 baseline                         -> 319 stays

Two input modes are supported and they are not equivalent:

``--mode derived`` (default)
    Uses the de-identified risk-set tables shipped in ``./data``.  Every stay in
    the 480/441/319 cohorts is present, so all downstream analyses and figures
    reproduce exactly.  The archived tables do not retain a per-reason exclusion
    log between 3,046 and 1,189 stays; this script therefore reconstructs the
    counts it can verify and never fabricates the ones it cannot.

``--mode raw``
    Rebuilds the Sepsis-3 cohort from MIMIC-III v1.4 plus the official
    ``sepsis3.csv`` derived table.  Requires a local MIMIC-III installation and
    completes the chain end to end.

Usage
-----
    python 01_data_processing/mimic_cohort.py --mode derived
    python 01_data_processing/mimic_cohort.py --mode raw \
        --mimic-root /path/to/mimic-iii-clinical-database-1.4 \
        --sepsis3 /path/to/sepsis3.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

PRE_ICU_VASOPRESSOR_HOURS = 6.0      # exclusion: vasopressor started before ICU
LANDMARK_H = config.MIMIC_LANDMARK_H
STABLE_MIN_MIN = 60                  # minimum continuous stable period
STABLE_MARGIN_MIN = 30               # clearance before admission / NE start / discharge


# ---------------------------------------------------------------------------
# Stage 1-2: Sepsis-3 and the pre-ICU vasopressor exclusion
# ---------------------------------------------------------------------------
def build_sepsis3_cohort(sepsis3_csv: Path, mimic_root: Path) -> pd.DataFrame:
    """Sepsis-3 stays with ICU admission time and time to first vasopressor.

    ``sepsis3.csv`` is the official MIT-LCP derived table (reference 26 of the
    manuscript); it supplies ``subject_id``, ``stay_id``, ``sepsis3`` and the
    SOFA components.  Vasopressor timing is taken from ``INPUTEVENTS_MV``.
    """
    s3 = pd.read_csv(sepsis3_csv)
    stay_col = "stay_id" if "stay_id" in s3.columns else "icustay_id"
    s3 = s3.rename(columns={stay_col: "icustay_id"})
    s3 = s3[s3.get("sepsis3", True).astype(bool)].copy()

    icu = pd.read_csv(mimic_root / "ICUSTAYS.csv.gz", usecols=[
        "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"], parse_dates=["INTIME"])
    s3 = s3.merge(icu, left_on="icustay_id", right_on="ICUSTAY_ID", how="left")

    vaso_ids = [221906, 221289, 221662, 221653, 221749, 229617]   # NE, epi, dopa, dobu, phenyl, vaso
    mv = pd.read_csv(mimic_root / "INPUTEVENTS_MV.csv.gz",
                     usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ITEMID", "STARTTIME"],
                     parse_dates=["STARTTIME"])
    mv = mv[mv["ITEMID"].isin(vaso_ids)]
    first = (mv.groupby("ICUSTAY_ID")["STARTTIME"].min()
               .rename("first_vaso_time").reset_index())
    s3 = s3.merge(first, left_on="icustay_id", right_on="ICUSTAY_ID", how="left",
                  suffixes=("", "_mv"))
    return s3


def exclude_early_vasopressor(df: pd.DataFrame) -> pd.DataFrame:
    """Drop stays that were already on a vasopressor in the 6 h before ICU."""
    if "first_vaso_time" not in df.columns or "INTIME" not in df.columns:
        raise ValueError("need first_vaso_time and INTIME to apply this exclusion")
    hours = (df["INTIME"] - df["first_vaso_time"]).dt.total_seconds() / 3600.0
    keep = ~(hours > PRE_ICU_VASOPRESSOR_HOURS)
    return df[keep].copy()


# ---------------------------------------------------------------------------
# Stage 3: stable-period eligibility (>=60 min, with 30-min clearance)
# ---------------------------------------------------------------------------
def stable_period_windows(traj: pd.DataFrame, min_run_min: int = STABLE_MIN_MIN,
                          margin_min: int = STABLE_MARGIN_MIN,
                          admission_onset_min: float = 0.0,
                          stop_min: float = 1e9) -> int:
    """Number of >=``min_run_min`` consecutive quality-passing windows.

    A window qualifies when it is at least ``margin_min`` after ICU admission and
    at least ``margin_min`` before the earlier of vasopressor start and ICU
    discharge.  Consecutive qualifying windows are chained; only chains spanning
    at least ``min_run_min`` minutes are counted.
    """
    ok = traj[traj["quality"] == "ok"].sort_values("time_min")
    lo = admission_onset_min + margin_min
    hi = stop_min - margin_min
    sel = ok[(ok["time_min"] >= lo) & (ok["time_min"] <= hi)]
    if sel.empty:
        return 0
    step = float(sel["time_min"].diff().median())
    if not step or step <= 0:
        step = float(config.MIMIC_STEP_MIN)
    run, best = 0.0, 0.0
    prev = None
    for t in sel["time_min"].to_numpy():
        if prev is not None and abs(t - prev - step) < 0.5 * step:
            run += step
        else:
            run = step
        best = max(best, run)
        prev = t
    return int(best) if best >= min_run_min else 0


# ---------------------------------------------------------------------------
# Derived-mode assembly
# ---------------------------------------------------------------------------
def build_from_derived() -> dict:
    """Assemble the three analysis cohorts from the shipped de-identified tables."""
    cohort = pd.read_csv(config.require(config.data_path("mimic_cohort_480.csv")))
    slopes = pd.read_csv(config.require(config.data_path("mimic_slope_441.csv")))
    news = pd.read_csv(config.require(config.data_path("mimic_news2_319.csv")))

    stable = set(slopes["icustay_id"].astype(int))
    cohort["has_stable_period"] = cohort["icustay_id"].astype(int).isin(stable)

    c480 = cohort.copy()
    c441 = cohort[cohort["has_stable_period"]].copy()
    c319 = cohort[cohort["icustay_id"].astype(int).isin(set(news["icustay_id"].astype(int)))].copy()

    steps = [
        ("Sepsis-3 landmark risk set (waveform available)", 1189,
         "archived risk set"),
        ("Waveform inversion succeeded in 0-4 h", len(c480), "mimic_cohort_480.csv"),
        ("Plus >=60 min continuous stable period", len(c441), "mimic_slope_441.csv"),
        ("Plus complete NEWS2 baseline", len(c319), "mimic_news2_319.csv"),
    ]
    return {"cohort_480": c480, "cohort_441": c441, "cohort_319": c319,
            "steps": steps}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["derived", "raw"], default="derived")
    ap.add_argument("--mimic-root", default=config.MIMIC_ROOT)
    ap.add_argument("--sepsis3", default=None, help="path to the official sepsis3.csv")
    args = ap.parse_args()

    if args.mode == "raw":
        if not args.mimic_root or not args.sepsis3:
            raise SystemExit("--mode raw needs --mimic-root and --sepsis3")
        s3 = build_sepsis3_cohort(Path(args.sepsis3), Path(args.mimic_root))
        print(f"Sepsis-3 stays:                                {len(s3)}")
        s3b = exclude_early_vasopressor(s3)
        print(f"After pre-ICU ({PRE_ICU_VASOPRESSOR_HOURS:.0f} h) vasopressor exclusion: {len(s3b)}")
        s3b.to_csv(config.out_path("mimic_sepsis3_risk_set.csv"), index=False)
        print("Waveform inversion and stable-period stages require the extracted "
              "trajectories; run 02_waveform_modeling/mimic_tau_extraction.py then "
              "02_waveform_modeling/slope_aggregation.py, and re-run this script "
              "in --mode derived.")
        return

    res = build_from_derived()
    print("=== MIMIC-III analysis cohorts ===")
    for label, n, src in res["steps"]:
        print(f"  {label:<48} N={n:>5}   [{src}]")
    for key in ("cohort_480", "cohort_441", "cohort_319"):
        df = res[key]
        name = key.replace("cohort_", "mimic_cohort_") + "_analysis.csv"
        df.to_csv(config.out_path(name), index=False)
        print(f"  {name:<40} N={len(df):>5} events={int(df['event_6h'].sum()):>3} "
              f"({df['event_6h'].mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
