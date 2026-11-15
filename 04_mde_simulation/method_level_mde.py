"""Method-level MDE matrix (Figure 3 / Supplementary Table S2).

Design
------
The detectable effect depends on how a slope is aggregated: a window-level slope
carries more measurement noise than a segment-level slope, which carries more than
a patient-level summary, and it also depends on how long the observation horizon is
(60-300 min).  The matrix therefore reports MDE for

    cohort       MIMIC-III, VitalDB
    granularity  ICU/case level, segment/track level, window level
    horizon      60, 90, 120, 150, 180, 240, 300 min

Each cell of the matrix is horizon-specific, so the matrix is carried through
exactly as reported in the manuscript (``data/mde_method_level_matrix.csv``).  It
is not recomputed here: the arrays behind a cell are slopes aggregated at one
specific granularity *and* one specific horizon, and the released tables contain
only the primary-horizon patient-level array.

What *is* recomputed is a single reference point per cohort: the MDE of the
patient-level slope array at the primary horizon, estimated with the same empirical
procedure used everywhere else in this repository.  Placing that number beside the
reported 180-min patient-level cell lets a reader check that the estimator
reproduces the order of magnitude being discussed.

Usage
-----
    python 04_mde_simulation/method_level_mde.py
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

# Granularity labels used in the reported matrix -> available per-unit arrays
PATIENT_LEVEL_LABEL = {"MIMIC-III": "ICU-level", "VitalDB": "Case-level"}


def reference_arrays() -> dict:
    """Patient-level slope arrays for the two cohorts (primary horizon only)."""
    arrs = {}
    mimic = pd.read_csv(config.require(config.data_path("mimic_slope_441.csv")))
    v = mimic["tau_proxy_slope_pct_h"].to_numpy(float)
    m = np.isfinite(v)
    arrs["MIMIC-III"] = (v[m], mimic["event_6h"].to_numpy(int)[m] == 1)

    feas = pd.read_csv(config.require(config.data_path("vitaldb_case_features.csv")))
    feas = feas[feas["n_eval_windows"] >= 1]
    v = feas["tau_slope_pct_h"].to_numpy(float)
    m = np.isfinite(v)
    arrs["VitalDB"] = (v[m], feas["event"].to_numpy(int)[m] == 1)
    return arrs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reported", default=str(config.data_path("mde_method_level_matrix.csv")))
    ap.add_argument("--n-sim", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=60)
    args = ap.parse_args()

    reported = pd.read_csv(config.require(Path(args.reported)))
    reported.to_csv(config.out_path("table_s2_mde_method_level.csv"), index=False)

    print("=== Method-level MDE, as reported (%/h) ===")
    cols = [c for c in reported.columns if "min" in str(c)]
    print(reported[[reported.columns[0], reported.columns[1]] + cols]
          .round(2).to_string(index=False))

    print("\n=== Recomputed reference point (patient level, primary horizon) ===")
    rows = []
    for cohort, (v, pos) in reference_arrays().items():
        res = empirical_mde(v, pos, n_boot=args.n_boot, n_sim=args.n_sim,
                            n_sim_boot=max(args.n_boot // 2, 20))
        label = PATIENT_LEVEL_LABEL[cohort]
        cohort_key = "MIMIC" if cohort == "MIMIC-III" else cohort
        cell = reported[(reported[reported.columns[0]] == cohort_key)
                        & (reported[reported.columns[1]] == label)]
        rep_180 = float(cell["180 min"].iloc[0]) if len(cell) and "180 min" in reported else np.nan
        rows.append({"cohort": cohort, "granularity": label,
                     "n_units": int(v.size), "events": int(pos.sum()),
                     "mde_recomputed": res["mde"],
                     "mde_lo": res["mde_lo"], "mde_hi": res["mde_hi"],
                     "mde_reported_180min": rep_180})
        print(f"  {cohort:<10} {label:<10} n={v.size:>5}  "
              f"recomputed {res['mde']:.2f} [{res['mde_lo']:.2f}, {res['mde_hi']:.2f}]  "
              f"reported(180 min) {rep_180:.2f}")

    pd.DataFrame(rows).to_csv(config.out_path("mde_reference_point.csv"), index=False)
    print(f"\nwritten: {config.out_path('table_s2_mde_method_level.csv')}")
    print(f"written: {config.out_path('mde_reference_point.csv')}")


if __name__ == "__main__":
    main()
