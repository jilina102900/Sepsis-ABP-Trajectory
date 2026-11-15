# -*- coding: utf-8 -*-
"""One-off export of de-identified analysis tables into ``./data``.

This script converts the archived, de-identified intermediate artefacts into the
flat CSV tables that every downstream script in this repository reads.  It is
NOT part of the analysis pipeline proper: it exists so that the analysis can be
re-run on machines that do not hold the protected raw sources.

Every export drops direct identifiers other than the pseudonymous
``subject_id`` / ``icustay_id`` / ``caseid`` keys that MIMIC-III and VitalDB
already ship as de-identified codes, and it drops no clinical variable used in
the manuscript.

Usage
-----
    python 01_data_processing/export_derived_data.py --source <archive_root>

``--source`` is the folder holding the archived artefacts.  The script only
reads; it never modifies the source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def _mimic_tables(src: Path) -> None:
    face = pd.read_pickle(src / "df_face_off_final.pkl")
    phys = pd.read_pickle(src / "df_physics_v2.pkl")
    news = pd.read_pickle(src / "news2_baseline_results.pkl")
    drift = pd.read_csv(src / "tau_drift_by_event_v2.csv")

    # ---- 480-case analysis table -----------------------------------------
    keep = (["subject_id", "icustay_id", "event_6h"]
            + config.MIMIC_EHR_COLS
            + ["MAP_mean", "HR_mean", "tau_slope", "tau_proxy_slope",
               "last_tau", "last_tau_proxy", "PP_mean"]
            + ["pred_ehr", "pred_both", "pred_phys"])
    cohort = face[[c for c in keep if c in face.columns]].copy()
    cohort = cohort.sort_values("icustay_id").reset_index(drop=True)
    cohort.to_csv(config.data_path("mimic_cohort_480.csv"), index=False)
    print(f"mimic_cohort_480.csv        rows={len(cohort):<6} events={int(cohort.event_6h.sum())}")

    # ---- 441-patient stable-period slope table ---------------------------
    m = drift.merge(phys[["icustay_id", "R_slope", "tau_slope"]], on="icustay_id", how="left")
    m = m.merge(face[["icustay_id", "subject_id"]].drop_duplicates("icustay_id"),
                on="icustay_id", how="left")
    out = pd.DataFrame({
        "icustay_id": m["icustay_id"].astype("int64"),
        "subject_id": m["subject_id"].astype("Int64"),
        "event_6h": m["event"].astype(int),
        # tau-proxy relative slope, %/h (primary between-group comparison)
        "tau_proxy_slope_pct_h": m["slope"] * 60.0,
        # tau relative slope, %/h
        "tau_slope_pct_h": m["tau_slope"],
        # trajectory-fit quality metric reported in Results (84.8% / median 0.0066)
        "trajectory_fit_metric": m["R_slope"],
    })
    out.to_csv(config.data_path("mimic_slope_441.csv"), index=False)
    print(f"mimic_slope_441.csv         rows={len(out):<6} events={int(out.event_6h.sum())}")

    # ---- NEWS2 subset (480 ∩ NEWS2 complete) -----------------------------
    sub = cohort[["icustay_id", "subject_id", "event_6h"]].merge(
        news[["icustay_id", "rr_mean", "spo2_mean", "temp_mean",
              "sbp_mean", "hr_mean", "gcs_mean", "news2"]],
        on="icustay_id", how="inner")
    sub = sub.sort_values("icustay_id").reset_index(drop=True)
    sub.to_csv(config.data_path("mimic_news2_319.csv"), index=False)
    print(f"mimic_news2_319.csv         rows={len(sub):<6} events={int(sub.event_6h.sum())}")

    # ---- outer bootstrap distribution of the MDE -------------------------
    boot = pd.read_csv(src / "mde_bootstrap_1000times_final.csv")
    boot.to_csv(config.data_path("mimic_mde_bootstrap_1000.csv"), index=False)
    print(f"mimic_mde_bootstrap_1000.csv rows={len(boot)}")

    # ---- method-level MDE matrix ----------------------------------------
    mat = pd.read_csv(src / "table_s2_mde_matrix.csv")
    mat.to_csv(config.data_path("mde_method_level_matrix.csv"), index=False)
    print(f"mde_method_level_matrix.csv  shape={mat.shape}")


def _vitaldb_tables(src: Path) -> None:
    traj = pd.read_pickle(src / "vitaldb_all_trajectories_with_hypo.pkl")
    traj.to_csv(config.data_path("vitaldb_trajectories.csv.gz"),
                index=False, compression="gzip")
    print(f"vitaldb_trajectories.csv.gz  rows={len(traj):<6} cases={traj.caseid.nunique()}")

    out = pd.read_csv(src / "vitaldb_case_outcomes.csv")
    out.to_csv(config.data_path("vitaldb_case_outcomes.csv"), index=False)
    print(f"vitaldb_case_outcomes.csv    rows={len(out):<6} events={int(out.has_hypotension.sum())}")

    slopes = pd.read_csv(src / "vitaldb_stable_slopes.csv")
    slopes.to_csv(config.data_path("vitaldb_window_slopes.csv"), index=False)
    print(f"vitaldb_window_slopes.csv    rows={len(slopes)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True,
                    help="folder holding the archived MIMIC-III artefacts")
    ap.add_argument("--vitaldb-source", default=None,
                    help="folder holding the archived VitalDB artefacts "
                         "(defaults to --source)")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.is_dir():
        raise FileNotFoundError(f"source folder not found: {src}")
    vsrc = Path(args.vitaldb_source) if args.vitaldb_source else src
    if not vsrc.is_dir():
        raise FileNotFoundError(f"VitalDB source folder not found: {vsrc}")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"data dir: {config.DATA_DIR}\n")

    print("[MIMIC-III]")
    _mimic_tables(src)
    print("\n[VitalDB]")
    _vitaldb_tables(vsrc)
    print("\nDone.")


if __name__ == "__main__":
    main()
