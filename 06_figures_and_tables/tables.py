"""Assemble every table in the manuscript from the generated result files.

Table 1  baseline characteristics of the three MIMIC cohorts and of VitalDB
Table 2  primary model performance and incremental value
Table 3  ridge, NEWS2 subset, decision curve and positive controls
Table 4  slope comparison under the four tail treatments
Table S2 method-level MDE matrix

Nothing is retyped: each table is assembled from a CSV in ``outputs/`` that a
preceding script wrote, so a change anywhere upstream propagates to the tables.

Usage
---------
    python 06_figures_and_tables/tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_processing"))
import config  # noqa: E402

CONTINUOUS = ["age", "map_mean", "map_min", "hr_mean", "lactate_mean", "urine_4h"]


def _row(label: str, values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"characteristic": label, "n": 0, "value": "NA"}
    if values.dtype == bool or set(np.unique(values)) <= {0, 1}:
        return {"characteristic": label, "n": values.size,
                "value": f"{values.sum()} ({values.mean()*100:.1f}%)"}
    return {"characteristic": label, "n": values.size,
            "value": f"{np.median(values):.2f} [{np.percentile(values, 25):.2f}, "
                     f"{np.percentile(values, 75):.2f}]"}


def table1() -> pd.DataFrame:
    """Baseline description of each analysis cohort."""
    blocks = {}
    for key, fname in (("MIMIC 480", "mimic_cohort_480.csv"),
                       ("MIMIC 441", "mimic_slope_441.csv"),
                       ("MIMIC 319", "mimic_news2_319.csv")):
        path = config.data_path(fname)
        if path.exists():
            blocks[key] = pd.read_csv(path)
    vit = config.data_path("vitaldb_case_features.csv")
    if vit.exists():
        blocks["VitalDB"] = pd.read_csv(vit)
    pred = config.data_path("vitaldb_prediction_set_1043.csv")
    if pred.exists():
        blocks["VitalDB 1,043"] = pd.read_csv(pred)

    rows = []
    for name, df in blocks.items():
        rows.append({"cohort": name, "characteristic": "n stays / cases",
                     "value": str(len(df)), "n": len(df)})
        if "event_6h" in df:
            ev = df["event_6h"].to_numpy(int)
            rows.append({"cohort": name, "characteristic": "sepsis events",
                         "value": f"{ev.sum()} ({ev.mean()*100:.1f}%)", "n": len(df)})
        if "event" in df and "n_eval_windows" in df:
            ev = df["event"].to_numpy(int)
            rows.append({"cohort": name, "characteristic": "IOH events",
                         "value": f"{ev.sum()} ({ev.mean()*100:.1f}%)", "n": len(df)})
        if "first_ioh_in_pred" in df:
            ev = df["first_ioh_in_pred"].to_numpy(int)
            rows.append({"cohort": name, "characteristic": "IOH events",
                         "value": f"{ev.sum()} ({ev.mean()*100:.1f}%)", "n": len(df)})
            rows.append({"cohort": name, "characteristic": "unique patients",
                         "value": str(df["subjectid"].nunique()), "n": len(df)})
            rows.append({"cohort": name, "characteristic": "eligible baseline window (min)",
                         "value": _row("", df["pre_landmark_valid_win"].to_numpy(float))["value"],
                         "n": len(df)})
        for col in CONTINUOUS + ["tau_slope", "tau_proxy_slope_pct_h", "tau_base_mean",
                                 "tau_level", "tau_median", "tau_r2_5min"]:
            if col not in df:
                continue
            r = _row(col, df[col].to_numpy(float))
            rows.append({"cohort": name, **r})
    return pd.DataFrame(rows)


def _load(name: str) -> pd.DataFrame | None:
    path = config.out_path(name)
    return pd.read_csv(path) if path.exists() else None


def table2() -> pd.DataFrame:
    """Primary models: MIMIC-III 480 stays and the VitalDB 1,043-case cohort."""
    parts = []
    for cohort, model_csv, delta_csv in (
            ("MIMIC-III 480", "table2_primary_models.csv", "table2_delta_auc.csv"),
            ("VitalDB 1,043", "vitaldb_prediction_m1m2m3.csv",
             "vitaldb_prediction_delta_auc.csv")):
        t, d = _load(model_csv), _load(delta_csv)
        if t is not None:
            t = t.copy()
            t.insert(0, "cohort", cohort)
            parts.append(t)
        if d is not None:
            rows = [{"cohort": cohort, "model": r["comparison"], "n": np.nan,
                     "events": np.nan, "auc": r["delta_auc"], "auc_lo": r["ci_lo"],
                     "auc_hi": r["ci_hi"], "brier": np.nan,
                     "cal_intercept": np.nan, "cal_slope": np.nan}
                    for _, r in d.iterrows()]
            parts.append(pd.DataFrame(rows))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    order = ["cohort", "model", "n", "events", "auc", "auc_lo", "auc_hi",
             "brier", "cal_intercept", "cal_slope"]
    return out[[c for c in order if c in out]]


def table3() -> pd.DataFrame:
    parts = []
    for name, prefix in (("ridge", "table3_ridge.csv"),
                         ("news2", "table3_news2.csv"),
                         ("positive_controls", "table3_positive_controls.csv")):
        df = _load(prefix)
        if df is not None:
            df = df.copy()
            df.insert(0, "analysis", name)
            parts.append(df)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def main() -> None:
    outdir = config.OUT_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    writer_path = outdir / "manuscript_tables.xlsx"
    with pd.ExcelWriter(writer_path) as xl:
        t1 = table1()
        t1.to_csv(outdir / "table1_cohort.csv", index=False)
        t2 = table2()
        t3 = table3()
        t4 = _load("table4_slope_sensitivity.csv")
        ts2 = _load("table_s2_mde_method_level.csv")

        for name, df in (("Table1", t1), ("Table2", t2), ("Table3", t3),
                         ("Table4", t4), ("TableS2", ts2)):
            if df is not None and not df.empty:
                df.to_excel(xl, sheet_name=name, index=False)
                print(f"sheet {name}: {df.shape[0]} rows")

    print(f"\nwritten: {writer_path}")


if __name__ == "__main__":
    main()
