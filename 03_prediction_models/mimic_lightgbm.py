# -*- coding: utf-8 -*-
"""MIMIC-III primary LightGBM models: EHR baseline vs EHR + ABP.

Three feature blocks are compared on the same 480-stay cohort:

    A  EHR baseline            13 features
    B  EHR + ABP               18 features
    C  pure ABP                 5 features

Out-of-fold predictions are produced with grouped 5-fold cross-validation on
``subject_id`` so that no patient contributes to both training and test folds;
median imputation is fitted inside each training fold to prevent leakage.

Two run modes
-------------
``--from-oof`` (default)
    Scores the archived out-of-fold predictions shipped in
    ``data/mimic_cohort_480.csv`` (columns ``pred_ehr``, ``pred_both``,
    ``pred_phys``).  This reproduces every number reported in Table 2.

``--retrain``
    Refits the three models from the feature columns.  Results are a re-estimate
    of the same design and can differ in the third decimal from Table 2 because
    the archived hyper-parameters were not all persisted.

Usage
-----
    python 03_prediction_models/mimic_lightgbm.py
    python 03_prediction_models/mimic_lightgbm.py --retrain
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
import metrics as M  # noqa: E402

N_SPLITS = 5
N_BOOT = 1000

FEATURE_SETS = {
    "A_EHR": config.MIMIC_EHR_COLS,
    "B_EHR_ABP": config.MIMIC_EHR_COLS + config.MIMIC_ABP_COLS,
    "C_ABP": config.MIMIC_ABP_COLS,
}
OOF_COLUMNS = {"A_EHR": "pred_ehr", "B_EHR_ABP": "pred_both", "C_ABP": "pred_phys"}


def oof_predict(df: pd.DataFrame, cols: list[str], groups: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Grouped 5-fold out-of-fold probabilities with in-fold imputation."""
    from lightgbm import LGBMClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GroupKFold

    X = df[cols].to_numpy(float)
    oof = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        imp = SimpleImputer(strategy="median").fit(X[tr])
        clf = LGBMClassifier(random_state=config.RANDOM_SEED, verbose=-1)
        clf.fit(imp.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(imp.transform(X[te]))[:, 1]
    return oof


def score_block(name: str, y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> dict:
    lo, hi = M.bootstrap_auc_ci(y, p, groups, n_boot=N_BOOT)
    cal = M.calibration(y, p)
    return {"model": name, "n": len(y), "events": int(y.sum()),
            "auc": M.auc(y, p), "auc_lo": lo, "auc_hi": hi,
            "brier": M.brier(y, p),
            "cal_intercept": cal["cal_intercept"], "cal_slope": cal["cal_slope"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", default=str(config.data_path("mimic_cohort_480.csv")))
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(config.require(Path(args.cohort)))
    y = df["event_6h"].to_numpy(int)
    groups = df["subject_id"].to_numpy()

    rows, preds = [], {}
    for name, cols in FEATURE_SETS.items():
        if args.retrain:
            cols = [c for c in cols if c in df.columns]
            p = oof_predict(df, cols, groups, y)
        else:
            p = df[OOF_COLUMNS[name]].to_numpy(float)
        preds[name] = p
        rows.append(score_block(name, y, p, groups))

    table = pd.DataFrame(rows)
    table.to_csv(config.out_path("table2_primary_models.csv"), index=False)
    print("=== Table 2: primary model performance (480 stays, 145 events) ===")
    print(table.round(3).to_string(index=False))

    # ---- constant-prediction reference ------------------------------------
    # The constant is the training-fold event rate; here the cohort prevalence,
    # which is what a model with no information would predict.
    ref = {"model": "constant_prediction", "n": len(y), "events": int(y.sum()),
           "auc": np.nan, "auc_lo": np.nan, "auc_hi": np.nan,
           "brier": M.constant_brier(y), "cal_intercept": np.nan, "cal_slope": np.nan}
    print(f"\nconstant prediction Brier = {ref['brier']:.3f} (event rate {y.mean():.3f})")

    # ---- incremental value -------------------------------------------------
    print("\n=== Incremental value (patient-level bootstrap, 1000 replicates) ===")
    inc = []
    for base, aug in (("A_EHR", "B_EHR_ABP"), ("C_ABP", "B_EHR_ABP")):
        d, lo, hi = M.delta_auc(y, preds[base], preds[aug], groups, n_boot=N_BOOT)
        inc.append({"comparison": f"{aug} - {base}", "delta_auc": d,
                    "ci_lo": lo, "ci_hi": hi})
        print(f"  {aug} - {base}: dAUC={d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    pd.DataFrame(inc).to_csv(config.out_path("table2_delta_auc.csv"), index=False)

    out = df[["subject_id", "icustay_id", "event_6h"]].copy()
    for name, col in OOF_COLUMNS.items():
        out[col] = preds[name]
    out.to_csv(config.out_path("mimic_oof_predictions.csv"), index=False)
    print(f"\nwritten: {config.out_path('table2_primary_models.csv')}")


if __name__ == "__main__":
    main()
