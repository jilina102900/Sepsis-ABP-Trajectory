# -*- coding: utf-8 -*-
"""Ridge logistic sensitivity analysis (model-choice robustness, Table 3).

The primary comparison uses gradient boosting; to show that the conclusion does
not hinge on that choice, the same two feature blocks are refitted with an
L2-regularised logistic regression, which is smooth, low-variance and fully
specified by a single penalty parameter.

Pipeline (fitted inside every training fold): median imputation -> standardisation
-> logistic regression with ``C = 1``.  Grouped 5-fold CV on ``subject_id``.

Usage
-----
    python 03_prediction_models/mimic_ridge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_processing"))
import config  # noqa: E402
import metrics as M  # noqa: E402

N_SPLITS = 5
N_BOOT = 1000
BLOCKS = {"A_EHR": config.MIMIC_EHR_COLS,
          "B_EHR_ABP": config.MIMIC_EHR_COLS + config.MIMIC_ABP_COLS}


def make_model() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000,
                                   random_state=config.RANDOM_SEED)),
    ])


def oof(df: pd.DataFrame, cols: list[str], y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    X = df[cols].to_numpy(float)
    out = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        model = make_model().fit(X[tr], y[tr])
        out[te] = model.predict_proba(X[te])[:, 1]
    return out


def main() -> None:
    df = pd.read_csv(config.require(config.data_path("mimic_cohort_480.csv")))
    y = df["event_6h"].to_numpy(int)
    groups = df["subject_id"].to_numpy()

    preds, rows = {}, []
    for name, cols in BLOCKS.items():
        cols = [c for c in cols if c in df.columns]
        p = oof(df, cols, y, groups)
        preds[name] = p
        cal = M.calibration(y, p)
        rows.append({"model": f"ridge_{name}", "n_features": len(cols),
                     "auc": M.auc(y, p), "brier": M.brier(y, p),
                     "cal_intercept": cal["cal_intercept"], "cal_slope": cal["cal_slope"]})

    table = pd.DataFrame(rows)
    d, lo, hi = M.delta_auc(y, preds["A_EHR"], preds["B_EHR_ABP"], groups, n_boot=N_BOOT)
    table.to_csv(config.out_path("table3_ridge.csv"), index=False)
    pd.DataFrame([{"comparison": "EHR+ABP - EHR", "delta_auc": d,
                   "ci_lo": lo, "ci_hi": hi}]).to_csv(
        config.out_path("table3_ridge_delta.csv"), index=False)

    print("=== Ridge logistic sensitivity analysis (480 stays) ===")
    print(table.round(4).to_string(index=False))
    print(f"\n  dAUC(EHR+ABP - EHR) = {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"written: {config.out_path('table3_ridge.csv')}")


if __name__ == "__main__":
    main()
