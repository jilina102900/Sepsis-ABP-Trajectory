# -*- coding: utf-8 -*-
"""Decision-curve analysis and positive controls (Table 3).

DCA
    Net benefit is evaluated on the 30-threshold grid 0.10-0.40 (0.01 apart), and
    the difference in mean net benefit between EHR+ABP and EHR is bootstrapped at
    the patient level.  A confidence interval that straddles zero means the
    augmentation adds no usable clinical utility.

Positive controls
    If the ABP features cannot add value, then a marker known to carry
    prognostic information must still improve the model on the same cohort, or
    the null result would be an artefact of the pipeline.  Three nested controls
    are fitted: minimum MAP alone, MAP + lactate, MAP + lactate + 4-h urine
    output.

Usage
-----
    python 03_prediction_models/mimic_dca.py
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

THRESHOLDS = np.round(np.arange(0.10, 0.401, 0.01), 2)
CONTROLS = {
    "map_min": ["map_min"],
    "map_min+lactate": ["map_min", "lactate_mean"],
    "map_min+lactate+urine": ["map_min", "lactate_mean", "urine_4h"],
}


def oof_control(df: pd.DataFrame, cols: list[str], y: np.ndarray,
                groups: np.ndarray) -> np.ndarray:
    """Grouped 5-fold out-of-fold probabilities for a positive-control model.

    The controls use the same learner family as the primary model so that only
    the feature content changes between rows of the table.
    """
    from lightgbm import LGBMClassifier

    X = df[cols].to_numpy(float)
    out = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("clf", LGBMClassifier(random_state=config.RANDOM_SEED,
                                                verbose=-1))])
        pipe.fit(X[tr], y[tr])
        out[te] = pipe.predict_proba(X[te])[:, 1]
    return out


def main() -> None:
    df = pd.read_csv(config.require(config.data_path("mimic_cohort_480.csv")))
    y = df["event_6h"].to_numpy(int)
    groups = df["subject_id"].to_numpy()
    p_ehr = df["pred_ehr"].to_numpy(float)
    p_both = df["pred_both"].to_numpy(float)

    # ---- decision curve ----------------------------------------------------
    point, lo, hi, thr, diff = M.dca_difference(
        y, p_ehr, p_both, THRESHOLDS, n_boot=1000, groups=groups)
    nb_a = M.net_benefit_curve(y, p_ehr, thr)
    nb_b = M.net_benefit_curve(y, p_both, thr)
    dca = pd.DataFrame({"threshold": thr, "net_benefit_EHR": nb_a,
                        "net_benefit_EHR_ABP": nb_b, "difference": diff})
    dca.to_csv(config.out_path("table3_dca.csv"), index=False)
    print("=== Decision-curve analysis (thresholds 0.10-0.40) ===")
    print(f"  thresholds evaluated: {len(thr)}")
    print(f"  mean net-benefit difference (EHR+ABP - EHR) = {point:+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}]")

    # ---- positive controls -------------------------------------------------
    print("\n=== Positive controls ===")
    rows = []
    for name, cols in CONTROLS.items():
        cols = [c for c in cols if c in df.columns]
        p = oof_control(df, cols, y, groups)
        rows.append({"control": name, "n_features": len(cols),
                     "auc": M.auc(y, p), "brier": M.brier(y, p)})
        print(f"  {name:<26} AUC={rows[-1]['auc']:.3f}  Brier={rows[-1]['brier']:.3f}")
    pd.DataFrame(rows).to_csv(config.out_path("table3_positive_controls.csv"), index=False)
    print(f"\nwritten: {config.out_path('table3_dca.csv')}")


if __name__ == "__main__":
    main()

