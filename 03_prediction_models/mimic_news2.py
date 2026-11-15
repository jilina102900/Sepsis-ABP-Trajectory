# -*- coding: utf-8 -*-
"""NEWS2 secondary analysis on the 319-stay subset with a complete baseline.

Three quantities are reported separately and must not be conflated:

* ``EHR``      -- the 13-feature EHR model, scored on this subset;
* ``EHR+ABP``  -- the 18-feature model, scored on this subset;
* ``NEWS2``    -- the aggregate NEWS2 score used *alone* as the predictor.

The distinction matters because the AUC of the EHR model on this 319-stay subset
(0.559) differs from its AUC on the full 480-stay cohort (0.656): the subset
model is **refitted inside the 319 stays** (grouped CV), not the 480-stay model
scored on a subset.  Scoring the 480-stay out-of-fold predictions on the subset
instead gives 0.619, which is the wrong quantity for this table row.

Usage
-----
    python 03_prediction_models/mimic_news2.py          # refit on the subset
    python 03_prediction_models/mimic_news2.py --no-retrain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_processing"))
import config  # noqa: E402
import metrics as M  # noqa: E402

N_BOOT = 1000
BLOCKS = {"EHR": config.MIMIC_EHR_COLS,
          "EHR+ABP": config.MIMIC_EHR_COLS + config.MIMIC_ABP_COLS}


def refit(df: pd.DataFrame, cols: list[str], y: np.ndarray,
          groups: np.ndarray) -> np.ndarray:
    """Grouped 5-fold LightGBM refitted on the NEWS2 subset itself."""
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrain", action=argparse.BooleanOptionalAction, default=True,
                    help="refit both models on the 319-stay subset (default); use "
                         "--no-retrain to score the 480-stay out-of-fold predictions")
    args = ap.parse_args()

    cohort = pd.read_csv(config.require(config.data_path("mimic_cohort_480.csv")))
    news = pd.read_csv(config.require(config.data_path("mimic_news2_319.csv")))

    sub = cohort.merge(news[["icustay_id", "news2"]], on="icustay_id", how="inner")
    y = sub["event_6h"].to_numpy(int)
    groups = sub["subject_id"].to_numpy()
    pred_ehr = sub["pred_ehr"].to_numpy(float)
    pred_both = sub["pred_both"].to_numpy(float)
    news2 = sub["news2"].to_numpy(float)

    if args.retrain:
        pred_ehr = refit(sub, [c for c in BLOCKS["EHR"] if c in sub], y, groups)
        pred_both = refit(sub, [c for c in BLOCKS["EHR+ABP"] if c in sub], y, groups)

    rows = []
    for name, p in (("EHR", pred_ehr), ("EHR+ABP", pred_both), ("NEWS2_only", news2)):
        lo, hi = M.bootstrap_auc_ci(y, p, groups, n_boot=N_BOOT)
        row = {"model": name, "n": len(y), "events": int(y.sum()),
               "auc": M.auc(y, p), "auc_lo": lo, "auc_hi": hi,
               "brier": np.nan}
        if name != "NEWS2_only":
            # NEWS2 is a count score, not a probability, so a Brier score is
            # undefined for it; only the model outputs are calibrated.
            row["brier"] = M.brier(y, p)
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(config.out_path("table3_news2.csv"), index=False)
    d, lo, hi = M.delta_auc(y, pred_ehr, pred_both, groups, n_boot=N_BOOT)
    pd.DataFrame([{"comparison": "EHR+ABP - EHR", "delta_auc": d,
                   "ci_lo": lo, "ci_hi": hi}]).to_csv(
        config.out_path("table3_news2_delta.csv"), index=False)

    print("=== NEWS2 subset (319 stays, 77 events) ===")
    print(table.round(4).to_string(index=False))
    print(f"\n  dAUC(EHR+ABP - EHR) = {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("\nNote: 'NEWS2_only' is the aggregate score alone; it is not the EHR "
          "baseline augmented with ABP.")
    print(f"written: {config.out_path('table3_news2.csv')}")


if __name__ == "__main__":
    main()
