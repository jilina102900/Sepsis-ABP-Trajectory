# -*- coding: utf-8 -*-
"""VitalDB prediction: M1 (clinical) -> M2 (+routine ABP) -> M3 (+waveform).

Cohort
------
By default the script reads ``data/vitaldb_prediction_set_1043.csv``: the
landmark cohort that survived waveform quality control (1,043 cases, 325
first-hypotension events inside the 30-min prediction window).  The wider
3,199-case trajectory table is only a descriptive resource and is *not* the
model input, so the two are never mixed.

Feature blocks
--------------

=============  ===========================================================
M1             six pre-operative clinical variables
M2             M1 plus eight conventional ABP summaries of the baseline
M3             M2 plus four physical waveform features (tau level, tau drift,
               tau dispersion, R level)
=============  ===========================================================

Hyper-parameters
----------------
The learner and the cross-validation scheme are fixed a priori to the values
used for the MIMIC analysis: LightGBM (100 trees, depth 3, learning rate 0.05,
row and column subsampling 0.8, seed 42) evaluated with grouped 5-fold
cross-validation that keeps every patient in a single fold.  Median imputation
is fitted inside each training fold, so no information leaks into the held-out
fold.  No hyper-parameter is tuned against a reported number.

Usage
-----
    python 03_prediction_models/vitaldb_m1m2m3.py
    python 03_prediction_models/vitaldb_m1m2m3.py --features path/to/table.csv
    python 03_prediction_models/vitaldb_m1m2m3.py --boot 2000 --split group
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_processing"))
import config  # noqa: E402
import metrics as M  # noqa: E402

# LightGBM configuration, fixed before any result was seen.  ``subsample`` is
# declared as in the Methods but ``subsample_freq`` is left at its default of 0,
# which is what the original configuration did: in LightGBM row subsampling only
# happens when ``subsample_freq`` >= 1, so setting it here would change the fitted
# models rather than reproduce them.  The setting is kept verbatim so the code and
# the Methods describe the same object.
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": config.RANDOM_SEED,
    "verbosity": -1,
}


def build_blocks(m1: list[str], m2: list[str], m3: list[str]) -> dict[str, list[str]]:
    """Nest the three feature blocks so each model is strictly a superset."""
    return {"M1": list(m1), "M2": list(m1) + list(m2), "M3": list(m1) + list(m2) + list(m3)}


def oof_predictions(df: pd.DataFrame, cols: list[str], y: np.ndarray,
                    groups: np.ndarray, split: str = "stratified_group") -> np.ndarray:
    """Out-of-fold probabilities for one feature block.

    Missing values are left in place: LightGBM routes them to the side of each
    split that fits the training fold best, which is how the analysis was run
    (no imputation was applied before the learner).  Keeping that behaviour also
    keeps the reported probabilities free of imputation artefacts such as a
    fixed value standing in for an unmeasured pre-operative variable.
    """
    from lightgbm import LGBMClassifier

    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    out = np.full(len(df), np.nan)

    if split == "stratified_group":
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
        folds = cv.split(X, y, groups)
    else:
        folds = GroupKFold(n_splits=5).split(X, y, groups)

    for tr, te in folds:
        clf = LGBMClassifier(**LGB_PARAMS)
        clf.fit(X[tr], y[tr])
        out[te] = clf.predict_proba(X[te])[:, 1]
    return out


def load_table(path: Path, id_col: str | None) -> pd.DataFrame:
    """Read the analysis table and normalise the outcome column name.

    Row order is preserved on purpose.  A shuffled grouped split assigns folds by
    position, so re-sorting the table would silently change the fold membership
    and therefore the reported out-of-fold predictions.
    """
    df = pd.read_csv(config.require(path))
    if "event" not in df.columns:
        for alias in ("first_ioh_in_pred", "label", "y", "outcome"):
            if alias in df.columns:
                df["event"] = df[alias]
                break
    if "event" not in df.columns:
        raise KeyError("no outcome column found (expected 'event' or 'first_ioh_in_pred')")
    df = df.dropna(subset=["event"]).copy()
    df["event"] = df["event"].astype(int)
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features",
                    default=str(config.data_path("vitaldb_prediction_set_1043.csv")),
                    help="one row per case, features and outcome in the same table")
    ap.add_argument("--labels", default=None,
                    help="optional separate label table merged on --id-column")
    ap.add_argument("--id-column", default="caseid")
    ap.add_argument("--m1-cols", default=",".join(config.VITAL_PRED_CLIN_COLS))
    ap.add_argument("--m2-cols", default=",".join(config.VITAL_PRED_ABP_COLS))
    ap.add_argument("--m3-cols", default=",".join(config.VITAL_PRED_PHYS_COLS))
    ap.add_argument("--split", choices=("stratified_group", "group"),
                    default="stratified_group")
    ap.add_argument("--boot", type=int, default=1000,
                    help="patient-level bootstrap replicates for the AUC intervals")
    args = ap.parse_args()

    df = load_table(Path(args.features), args.id_column)

    if args.labels:
        lab = pd.read_csv(config.require(Path(args.labels)))
        cols = [c for c in (args.id_column, "event", "first_ioh_in_pred")
                if c in lab.columns]
        lab = lab[cols].rename(columns={"first_ioh_in_pred": "event"})
        df = df.merge(lab, on=args.id_column, how="left", suffixes=("", "_label"))
        if "event_label" in df.columns:
            df["event"] = df["event_label"].fillna(df.get("event"))
            df = df.drop(columns=["event_label"])

    blocks = build_blocks(args.m1_cols.split(","), args.m2_cols.split(","),
                          args.m3_cols.split(","))
    blocks = {k: [c.strip() for c in v if c.strip()] for k, v in blocks.items()}

    groups = (df["subjectid"].to_numpy() if "subjectid" in df.columns
              else np.arange(len(df)))
    y = df["event"].to_numpy(int)

    print(f"cohort: {len(df)} cases | events {int(y.sum())} "
          f"({y.mean() * 100:.2f}%) | patients {len(np.unique(groups))}")
    print(f"split: {args.split} (5 folds) | LightGBM seed {config.RANDOM_SEED}")

    preds, rows = {}, []
    for name, cols in blocks.items():
        present = [c for c in cols if c in df.columns]
        absent = [c for c in cols if c not in df.columns]
        if absent:
            print(f"  {name}: not present in the table, skipped -> {absent}")
        if not present:
            continue
        p = oof_predictions(df, present, y, groups, args.split)
        preds[name] = p
        lo, hi = M.bootstrap_auc_ci(y, p, groups, n_boot=args.boot)
        cal = M.calibration(y, p)
        rows.append({"model": name, "n_features": len(present), "n": len(y),
                     "events": int(y.sum()), "auc": M.auc(y, p),
                     "auc_lo": lo, "auc_hi": hi, "brier": M.brier(y, p),
                     "cal_intercept": cal["cal_intercept"],
                     "cal_slope": cal["cal_slope"]})

    table = pd.DataFrame(rows)
    table.to_csv(config.out_path("vitaldb_prediction_m1m2m3.csv"), index=False)
    print("\n=== VitalDB M1 / M2 / M3 (out-of-fold) ===")
    print(table.round(4).to_string(index=False))

    print("\n=== Incremental value ===")
    inc = []
    for base, aug in (("M1", "M2"), ("M2", "M3")):
        if base not in preds or aug not in preds:
            continue
        d, lo, hi = M.delta_auc(y, preds[base], preds[aug], groups, n_boot=args.boot)
        inc.append({"comparison": f"{aug} - {base}", "delta_auc": d,
                    "ci_lo": lo, "ci_hi": hi})
        print(f"  {aug} - {base}: dAUC = {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    pd.DataFrame(inc).to_csv(config.out_path("vitaldb_prediction_delta_auc.csv"), index=False)

    out = df[[c for c in (args.id_column, "subjectid", "event") if c in df.columns]].copy()
    for name, p in preds.items():
        out[f"pred_{name}"] = p
    out.to_csv(config.out_path("vitaldb_prediction_oof.csv"), index=False)
    print(f"\nwritten: {config.out_path('vitaldb_prediction_m1m2m3.csv')}")


if __name__ == "__main__":
    main()
