"""Feature importance by gain share, and what the 22.3 % refers to.

Two quantities circulate in the manuscript and they must not be mixed up:

* **gain share** -- the fraction of total split gain attributable to a feature
  block, summed over the fitted trees;
* **split share** -- the fraction of splits that use a feature block, which
  counts frequency rather than magnitude and therefore weights weak splits the
  same as strong ones.

This script reports both, for both cohorts, so that any quoted percentage can be
traced to its definition and its cohort.

Reported percentage (22.3 %) is the gain share of the **waveform block in
VitalDB** (M3 minus M2).  The MIMIC analogue is printed for comparison but is a
different quantity on a different cohort.

Usage
-----
    python 05_statistical_analysis/feature_importance.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_processing"))
import config  # noqa: E402

N_FOLDS = 5


def _importance_block(df: pd.DataFrame, cols: list[str], y: np.ndarray,
                      groups: np.ndarray, impute: bool = True) -> pd.DataFrame:
    """Average gain and split counts across the CV folds.

    ``impute=False`` leaves the missing values for LightGBM to route itself, which
    is the configuration the prediction models use; the two cohorts whose numbers
    were already reported keep median imputation so their shares stay traceable.
    """
    from lightgbm import LGBMClassifier

    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    gains = np.zeros(len(cols))
    splits = np.zeros(len(cols))
    n_fold = 0
    for tr, _ in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        if impute:
            imp = SimpleImputer(strategy="median").fit(X[tr])
            Xtr = imp.transform(X[tr])
        else:
            Xtr = X[tr]
        clf = LGBMClassifier(random_state=config.RANDOM_SEED, verbose=-1)
        clf.fit(Xtr, y[tr])
        gains += clf.booster_.feature_importance(importance_type="gain")
        splits += clf.booster_.feature_importance(importance_type="split")
        n_fold += 1
    gains, splits = gains / n_fold, splits / n_fold
    return pd.DataFrame({"feature": cols, "gain": gains, "splits": splits})


def block_share(imp: pd.DataFrame, block_cols: list[str]) -> dict:
    g = imp["gain"].sum()
    s = imp["splits"].sum()
    return {"gain_share": float(imp.loc[imp["feature"].isin(block_cols), "gain"].sum() / g),
            "split_share": float(imp.loc[imp["feature"].isin(block_cols), "splits"].sum() / s)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mimic", default=str(config.data_path("mimic_cohort_480.csv")))
    ap.add_argument("--vitaldb", default=str(config.data_path("vitaldb_case_features.csv")))
    ap.add_argument("--vitaldb-prediction",
                    default=str(config.data_path("vitaldb_prediction_set_1043.csv")))
    args = ap.parse_args()

    print("=== MIMIC-III: EHR + ABP block ===")
    mdf = pd.read_csv(config.require(Path(args.mimic)))
    my = mdf["event_6h"].to_numpy(int)
    mg = mdf["subject_id"].to_numpy()
    mcols = [c for c in config.MIMIC_EHR_COLS + config.MIMIC_ABP_COLS if c in mdf]
    mimp = _importance_block(mdf, mcols, my, mg)
    mimp.to_csv(config.out_path("importance_mimic_ehr_abp.csv"), index=False)
    mshare = block_share(mimp, config.MIMIC_ABP_COLS)
    print(mimp.sort_values("gain", ascending=False).round(1).to_string(index=False))
    print(f"  ABP block: gain {mshare['gain_share']*100:.1f}%  "
          f"split {mshare['split_share']*100:.1f}%")

    print("\n=== VitalDB: M3 (clinical + routine ABP + waveform) ===")
    vdf = pd.read_csv(config.require(Path(args.vitaldb)))
    vdf = vdf[vdf["n_eval_windows"] >= 1].copy()
    vy = vdf["event"].to_numpy(int)
    vg = vdf["subjectid"].to_numpy()
    vcols = [c for c in (config.VITAL_M1_COLS + config.VITAL_M2_COLS
                         + config.VITAL_M3_COLS) if c in vdf]
    vimp = _importance_block(vdf, vcols, vy, vg)
    vimp.to_csv(config.out_path("importance_vitaldb_m3.csv"), index=False)
    vshare = block_share(vimp, config.VITAL_M3_COLS)
    print(vimp.sort_values("gain", ascending=False).round(1).to_string(index=False))
    print(f"  waveform block: gain {vshare['gain_share']*100:.1f}%  "
          f"split {vshare['split_share']*100:.1f}%")

    summary = pd.DataFrame([
        {"cohort": "MIMIC-III", "block": "ABP (5 features)",
         **mshare},
        {"cohort": "VitalDB", "block": "waveform (5 features)",
         **vshare},
    ])

    print("\n=== VitalDB prediction set (1,043 cases): M3 ===")
    pdf = pd.read_csv(config.require(Path(args.vitaldb_prediction)))
    pcols = [c for c in (config.VITAL_PRED_CLIN_COLS + config.VITAL_PRED_ABP_COLS
                         + config.VITAL_PRED_PHYS_COLS) if c in pdf]
    pimp = _importance_block(pdf, pcols, pdf["first_ioh_in_pred"].to_numpy(int),
                             pdf["subjectid"].to_numpy(), impute=False)
    pimp.to_csv(config.out_path("importance_vitaldb_prediction_m3.csv"), index=False)
    pshare = block_share(pimp, config.VITAL_PRED_PHYS_COLS)
    print(pimp.sort_values("gain", ascending=False).round(1).to_string(index=False))
    print(f"  waveform block: gain {pshare['gain_share']*100:.1f}%  "
          f"split {pshare['split_share']*100:.1f}%")

    # Same cohort, one model fitted on all rows, missing clinical values set to
    # zero: this is the convention of the original importance audit, and a block
    # share depends on whether it is averaged over folds, so both are reported.
    from lightgbm import LGBMClassifier
    X = pdf[pcols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    clf = LGBMClassifier(n_estimators=100, max_depth=3,
                         random_state=config.RANDOM_SEED, verbose=-1)
    clf.fit(X, pdf["first_ioh_in_pred"].to_numpy(int))
    g = clf.booster_.feature_importance(importance_type="gain")
    idx = [pcols.index(c) for c in config.VITAL_PRED_PHYS_COLS]
    single = float(g[idx].sum() / g[g > 0].sum())
    print(f"  same table, single model fitted on all rows: gain {single*100:.1f}%")
    summary = pd.concat([summary, pd.DataFrame([{
        "cohort": "VitalDB prediction set (1,043)",
        "block": "waveform (4 features)", **pshare,
        "gain_share_single_fit": single}])], ignore_index=True)
    summary.to_csv(config.out_path("feature_importance_summary.csv"), index=False)

    print("\n=== Summary ===")
    print(summary.round(4).to_string(index=False))
    print("\nThe quoted 22.3 % is the VitalDB waveform gain share; the MIMIC value "
          "is a different quantity and must not be substituted for it.")
    print(f"written: {config.out_path('feature_importance_summary.csv')}")


if __name__ == "__main__":
    main()
