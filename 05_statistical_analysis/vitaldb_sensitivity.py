"""VitalDB robustness: alternative MAP thresholds and slope tail handling.

Threshold substitution
----------------------
The primary outcome uses MAP < 65 mmHg sustained for at least 1 min.  Two
alternatives are evaluated on the same records:

    MAP < 60 mmHg   a stricter definition, fewer events
    MAP < 55 mmHg   a much stricter definition, used to check that the conclusion
                    does not depend on where the line is drawn

The labels are recomputed from the 0.5 Hz ``Solar8000/ART_MBP`` numeric stream
(``num/<tid>``), which is a *numeric* stream, not the 500 Hz waveform -- the
waveform is used only for the physical features.

Slope tail handling
-------------------
The same four tail treatments applied to the MIMIC slopes (raw, 1.5 IQR, 3.0 IQR,
1%/99% winsorisation) are applied to the VitalDB case-level tau slopes.

Usage
-----
    # slope-only robustness (fast, uses the shipped features)
    python 05_statistical_analysis/vitaldb_sensitivity.py --slopes-only

    # recompute the outcome at alternative thresholds (needs the raw streams)
    python 05_statistical_analysis/vitaldb_sensitivity.py \
        --vitaldb-root /path/to/vitaldb --limit 200
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
from vitaldb_cohort import read_numeric_stream, sustained_hypotension  # noqa: E402


def slope_variants(df: pd.DataFrame, column: str) -> pd.DataFrame:
    v = df[column].to_numpy(float)
    e = df["event"].to_numpy(int) == 1
    m = np.isfinite(v)
    v, e = v[m], e[m]
    rows = [{"variant": "raw", "n": int(v.size), "events": int(e.sum()),
             "median_event": float(np.median(v[e])),
             "median_noevent": float(np.median(v[~e])) if (~e).sum() else np.nan,
             "diff_pct_h": float(np.median(v[e]) - np.median(v[~e]))}]

    q25, q75 = np.percentile(v, [25, 75])
    iqr = q75 - q25
    for k in (1.5, 3.0):
        keep = np.abs(v - np.median(v)) <= k * iqr
        rows.append({"variant": f"iqr_{k}", "n": int(keep.sum()),
                     "events": int(e[keep].sum()),
                     "median_event": float(np.median(v[keep][e[keep]])),
                     "median_noevent": float(np.median(v[keep][~e[keep]])),
                     "diff_pct_h": float(np.median(v[keep][e[keep]])
                                         - np.median(v[keep][~e[keep]]))})
    lo_p, hi_p = np.percentile(v, [1, 99])
    w = np.clip(v, lo_p, hi_p)
    rows.append({"variant": "winsor_1_99", "n": int(v.size), "events": int(e.sum()),
                 "median_event": float(np.median(w[e])),
                 "median_noevent": float(np.median(w[~e])),
                 "diff_pct_h": float(np.median(w[e]) - np.median(w[~e]))})
    return pd.DataFrame(rows)


def threshold_variants(root: Path, cohort: pd.DataFrame, limit: int | None,
                       thresholds=(65, 60, 55)) -> pd.DataFrame:
    todo = cohort.head(limit) if limit else cohort
    rows = []
    for rec in todo.itertuples(index=False):
        path = root / "num" / str(rec.tid)
        if not path.exists():
            continue
        t, v = read_numeric_stream(path)
        row = {"caseid": int(rec.caseid)}
        for thr in thresholds:
            onset, n_ev = sustained_hypotension(t, v, threshold=float(thr))
            row[f"event_map{thr}"] = int(n_ev > 0)
            row[f"n_events_map{thr}"] = int(n_ev)
            row[f"onset_map{thr}"] = onset / 60.0 if np.isfinite(onset) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_robustness(df: pd.DataFrame, tau_col: str = "tau_slope",
                          n_boot: int = 1000) -> pd.DataFrame:
    """ΔAUC(M3 − M2) after trimming or winsorising the tau slope.

    Both treatments touch only the slope feature, so the same clinical and routine
    ABP blocks are refitted with the M3 learner and the paired ΔAUC is recomputed.
    Trimming changes the cohort size, which is reported alongside.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_prediction_models"))
    from metrics import delta_auc
    from vitaldb_m1m2m3 import oof_predictions

    m1 = [c for c in config.VITAL_PRED_CLIN_COLS if c in df]
    m2 = m1 + [c for c in config.VITAL_PRED_ABP_COLS if c in df]
    m3 = m2 + [c for c in config.VITAL_PRED_PHYS_COLS if c in df]
    blocks = {"M2": m2, "M3": m3}

    v = df[tau_col].to_numpy(float)
    q25, q75 = np.percentile(v[np.isfinite(v)], [25, 75])
    iqr = q75 - q25
    lo_p, hi_p = np.percentile(v[np.isfinite(v)], [2.5, 97.5])

    variants = {
        "raw": df,
        "iqr_1.5": df[(v >= q25 - 1.5 * iqr) & (v <= q75 + 1.5 * iqr)],
        "winsor_2.5_97.5": df.assign(**{tau_col: np.clip(v, lo_p, hi_p)}),
    }
    rows = []
    for label, sub in variants.items():
        sub = sub.reset_index(drop=True)
        y = sub["event"].to_numpy(int)
        g = sub["subjectid"].to_numpy()
        p2 = oof_predictions(sub, blocks["M2"], y, g)
        p3 = oof_predictions(sub, blocks["M3"], y, g)
        d, lo, hi = delta_auc(y, p2, p3, g, n_boot=n_boot)
        rows.append({"variant": label, "n": int(len(sub)),
                     "events": int(y.sum()),
                     "auc_m2": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"])
                                     .roc_auc_score(y, p2)),
                     "auc_m3": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"])
                                     .roc_auc_score(y, p3)),
                     "delta_auc": d, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(config.data_path("vitaldb_case_features.csv")))
    ap.add_argument("--column", default="tau_slope_pct_h")
    ap.add_argument("--vitaldb-root", default=config.VITALDB_ROOT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--slopes-only", action="store_true")
    ap.add_argument("--prediction", default=None,
                    help="1,043-case prediction cohort; refits M2/M3 under slope "
                         "trimming and winsorisation")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    if args.prediction:
        pdf = pd.read_csv(config.require(Path(args.prediction)))
        if "event" not in pdf:
            pdf["event"] = pdf["first_ioh_in_pred"]
        rob = prediction_robustness(pdf, n_boot=args.n_boot)
        rob.to_csv(config.out_path("vitaldb_prediction_robustness.csv"), index=False)
        print("=== VitalDB prediction cohort: slope-tail robustness ===")
        print(rob.round(4).to_string(index=False))
        print(f"\nwritten: {config.out_path('vitaldb_prediction_robustness.csv')}")
        return

    df = pd.read_csv(config.require(Path(args.features)))
    df = df[df["n_eval_windows"] >= 1].copy()

    slopes = slope_variants(df, args.column)
    slopes.to_csv(config.out_path("vitaldb_slope_sensitivity.csv"), index=False)
    print("=== VitalDB tau-slope robustness (%/h) ===")
    print(slopes.round(2).to_string(index=False))

    if args.slopes_only:
        print(f"\nwritten: {config.out_path('vitaldb_slope_sensitivity.csv')}")
        return
    if not args.vitaldb_root:
        print("\n(skipping threshold recomputation: no --vitaldb-root)")
        return

    cohort = df.rename(columns={"n_ok_windows": "n_eval_windows"})[
        ["caseid", "tid", "n_eval_windows", "event"]]
    thr = threshold_variants(Path(args.vitaldb_root), cohort, args.limit)
    thr.to_csv(config.out_path("vitaldb_threshold_sensitivity.csv"), index=False)

    print("\n=== Alternative MAP thresholds ===")
    for t in (65, 60, 55):
        col = f"event_map{t}"
        if col in thr:
            print(f"  MAP<{t}: {int(thr[col].sum())}/{len(thr)} events "
                  f"({thr[col].mean()*100:.1f}%)")
    print(f"written: {config.out_path('vitaldb_threshold_sensitivity.csv')}")


if __name__ == "__main__":
    main()
