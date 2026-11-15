"""Slope comparison and robustness of the +1.91 %/h difference.

The headline slope comparison is a difference of medians between stays that did
and did not develop vasopressor-requiring sepsis, expressed in %/h:

    median(event) - median(no event) = +1.91 %/h

Because tau-slope distributions are strongly heavy-tailed (the MIMIC cohort
contains slopes from -1456 to +1858 %/h), the same comparison is repeated under
four different treatments of the tails.  If the difference survives all four, the
estimate is a property of the cohort rather than of the outliers.

Variants
    raw            all stays
    iqr_1.5        drop stays outside 1.5 IQR of the pooled slope
    iqr_3.0        drop stays outside 3.0 IQR of the pooled slope
    winsor_1_99    clip the pooled slope to its 1st and 99th percentiles

Usage
-----
    python 05_statistical_analysis/slope_sensitivity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_processing"))
import config  # noqa: E402

N_BOOT = 2000


def median_diff_boot(values: np.ndarray, event: np.ndarray,
                     n_boot: int = N_BOOT, seed: int = config.RANDOM_SEED):
    """Bootstrap CI of the median difference, resampling whole stays."""
    rng = np.random.default_rng(seed)
    n = values.size
    diffs = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        v, e = values[idx], event[idx]
        if e.sum() == 0 or (1 - e).sum() == 0:
            continue
        diffs[b] = np.median(v[e]) - np.median(v[~e])
    lo, hi = np.percentile(diffs[np.isfinite(diffs)], [2.5, 97.5])
    return float(lo), float(hi)


def summarise(label: str, values: np.ndarray, event: np.ndarray) -> dict:
    m1 = float(np.median(values[event]))
    m0 = float(np.median(values[~event]))
    diff = m1 - m0
    lo, hi = median_diff_boot(values, event)
    p = float(mannwhitneyu(values[event], values[~event],
                           alternative="two-sided").pvalue)
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    return {"variant": label, "n": int(values.size),
            "n_event": int(event.sum()), "median_event": m1, "median_noevent": m0,
            "diff_pct_h": diff, "diff_lo": lo, "diff_hi": hi,
            "p_mannwhitney": p, "pooled_iqr": iqr}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slopes", default=str(config.data_path("mimic_slope_441.csv")))
    ap.add_argument("--column", default="tau_proxy_slope_pct_h")
    args = ap.parse_args()

    df = pd.read_csv(config.require(Path(args.slopes)))
    v = df[args.column].to_numpy(float)
    e = df["event_6h"].to_numpy(int) == 1
    keep = np.isfinite(v)
    v, e = v[keep], e[keep]

    rows = [summarise("raw", v, e)]

    q25, q75 = np.percentile(v, [25, 75])
    iqr = q75 - q25
    for k in (1.5, 3.0):
        m = np.abs(v - np.median(v)) <= k * iqr
        rows.append(summarise(f"iqr_{k}", v[m], e[m]))

    lo_p, hi_p = np.percentile(v, [1, 99])
    rows.append(summarise("winsor_1_99", np.clip(v, lo_p, hi_p), e))

    table = pd.DataFrame(rows)
    table.to_csv(config.out_path("table4_slope_sensitivity.csv"), index=False)

    print("=== Slope comparison under four tail treatments (%/h) ===")
    print(table.round(3).to_string(index=False))
    print(f"\nwritten: {config.out_path('table4_slope_sensitivity.csv')}")


if __name__ == "__main__":
    main()
