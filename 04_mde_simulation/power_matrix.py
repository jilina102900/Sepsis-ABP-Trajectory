"""Power as a function of sample size and effect size (Figure 2).

For each candidate cohort size ``N`` the cohort's slope noise is resampled to ``N``
units at the reported event prevalence, a known shift is added to the positive
group, and the fraction of significant Mann-Whitney tests is recorded.  This
answers the planning question directly: how large would a cohort have to be to
detect a difference of 5, 7.5 or 10 %/h with 80 % power?

The observed effect is overlaid on the curve, so a reader can see at once whether
the cohort that produced the paper could ever have detected its own effect.

Usage
-----
    python 04_mde_simulation/power_matrix.py
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
from empirical_mde import pooled_noise, power_grid  # noqa: E402

N_SIM = 5000          # the reported power curve used 5000 draws per cell
SHIFTS = np.asarray(config.POWER_SHIFTS_PCT_H, float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slopes", default=str(config.data_path("mimic_slope_441.csv")))
    ap.add_argument("--column", default="tau_proxy_slope_pct_h")
    ap.add_argument("--n-sim", type=int, default=N_SIM)
    args = ap.parse_args()

    df = pd.read_csv(config.require(Path(args.slopes)))
    raw = df[args.column].to_numpy(float)
    keep = np.isfinite(raw)
    values = raw[keep]
    positive = df["event_6h"].to_numpy(int)[keep] == 1

    # One noise distribution for both groups: at zero shift the test must reject
    # at the nominal rate, otherwise the simulated power reflects the observed
    # difference rather than the measurement noise.
    pool = pooled_noise(values)

    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []
    for n in config.POWER_TOTAL_N:
        n_pos = int(round(n * config.POWER_POS_RATIO))
        n_neg = n - n_pos
        power, type_one = power_grid(pool, n_pos, n_neg, SHIFTS, args.n_sim, rng)
        for s, pw in zip(SHIFTS, power):
            rows.append({"n_total": n, "n_positive": n_pos, "shift_pct_h": float(s),
                         "power": float(pw), "type_one_error": float(type_one)})

    table = pd.DataFrame(rows)
    table.to_csv(config.out_path("power_matrix.csv"), index=False)

    print("=== Power by cohort size and effect size ===")
    print(table.pivot(index="n_total", columns="shift_pct_h",
                      values="power").round(3).to_string())
    print("\n=== Type-I error at zero shift (nominal 0.05) ===")
    print(table.groupby("n_total")["type_one_error"].first().round(3).to_string())

    # smallest cohort detecting each shift with 80 % power
    print("\n=== Cohort size needed for 80 % power ===")
    for s in SHIFTS:
        sub = table[table["shift_pct_h"] == s]
        ok = sub[sub["power"] >= config.MDE_TARGET_POWER]
        need = int(ok["n_total"].min()) if len(ok) else -1
        print(f"  shift {s:>4.1f} %/h -> "
              f"{'N = ' + str(need) if need > 0 else 'not reached up to N = '
               + str(int(sub['n_total'].max()))}")

    observed = float(np.median(values[positive]) - np.median(values[~positive]))
    print(f"\nobserved median difference: {observed:+.2f} %/h")
    print(f"written: {config.out_path('power_matrix.csv')}")


if __name__ == "__main__":
    main()
