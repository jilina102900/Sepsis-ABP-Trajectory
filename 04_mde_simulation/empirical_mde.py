"""Empirical minimal detectable effect for tau-slope drift.

The question
------------
A study of this size can only detect a slope difference if it is larger than the
noise inherent in the measurement.  Rather than assuming a parametric distribution,
the detectable difference is estimated from the cohort's own slope distribution:

1. Split the observed units into a positive and a negative group at the reported
   ratio and add a known shift ``s`` to the positive group.
2. Repeat this Monte-Carlo ``n_sim`` times and record, for every shift on a
   0.5 %/h grid, the proportion of tests reaching ``p < 0.05``; that proportion
   is the power.
3. The minimal detectable effect (MDE) is the smallest shift whose power reaches
   the target (80 %).
4. Repeat the whole procedure in an outer patient-level bootstrap so the MDE
   itself gets a confidence interval.

Zero-shift calibration
----------------------
The MDE describes measurement noise, not the effect that happens to be present in
this cohort.  Each group is therefore centred on its own median before a shift is
added, so the null case rejects at the nominal 5 % rate rather than at whatever
rate the observed group difference would produce.  Without this step the
simulated power is inflated and the MDE comes out too small.

Why the test is fast
--------------------
A shift adds a constant to one group, so only the cross-group difference matrix
``D = y_pos - y_neg`` has to be built.  Sorting ``D`` once turns every shift on
the grid into a single ``searchsorted`` count -- the Mann-Whitney U statistic --
and the p-values for the whole grid follow from one vectorised ``erfc``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erfc

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_processing"))
import config  # noqa: E402

N_SIM_BOOT_DEFAULT = 200   # draws inside each bootstrap replicate


def pooled_noise(values: np.ndarray) -> np.ndarray:
    """Per-unit noise distribution: all slopes centred on the pooled median.

    The MDE describes measurement noise, not the effect that happens to be present
    in this cohort, so the two simulated groups must share one distribution: only
    then does a zero shift reject at the nominal rate.  Centring each group on
    *its own* median would leave the groups with different shapes, and the power
    would be inflated by the observed difference the simulation is meant to
    abstract away.
    """
    return np.asarray(values, float) - np.median(values)


def smooth_draw(pool: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw from the empirical distribution through its smoothed quantile function.

    Resampling with replacement from a finite pool duplicates extreme values, which
    inflates the apparent spread as the requested sample grows.  Interpolating the
    quantile function instead keeps the distribution shape fixed while the cohort
    size changes, so power varies monotonically with ``n``.
    """
    u = rng.uniform(size=size)
    return np.quantile(pool, u)


def _pvalues_normal(counts: np.ndarray, n_pos: int, n_neg: int) -> np.ndarray:
    """Two-sided p-values for the Mann-Whitney U statistic (normal approximation).

    ``counts`` is the number of pairs in which a positive unit ranks above a
    negative one.  The continuity correction is applied because slopes on the
    simulation grid produce integers.
    """
    mean_u = n_pos * n_neg / 2.0
    var_u = n_pos * n_neg * (n_pos + n_neg + 1) / 12.0
    if var_u <= 0:
        return np.ones_like(counts, dtype=float)
    z = (np.abs(counts - mean_u) - 0.5) / np.sqrt(var_u)
    return erfc(np.maximum(z, 0.0) / np.sqrt(2.0)).astype(float)


def power_grid(pool: np.ndarray, n_pos: int, n_neg: int, shifts: np.ndarray,
               n_sim: int, rng: np.random.Generator,
               smooth: bool = True) -> tuple[np.ndarray, float]:
    """Power for every shift on the grid, plus the empirical type-I error.

    Both groups are drawn from the same noise pool at the requested sizes, so the
    zero-shift rejection rate should land near ``config.MDE_ALPHA``; that rate is
    returned rather than assumed, which is what makes the simulation checkable.
    """
    shifts = np.asarray(shifts, float)
    draw = smooth_draw if smooth else (
        lambda p, size, r: r.choice(p, size=size, replace=True))

    hits = np.zeros(shifts.size)
    null_hits = 0
    for _ in range(n_sim):
        y_pos = draw(pool, n_pos, rng)
        y_neg = draw(pool, n_neg, rng)
        diff = np.subtract.outer(y_pos, y_neg).ravel()
        diff.sort()
        # a pair counts as concordant when y_pos + s > y_neg, i.e. diff > -s
        counts = diff.size - np.searchsorted(diff, -shifts, side="right")
        hits += _pvalues_normal(counts, n_pos, n_neg) < config.MDE_ALPHA
        null_hits += int(
            _pvalues_normal(np.array([diff.size - np.searchsorted(diff, 0.0, side="right")]),
                            n_pos, n_neg)[0] < config.MDE_ALPHA)
    return hits / n_sim, null_hits / n_sim


def power_at_shifts(values: np.ndarray, positive_mask: np.ndarray,
                    shifts: np.ndarray, n_sim: int,
                    rng: np.random.Generator,
                    smooth: bool = True) -> tuple[np.ndarray, float]:
    """Power at the cohort's own group sizes, on the pooled noise distribution."""
    n_pos = int(np.asarray(positive_mask, bool).sum())
    n_neg = int(np.asarray(positive_mask, bool).size - n_pos)
    return power_grid(pooled_noise(values), n_pos, n_neg, shifts, n_sim, rng, smooth)


def mde_from_power(shifts: np.ndarray, power: np.ndarray,
                   target: float = config.MDE_TARGET_POWER) -> float:
    """Smallest shift reaching the target power (NaN when never reached)."""
    ok = np.flatnonzero(power >= target)
    return float(shifts[ok[0]]) if ok.size else float("nan")


def empirical_mde(values: np.ndarray, positive_mask: np.ndarray,
                  n_boot: int = config.MDE_N_BOOT,
                  n_sim: int = config.MDE_N_SIM,
                  n_sim_boot: int = N_SIM_BOOT_DEFAULT,
                  seed: int = config.RANDOM_SEED,
                  shifts: np.ndarray | None = None) -> dict:
    """Outer bootstrap of the MDE plus the power curve at the point estimate."""
    shifts = (np.asarray(config.MDE_SHIFT_GRID_PCT_H, float)
              if shifts is None else np.asarray(shifts, float))
    rng = np.random.default_rng(seed)
    raw = np.asarray(values, float)
    pos = np.asarray(positive_mask, bool)

    power, type_one = power_at_shifts(raw, pos, shifts, n_sim, rng)
    mde = mde_from_power(shifts, power)

    boot = []
    for _ in range(n_boot):                       # patient-level resampling
        idx = rng.integers(0, raw.size, raw.size)
        b_vals = raw[idx]
        b_pos = pos[idx]
        if b_pos.sum() < 5 or (~b_pos).sum() < 5:
            continue
        p, _ = power_at_shifts(b_vals, b_pos, shifts, n_sim_boot, rng)
        boot.append(mde_from_power(shifts, p))
    boot = np.asarray([b for b in boot if np.isfinite(b)], float)

    return {"mde": mde, "power": power, "shifts": shifts,
            "type_one": type_one,
            "mde_boot": boot,
            "mde_lo": float(np.percentile(boot, 2.5)) if boot.size else np.nan,
            "mde_hi": float(np.percentile(boot, 97.5)) if boot.size else np.nan,
            "n_units": int(raw.size), "n_pos": int(pos.sum())}


def load_mimic_slopes(column: str = "tau_proxy_slope_pct_h"):
    """441-stay slope array and its outcome flag."""
    df = pd.read_csv(config.require(config.data_path("mimic_slope_441.csv")))
    v = df[column].to_numpy(float)
    keep = np.isfinite(v)
    return v[keep], df["event_6h"].to_numpy(int)[keep] == 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sim", type=int, default=config.MDE_N_SIM)
    ap.add_argument("--n-boot", type=int, default=config.MDE_N_BOOT)
    ap.add_argument("--grid-step", type=float, default=0.5,
                    help="resolution of the imposed-shift grid, in %%/h; "
                         "0.5 is the reporting grid, a finer step locates the "
                         "80 %% crossing without changing the estimate itself")
    ap.add_argument("--grid-max", type=float, default=max(config.MDE_SHIFT_GRID_PCT_H))
    args = ap.parse_args()

    values, positive = load_mimic_slopes()
    print(f"units: {values.size}  positive: {positive.sum()} "
          f"({positive.mean() * 100:.1f}%)")
    print(f"median {np.median(values):.2f} %/h   SD {np.std(values):.2f} %/h")

    step = float(args.grid_step)
    shifts = np.round(np.arange(step, float(args.grid_max) + step / 2.0, step), 4)
    res = empirical_mde(values, positive, n_boot=args.n_boot, n_sim=args.n_sim,
                        shifts=shifts)

    print(f"type-I error at zero shift: {res['type_one']:.3f} "
          f"(nominal {config.MDE_ALPHA})")
    print(f"MDE (80 % power) = {res['mde']:.1f} %/h  "
          f"[{res['mde_lo']:.1f}, {res['mde_hi']:.1f}]  "
          f"from {res['mde_boot'].size} bootstrap replicates")

    curve = pd.DataFrame({"shift_pct_h": res["shifts"], "power": res["power"]})
    curve.to_csv(config.out_path("mimic_power_curve.csv"), index=False)
    pd.DataFrame({"mde_boot": res["mde_boot"]}).to_csv(
        config.out_path("mimic_mde_bootstrap.csv"), index=False)
    print(f"written: {config.out_path('mimic_power_curve.csv')}")
    print(f"written: {config.out_path('mimic_mde_bootstrap.csv')}")


if __name__ == "__main__":
    main()
