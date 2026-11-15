# -*- coding: utf-8 -*-
"""Synthetic validation of tau recovery (Supplementary Figure S1, data side).

A Windkessel forward simulation with a *known* tau is passed through the same
two fitting routes used on real data, and the relative recovery error is
reported per true tau and per noise level:

* ``freePinf``  -- the non-linear free-asymptote model used for MIMIC-III
  (Methods, primary route);
* ``loglinear`` -- the log-linear shared-slope fit used for waveform QC
  (``windkessel_lib.fit_tau``).

Reporting both routes matters: they do not have the same bias, and the
"recovery error < 5 %" statement in Methods refers to the free-asymptote route.

Outputs
-------
``outputs/synthetic_validation.csv``   per-tau recovery error for both routes
``outputs/synthetic_example.csv``      one representative waveform + fitted curve

Usage
-----
    python 02_waveform_modeling/synthetic_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_processing"))
import config  # noqa: E402
import windkessel_lib as wk  # noqa: E402

TAU_GRID_S = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.80, 1.20]
NOISE_MMHG = [0.0, 1.0, 2.0]
HR, SV, MAP_TARGET = 80.0, 62.5, 75.0     # CO = 5 L/min
FS = 125.0


def build_case(tau_true: float, noise_sd: float, seed: int):
    """Simulate one beat train whose true tau equals ``tau_true``."""
    co = SV * HR / 1000.0
    r = MAP_TARGET / co                       # mmHg*min/L
    c = tau_true / (r * wk.R_MINL_TO_SML)     # mL/mmHg
    sim = wk.simulate_windkessel(r, c, 0.05, HR, SV, fs=FS,
                                noise_sd=noise_sd, quant=0.05, seed=seed)
    return sim, r, c


def recover(sim) -> dict:
    """Estimate tau from a simulated trace with both fitting routes."""
    p, fs = sim["p"], FS
    peaks = wk.detect_peaks(p, fs)
    free = wk.estimate_tau_freePinf(
        p, fs, peaks=peaks,
        start_after_peak_s=config.MIMIC_FIT_START_AFTER_PEAK_S,
        min_window_s=config.MIMIC_MIN_WINDOW_S,
        max_nrmse=config.MIMIC_MAX_NRMSE,
    )
    loglin = wk.invert_waveform(
        p, fs, delay_s=config.MIMIC_FIT_START_AFTER_PEAK_S, qc=True)
    tau_ll = loglin["tau_robust"]
    if not np.isfinite(tau_ll):
        tau_ll = loglin["tau_hat"]
    return {"tau_freePinf": free["tau"], "tau_loglinear": tau_ll,
            "n_beats_freePinf": free["n_beats_accepted"]}


def main() -> None:
    rows = []
    for tau_true in TAU_GRID_S:
        for noise in NOISE_MMHG:
            for seed in range(3):
                sim, _, _ = build_case(tau_true, noise, seed)
                got = recover(sim)
                for route, key in (("freePinf", "tau_freePinf"),
                                   ("loglinear", "tau_loglinear")):
                    est = got[key]
                    rows.append({
                        "tau_true_s": tau_true, "noise_mmhg": noise, "seed": seed,
                        "route": route, "tau_est_s": est,
                        "rel_error": wk.rel_err(est, tau_true),
                        "n_beats": got["n_beats_freePinf"],
                    })
    df = pd.DataFrame(rows)
    df.to_csv(config.out_path("synthetic_validation.csv"), index=False)

    # Representative example for the figure (true tau = 0.200 s, 1 mmHg noise)
    sim, _, _ = build_case(0.200, 1.0, seed=0)
    peaks = wk.detect_peaks(sim["p"], FS)
    free = wk.estimate_tau_freePinf(
        sim["p"], FS, peaks=peaks,
        start_after_peak_s=config.MIMIC_FIT_START_AFTER_PEAK_S,
        min_window_s=config.MIMIC_MIN_WINDOW_S,
    )
    ex = pd.DataFrame({"t_s": sim["t"], "p_mmhg": sim["p"],
                       "p_clean_mmhg": sim["p_clean"]})
    ex.to_csv(config.out_path("synthetic_example.csv"), index=False)
    ex.attrs = {}

    print("=== tau recovery error by true tau (median over 3 seeds) ===")
    piv = (df.groupby(["tau_true_s", "noise_mmhg", "route"])["rel_error"]
             .median().mul(100).unstack("route").round(2))
    print(piv.to_string())
    print("\nmedian relative error, freePinf route: "
          f"{df.loc[df.route=='freePinf', 'rel_error'].median()*100:.2f}%")
    print("median relative error, loglinear route: "
          f"{df.loc[df.route=='loglinear', 'rel_error'].median()*100:.2f}%")
    print(f"\nexample (true tau=0.200 s, noise 1 mmHg): "
          f"tau_freePinf={free['tau']:.3f} s "
          f"(error {wk.rel_err(free['tau'], 0.2)*100:.1f}%), "
          f"beats accepted={free['n_beats_accepted']}")
    print(f"\nwritten: {config.out_path('synthetic_validation.csv')}")


if __name__ == "__main__":
    main()
