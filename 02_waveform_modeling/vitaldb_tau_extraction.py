# -*- coding: utf-8 -*-
"""VitalDB invasive-ABP preprocessing and tau extraction.

VitalDB differs from MIMIC-III in three ways that had to be handled explicitly
(Methods, "VitalDB preprocessing adaptation"):

1. the exported ``wav/<tid>`` files are plain text (``Time,SNUADC/ART``, 500 Hz)
   and carry slow baseline drift, removed here with a 1-s running median before
   any beat segmentation;
2. estimation windows are 5 min wide (step 5 min) instead of 15 min, because
   anaesthetic records are shorter and the trajectory must resolve a
   30-min landmark;
3. fitted tau > 0.5 s is treated as a degenerate fit and discarded.  Applying the
   MIMIC pipeline unchanged to these records fails in roughly half of the
   windows, so this guard is part of the pipeline rather than a fallback.

Usage
-----
    python 02_waveform_modeling/vitaldb_tau_extraction.py \
        --vitaldb-root /path/to/vitaldb --limit 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_processing"))
import config  # noqa: E402
import windkessel_lib as wk  # noqa: E402


def load_vitaldb_waveform(path: Path):
    """Read a VitalDB ``wav/<tid>`` text export and return ``(signal, fs)``."""
    head = pd.read_csv(path, nrows=5)
    if "Time" not in head.columns:
        raise ValueError(f"unexpected waveform layout in {path.name}: {list(head.columns)}")
    df = pd.read_csv(path)
    t = df["Time"].dropna().to_numpy(float)
    fs = float(round(1.0 / np.median(np.diff(t[:2000]))))
    signal = df[df.columns[1]].to_numpy(float)
    return signal, fs


def remove_baseline_drift(signal: np.ndarray, fs: float,
                          window_s: float = config.VITAL_MEDIAN_FILTER_S) -> np.ndarray:
    """Subtract a 1-s running median, then restore the median pressure level.

    The pulse period (0.4-1.2 s) is shorter than two filter lengths, so the
    median tracks only the slow component and leaves the systolic/diastolic
    morphology intact.
    """
    size = max(3, int(round(window_s * fs)))
    if size % 2 == 0:
        size += 1
    baseline = median_filter(signal, size=size, mode="nearest")
    return signal - baseline + float(np.nanmedian(signal))


def summarise(window: np.ndarray, fs: float) -> dict:
    """SBP/DBP/MAP/HR/PP plus the tau-proxy for a single window."""
    filled = np.where(np.isnan(window), np.nanmean(window), window)
    peaks, _ = find_peaks(filled, distance=int(0.4 * fs), prominence=10)
    troughs, _ = find_peaks(-filled, distance=int(0.4 * fs), prominence=10)
    if len(peaks) < 3 or len(troughs) < 3:
        return {}
    sbp = float(np.median(filled[peaks]))
    dbp = float(np.median(filled[troughs]))
    map_val = float(np.nanmean(window))
    hr = float(60.0 / np.median(np.diff(peaks) / fs))
    pp = sbp - dbp
    return {"SBP": sbp, "DBP": dbp, "MAP": map_val, "HR": hr, "PP": pp,
            "R": map_val / (hr * pp) if pp > 0 and hr > 0 else np.nan,
            "peaks": peaks}


def process_record(signal: np.ndarray, fs: float,
                   window_min: int = config.VITAL_WINDOW_MIN,
                   step_min: int = config.VITAL_WINDOW_MIN) -> pd.DataFrame:
    """5-min windows, 5-min step, tau guard at ``VITAL_TAU_MAX_S``."""
    clean = remove_baseline_drift(signal, fs)
    win = int(window_min * 60 * fs)
    step = int(step_min * 60 * fs)
    rows = []
    for start in range(0, max(len(clean) - win, 0) + 1, step):
        chunk = clean[start:start + win]
        t0 = start / fs / 60.0
        if np.isnan(chunk).mean() > 0.5 or np.nanmean(chunk) < 20 or np.nanmean(chunk) > 200:
            rows.append({"time_min": t0, "quality": "out_of_range"})
            continue
        feats = summarise(chunk, fs)
        if not feats:
            rows.append({"time_min": t0, "quality": "no_beats"})
            continue
        fit = wk.estimate_tau_freePinf(
            chunk, fs, peaks=feats.pop("peaks"),
            start_after_peak_s=config.MIMIC_FIT_START_AFTER_PEAK_S,
            min_window_s=config.MIMIC_MIN_WINDOW_S,
            max_nrmse=config.MIMIC_MAX_NRMSE,
        )
        tau = fit["tau"]
        quality = "ok"
        if not np.isfinite(tau):
            quality = "fit_failed"
        elif tau > config.VITAL_TAU_MAX_S:
            quality, tau = "degenerate_tau", np.nan
        rows.append({"time_min": t0, **feats, "tau": tau,
                     "n_beats_accepted": fit["n_beats_accepted"], "quality": quality})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vitaldb-root", required=True)
    ap.add_argument("--out-dir", default=str(config.out_path("vitaldb_trajectories")))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.vitaldb_root)
    trks = pd.read_csv(config.require(root / "trks.csv"))
    trks["signal"] = trks["tname"].str.split("/").str[-1]
    art = trks[trks["signal"].isin(["ART", "ABP"])].copy()
    tid_to_case = dict(zip(art["tid"], art["caseid"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([f.name for f in (root / "wav").iterdir()])
    todo = files[:args.limit] if args.limit else files

    failures = []
    for i, tid in enumerate(todo, 1):
        dst = out_dir / f"{tid}.csv"
        if dst.exists() and dst.stat().st_size > 100:
            continue
        if tid not in tid_to_case:
            failures.append((tid, "tid not mapped to an ART track"))
            continue
        try:
            signal, fs = load_vitaldb_waveform(root / "wav" / tid)
            traj = process_record(signal, fs)
            traj["tid"] = tid
            traj["caseid"] = tid_to_case[tid]
            traj.to_csv(dst, index=False)
        except Exception as exc:  # noqa: BLE001
            failures.append((tid, str(exc)[:120]))
        if i % 100 == 0:
            print(f"  processed {i}/{len(todo)}  failures={len(failures)}")

    if failures:
        pd.DataFrame(failures, columns=["tid", "error"]).to_csv(
            out_dir / "failures.csv", index=False)
    print(f"\nrecords processed: {len(todo) - len(failures)}  failed: {len(failures)}")


if __name__ == "__main__":
    main()
