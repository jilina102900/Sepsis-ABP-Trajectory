# -*- coding: utf-8 -*-
"""MIMIC-III arterial-waveform segmentation and tau extraction.

Implements the primary extraction route described in Methods:

* channel selection: any label containing ``ABP`` or ``ART`` (case-insensitive);
* segment-level QC: >=5 s usable signal, <50 % missing samples, mean pressure
  inside 20-200 mmHg, no flat stretch;
* 15-min rolling windows advancing every 5 min over the 0-4 h landmark window;
* tau from the free-asymptote decay model (``windkessel_lib.estimate_tau_freePinf``),
  decay window opening 0.15 s after the systolic peak, beats with
  NRMSE > 0.08 rejected, tau = median of accepted beat-level values (seconds);
* tau-proxy ``R = MAP / (HR * PP)`` in min (dimensionless ratio with time unit);
* window aggregation of tau uses the median of accepted beats.

The raw waveforms are protected and are not redistributed; this script therefore
requires a local MIMIC-III Waveform Database installation.  It writes one
trajectory CSV per segment, which is the input of
``02_waveform_modeling/slope_aggregation.py``.

Usage
-----
    python 02_waveform_modeling/mimic_tau_extraction.py \
        --wave-dir  /path/to/mimic3wdb-matched \
        --metadata  data/mimic_segment_metadata.csv \
        --out-dir   outputs/mimic_trajectories
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
import windkessel_lib as wk  # noqa: E402


def load_waveform_segment(hea_path: Path):
    """Read one WFDB segment and return ``(signal, fs, channel_label)``.

    The ABP/ART channel is selected by label; ``wfdb`` is imported lazily so that
    the rest of the repository runs without it.
    """
    import wfdb

    rec = wfdb.rdrecord(str(hea_path.with_suffix("")))
    labels = [str(s).upper() for s in rec.sig_name]
    candidates = [i for i, s in enumerate(labels) if "ABP" in s or "ART" in s]
    if not candidates:
        raise ValueError(f"no ABP/ART channel in {hea_path.name}: {labels}")
    idx = candidates[0]
    signal = rec.p_signal[:, idx].astype(float)
    return signal, float(rec.fs), labels[idx]


def quality_check(window: np.ndarray, fs: float):
    """Segment-level QC. Returns ``(ok, reason)``."""
    if len(window) < fs * 5:
        return False, "too_short"
    if np.isnan(window).mean() > 0.5:
        return False, "too_many_nan"
    valid = window[~np.isnan(window)]
    if valid.size < fs * 5:
        return False, "few_valid"
    if not (20 <= valid.mean() <= 200):
        return False, "out_of_range"
    seg = int(5 * fs)
    for i in range(0, len(valid) - seg, seg):
        if np.std(valid[i:i + seg]) < 1e-3:
            return False, "flat"
    return True, "ok"


def summarise_beat_features(window: np.ndarray, fs: float) -> dict:
    """SBP/DBP/MAP/HR/PP and the tau-proxy for one window."""
    from scipy.signal import find_peaks

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
    return {
        "SBP": sbp, "DBP": dbp, "MAP": map_val, "HR": hr, "PP": pp,
        "R": map_val / (hr * pp) if pp > 0 and hr > 0 else np.nan,
        "peaks": peaks,
    }


def process_segment(signal: np.ndarray, fs: float,
                    window_min: int = config.MIMIC_WINDOW_MIN,
                    step_min: int = config.MIMIC_STEP_MIN,
                    horizon_min: float = 240.0) -> pd.DataFrame:
    """Slide over the 0-4 h landmark window and emit one row per window."""
    win = int(window_min * 60 * fs)
    step = int(step_min * 60 * fs)
    horizon = int(horizon_min * 60 * fs)
    rows = []
    for start in range(0, min(len(signal), horizon) - win + 1, step):
        chunk = signal[start:start + win]
        ok, reason = quality_check(chunk, fs)
        row = {"time_min": start / fs / 60.0, "quality": reason}
        if not ok:
            rows.append({**row, "R": np.nan, "tau": np.nan, "MAP": np.nan,
                         "SBP": np.nan, "DBP": np.nan, "HR": np.nan,
                         "PP": np.nan, "n_beats_accepted": 0})
            continue
        feats = summarise_beat_features(chunk, fs)
        if not feats:
            rows.append({**row, "quality": "no_beats", "R": np.nan, "tau": np.nan,
                         "MAP": np.nan, "SBP": np.nan, "DBP": np.nan,
                         "HR": np.nan, "PP": np.nan, "n_beats_accepted": 0})
            continue
        fit = wk.estimate_tau_freePinf(
            chunk, fs, peaks=feats.pop("peaks"),
            start_after_peak_s=config.MIMIC_FIT_START_AFTER_PEAK_S,
            min_window_s=config.MIMIC_MIN_WINDOW_S,
            max_nrmse=config.MIMIC_MAX_NRMSE,
        )
        rows.append({**row, **feats, "tau": fit["tau"],
                     "n_beats_accepted": fit["n_beats_accepted"],
                     "quality": "ok" if np.isfinite(fit["tau"]) else "fit_failed"})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wave-dir", required=True,
                    help="directory holding <segment>.hea / <segment>.dat")
    ap.add_argument("--metadata", default=str(config.data_path("mimic_segment_metadata.csv")),
                    help="CSV with columns subject_id, icustay_id, segment")
    ap.add_argument("--out-dir", default=str(config.out_path("mimic_trajectories")))
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N segments (smoke test)")
    args = ap.parse_args()

    wave_dir = Path(args.wave_dir)
    meta = pd.read_csv(config.require(Path(args.metadata)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = meta.head(args.limit) if args.limit else meta
    failures = []
    for i, rec in enumerate(todo.itertuples(index=False), 1):
        dst = out_dir / f"{rec.segment}.csv"
        if dst.exists() and dst.stat().st_size > 100:
            continue
        try:
            signal, fs, label = load_waveform_segment(wave_dir / f"{rec.segment}.hea")
            traj = process_segment(signal, fs)
            traj["tid"] = rec.segment
            traj["icustay_id"] = rec.icustay_id
            traj["subject_id"] = rec.subject_id
            traj["channel"] = label
            traj["fs"] = fs
            traj.to_csv(dst, index=False)
        except Exception as exc:  # noqa: BLE001 - one bad segment must not stop the batch
            failures.append((rec.segment, str(exc)[:120]))
        if i % 50 == 0:
            print(f"  processed {i}/{len(todo)}  failures={len(failures)}")

    if failures:
        pd.DataFrame(failures, columns=["segment", "error"]).to_csv(
            out_dir / "failures.csv", index=False)
    print(f"\nsegments processed: {len(todo) - len(failures)}  failed: {len(failures)}")
    print(f"trajectories written to {out_dir}")


if __name__ == "__main__":
    main()
