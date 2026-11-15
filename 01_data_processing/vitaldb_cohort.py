# -*- coding: utf-8 -*-
"""VitalDB risk-set construction and intra-operative hypotension labelling.

Cascade (counts reported by the manuscript are printed as a reference column):

    6,388 anaesthetic records
      -> adults undergoing non-cardiac surgery                      3,164
      -> a usable invasive arterial waveform                         2,766
      -> no hypotension before the 30-min model landmark             1,981
      -> >=1 evaluable 5-min estimation window (main risk set)        1,059
      -> waveform artefact (13) or numeric-stream dropout (3) removed 1,043

Two labels exist in this cohort and they must not be conflated:

* the **outcome** for prediction follows Methods: MAP < 65 mmHg sustained for at
  least 1 min, read from the 0.5 Hz ``Solar8000/ART_MBP`` numeric stream in
  ``num/<tid>`` (a *numeric* stream, not the 500 Hz waveform);
* the **evaluability** criterion uses the 500 Hz waveform, 5-min windows, with
  tau > 0.5 s treated as a degenerate fit.

Usage
-----
    python 01_data_processing/vitaldb_cohort.py --vitaldb-root /path/to/vitaldb
    python 01_data_processing/vitaldb_cohort.py --vitaldb-root ... --limit 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

REFERENCE_COUNTS = {"records": 6388, "adult_noncardiac": 3164, "waveform": 2766,
                    "after_landmark": 1981, "risk_set": 1059, "final": 1043,
                    "events": 325}
CARDIAC_KEYWORDS = ("cardiac", "cabg", "valve", "bypass", "aortic surgery",
                    "congenital", "thoracic")


# ---------------------------------------------------------------------------
# Outcome: sustained hypotension from the numeric stream
# ---------------------------------------------------------------------------
def read_numeric_stream(path: Path, signal_hint: str = "ART_MBP"):
    """Read ``num/<tid>`` and return ``(time_s, value)`` for the MAP channel."""
    head = pd.read_csv(path, nrows=3)
    cols = list(head.columns)
    target = next((c for c in cols[1:] if signal_hint in c.upper()), None)
    if target is None:
        target = cols[1]
    df = pd.read_csv(path, usecols=["Time", target])
    t = df["Time"].to_numpy(float)
    v = df[target].to_numpy(float)
    return t, v


def sustained_hypotension(t: np.ndarray, v: np.ndarray,
                          threshold: float = config.VITAL_IOH_MAP,
                          min_duration_s: float = config.VITAL_IOH_MIN_DURATION_MIN * 60,
                          t0_s: float = 0.0):
    """First onset (seconds after ``t0_s``) of a sustained MAP drop.

    A run of consecutive samples below ``threshold`` must span at least
    ``min_duration_s``; the onset is the first sample of that run.
    Returns ``(onset_s or nan, n_events)``.
    """
    valid = np.isfinite(v) & np.isfinite(t)
    t, v = t[valid], v[valid]
    if t.size == 0:
        return np.nan, 0
    below = v < threshold
    onset, n_events = np.nan, 0
    i = 0
    while i < len(below):
        if not below[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(below) and below[j + 1]:
            j += 1
        if t[j] - t[i] >= min_duration_s:
            n_events += 1
            if not np.isfinite(onset):
                onset = float(t[i] - t0_s)
        i = j + 1
    return onset, n_events


# ---------------------------------------------------------------------------
# Waveform-side evaluability
# ---------------------------------------------------------------------------
def waveform_qc(path: Path, fs: float = 500.0) -> dict:
    """Cheap QC pass over one ``wav/<tid>`` export (no tau fitting).

    Returns window counts and artefact flags used by the exclusion steps.
    """
    head = pd.read_csv(path, nrows=5)
    if "Time" not in head.columns:
        return {"n_windows": 0, "flat_fraction": 1.0, "usable": False}
    signal = pd.read_csv(path)[lambda d: d.columns[1]].to_numpy(float)
    win = int(config.VITAL_WINDOW_MIN * 60 * fs)
    n_win = max(len(signal) // win, 0)
    n_ok, n_flat = 0, 0
    for k in range(n_win):
        chunk = signal[k * win:(k + 1) * win]
        if np.isnan(chunk).mean() > 0.5:
            continue
        if np.nanmean(chunk) < 20 or np.nanmean(chunk) > 200:
            continue
        if np.nanstd(chunk) < 1e-3:
            n_flat += 1
            continue
        n_ok += 1
    flat_fraction = n_flat / n_win if n_win else 1.0
    return {"n_windows": n_ok, "flat_fraction": flat_fraction,
            "usable": bool(n_ok >= 1 and flat_fraction < 0.5)}


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------
def adult_noncardiac(cases: pd.DataFrame) -> pd.DataFrame:
    adult = cases[cases["age"] >= 18].copy()
    text = (adult["opname"].astype(str) + " " + adult["dx"].astype(str)).str.lower()
    is_cardiac = text.str.contains("|".join(CARDIAC_KEYWORDS), na=False)
    is_cardiac |= adult["optype"].astype(str).str.lower().str.contains("thoracic|cardiac", na=False)
    return adult[~is_cardiac].copy()


def art_tracks(root: Path) -> pd.DataFrame:
    """Arterial-pressure tracks that actually have a waveform file on disk."""
    trks = pd.read_csv(config.require(root / "trks.csv"))
    trks["signal"] = trks["tname"].str.split("/").str[-1]
    art = trks[trks["signal"].isin(["ART", "ABP"])]
    have = {f.name for f in (root / "wav").iterdir()}
    return art[art["tid"].isin(have)].drop_duplicates("caseid", keep="first")


def build_cascade(root: Path, limit: int | None = None) -> pd.DataFrame:
    """Walk the cascade and return the evaluable risk set with its labels."""
    cases = pd.read_csv(config.require(root / "cases.csv"))
    print(f"  records                                        {len(cases):>6}"
          f"   (reference {REFERENCE_COUNTS['records']})")
    adult = adult_noncardiac(cases)
    print(f"  adults, non-cardiac surgery                    {len(adult):>6}"
          f"   (reference {REFERENCE_COUNTS['adult_noncardiac']})")

    art = art_tracks(root)
    with_wave = adult[adult["caseid"].isin(set(art["caseid"]))].copy()
    print(f"  usable invasive arterial waveform              {len(with_wave):>6}"
          f"   (reference {REFERENCE_COUNTS['waveform']})")

    tid_of = dict(zip(art["caseid"], art["tid"]))
    rows = []
    todo = with_wave.head(limit) if limit else with_wave
    for i, rec in enumerate(todo.itertuples(index=False), 1):
        tid = tid_of.get(rec.caseid)
        row = {"caseid": rec.caseid, "tid": tid}
        num_path = root / "num" / str(tid)
        if num_path.exists():
            t, v = read_numeric_stream(num_path)
            onset_s, n_events = sustained_hypotension(t, v)
            row["ioh_onset_min"] = onset_s / 60.0 if np.isfinite(onset_s) else np.nan
            row["n_ioh_events"] = n_events
            row["numeric_stream_ok"] = bool(np.isfinite(v).mean() > 0.5)
        else:
            row["ioh_onset_min"], row["n_ioh_events"] = np.nan, 0
            row["numeric_stream_ok"] = False
        qc = waveform_qc(root / "wav" / str(tid))
        row.update({"n_eval_windows": qc["n_windows"],
                    "flat_fraction": qc["flat_fraction"]})
        row["event"] = int(row["n_ioh_events"] > 0)
        rows.append(row)
        if i % 100 == 0:
            print(f"    processed {i}/{len(todo)}")

    df = pd.DataFrame(rows)
    after_landmark = df[~((df["ioh_onset_min"].notna())
                          & (df["ioh_onset_min"] < config.VITAL_LANDMARK_MIN))]
    print(f"  no hypotension before the 30-min landmark      {len(after_landmark):>6}"
          f"   (reference {REFERENCE_COUNTS['after_landmark']})")
    risk = after_landmark[after_landmark["n_eval_windows"] >= 1]
    print(f"  >=1 evaluable 5-min window (main risk set)      {len(risk):>6}"
          f"   (reference {REFERENCE_COUNTS['risk_set']})")
    final = risk[risk["flat_fraction"] < 0.5]
    print(f"  after waveform artefact removal                 {len(final):>6}"
          f"   (reference {REFERENCE_COUNTS['final']})")
    print(f"  events                                          {int(final['event'].sum()):>6}"
          f"   (reference {REFERENCE_COUNTS['events']})")
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vitaldb-root", default=config.VITALDB_ROOT)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N candidates (smoke test)")
    ap.add_argument("--out", default=str(config.data_path("vitaldb_risk_set.csv")))
    args = ap.parse_args()
    if not args.vitaldb_root:
        raise SystemExit("set --vitaldb-root or APREPO_VITALDB_ROOT")

    print("=== VitalDB cascade ===")
    cohort = build_cascade(Path(args.vitaldb_root), limit=args.limit)
    cohort.to_csv(args.out, index=False)
    print(f"\nwritten: {args.out}  (N={len(cohort)}, events={int(cohort.event.sum())})")


if __name__ == "__main__":
    main()
