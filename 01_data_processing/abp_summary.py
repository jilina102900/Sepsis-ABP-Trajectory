# -*- coding: utf-8 -*-
"""Conventional arterial-pressure statistics shared by both cohorts.

These are the "routine ABP" features that must be available for the comparison
to be fair: the augmentation in model M2 / EHR+ABP is judged against a baseline
that already contains every conventional pressure summary.  Keeping this in one
place guarantees both cohorts compute them identically.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

SUMMARY_COLS = ["MAP_mean", "MAP_min", "SBP_mean", "DBP_mean", "HR_mean", "PP_mean"]


def conventional_abp(window: pd.DataFrame) -> dict:
    """Mean/min MAP plus mean SBP, DBP, HR and pulse pressure for one block."""
    if window.empty:
        return {c: np.nan for c in SUMMARY_COLS}
    return {
        "MAP_mean": float(np.nanmean(window["MAP"])) if "MAP" in window else np.nan,
        "MAP_min": float(np.nanmin(window["MAP"])) if "MAP" in window else np.nan,
        "SBP_mean": float(np.nanmean(window["SBP"])) if "SBP" in window else np.nan,
        "DBP_mean": float(np.nanmean(window["DBP"])) if "DBP" in window else np.nan,
        "HR_mean": float(np.nanmean(window["HR"])) if "HR" in window else np.nan,
        "PP_mean": float(np.nanmean(window["PP"])) if "PP" in window else np.nan,
    }


def block_statistics(traj: pd.DataFrame, start_min: float, end_min: float,
                     quality: str = "ok") -> dict:
    """Conventional statistics over ``[start_min, end_min)`` of a trajectory."""
    sel = traj[(traj["quality"] == quality)
               & (traj["time_min"] >= start_min)
               & (traj["time_min"] < end_min)]
    out = conventional_abp(sel)
    out["n_windows"] = int(len(sel))
    return out


def summarise_by_case(traj: pd.DataFrame, start_min: float, end_min: float) -> pd.DataFrame:
    """Apply :func:`block_statistics` to every case in a pooled trajectory table."""
    rows = []
    for caseid, grp in traj.groupby("caseid"):
        row = block_statistics(grp, start_min, end_min)
        row["caseid"] = caseid
        rows.append(row)
    cols: Sequence[str] = ["caseid", "n_windows"] + SUMMARY_COLS
    return pd.DataFrame(rows)[[c for c in cols if c in pd.DataFrame(rows).columns]]
