# -*- coding: utf-8 -*-
"""Global configuration and path resolution.

All scripts in this repository resolve inputs through :data:`DATA_DIR` and write
outputs through :data:`OUT_DIR`, so no absolute path is hard-coded anywhere.
By default both point to ``./data`` and ``./outputs`` at the repository root.

Optional environment overrides (used only when rebuilding from raw sources):

    APREPO_DATA_DIR   directory holding the de-identified analysis tables
    APREPO_OUT_DIR    directory where results are written

Waveform rebuilds additionally require the raw sources:

    APREPO_MIMIC_ROOT       MIMIC-III root (clinical tables + waveform segments)
    APREPO_MIMIC_WAVE_DIR   directory holding the ``<segment>.hea/.dat`` files
    APREPO_VITALDB_ROOT     VitalDB root (cases.csv, trks.csv, wav/, num/)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("APREPO_DATA_DIR", REPO_ROOT / "data")).resolve()
OUT_DIR = Path(os.environ.get("APREPO_OUT_DIR", REPO_ROOT / "outputs")).resolve()

MIMIC_ROOT = os.environ.get("APREPO_MIMIC_ROOT", "")
MIMIC_WAVE_DIR = os.environ.get("APREPO_MIMIC_WAVE_DIR", "")
VITALDB_ROOT = os.environ.get("APREPO_VITALDB_ROOT", "")

# ---------------------------------------------------------------------------
# Analysis constants reproduced from the manuscript (Methods)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# --- MIMIC-III ------------------------------------------------------------
# Tau extraction, free-P_inf model.
MIMIC_FIT_START_AFTER_PEAK_S = 0.15   # decay window opens 0.15 s after systolic peak
MIMIC_MIN_WINDOW_S = 0.15            # minimum usable decay length
MIMIC_MAX_NRMSE = 0.08              # beats above this normalised RMSE are rejected
MIMIC_WINDOW_MIN = 15                # 15-min rolling windows ...
MIMIC_STEP_MIN = 5                   # ... advancing every 5 min
MIMIC_LANDMARK_H = 4.0               # 4-h landmark for prediction

# --- VitalDB --------------------------------------------------------------
VITAL_MEDIAN_FILTER_S = 1            # 1-s median filter removes baseline drift
VITAL_TAU_MAX_S = 0.5                # fitted tau above this is treated as degenerate
VITAL_WINDOW_MIN = 5                 # 5-min estimation windows for the model
VITAL_IOH_MAP = 65                   # intra-operative hypotension threshold (mmHg)
VITAL_IOH_MIN_DURATION_MIN = 1       # sustained for at least 1 min
VITAL_LANDMARK_MIN = 30              # prediction landmark after anaesthesia start

# --- Detection limit ------------------------------------------------------
MDE_N_SIM = 2000                     # Monte-Carlo draws per group comparison
MDE_N_BOOT = 1000                    # outer patient-level bootstrap replicates
MDE_ALPHA = 0.05                     # two-sided
MDE_TARGET_POWER = 0.80
MDE_SHIFT_GRID_PCT_H = [i * 0.5 for i in range(0, 41)]   # 0 to 20 %/h
POWER_TOTAL_N = [300, 441, 600, 800, 1000, 1200, 1500, 1800]
POWER_SHIFTS_PCT_H = [5.0, 7.5, 10.0]
POWER_POS_RATIO = 0.30
METHOD_MDE_WINDOWS_MIN = [60, 120, 180, 240, 300]

# --- Feature blocks -------------------------------------------------------
MIMIC_EHR_COLS = [
    "age", "gender", "map_mean", "map_min", "hr_mean", "hr_max",
    "lactate_mean", "urine_4h", "mech_vent", "creatinine",
    "platelet", "potassium", "sodium",
]
MIMIC_ABP_COLS = [
    "tau_slope", "tau_proxy_slope", "last_tau", "last_tau_proxy", "PP_mean",
]

VITAL_CLIN_COLS = [
    "age", "sex", "bmi", "asa", "preop_htn", "preop_dm", "preop_hb",
    "preop_plt", "preop_na", "preop_k", "preop_gluc", "preop_alb",
    "preop_bun", "preop_cr",
]
VITAL_ABP_COLS = [
    "MAP_mean", "MAP_min", "SBP_mean", "DBP_mean", "HR_mean", "PP_mean",
]
VITAL_PHYS_COLS = [
    "tau_base_mean", "tau_last", "R_base_mean", "tau_slope_pct_h", "tau_fit_r2",
]

# Nested model blocks: M1 clinical -> M2 + routine ABP -> M3 + waveform.
VITAL_M1_COLS = VITAL_CLIN_COLS
VITAL_M2_COLS = VITAL_ABP_COLS
VITAL_M3_COLS = VITAL_PHYS_COLS

# --- VitalDB prediction set (1,043 cases) ---------------------------------
# The prediction arm of the manuscript is the landmark cohort that survived
# waveform quality control, not the full 3,199-case trajectory table above.
# Its shipped table already carries the derived binary covariates
# ``is_thoracic`` and the waveform summaries ``tau_level`` / ``r_level``.
VITAL_PRED_CLIN_COLS = ["age", "sex", "bmi", "asa", "emop", "is_thoracic"]
VITAL_PRED_ABP_COLS = [
    "map_mean", "map_min", "map_cv", "map_slope",
    "hr_mean", "pp_mean", "pp_cv", "pp_slope",
]
VITAL_PRED_PHYS_COLS = ["tau_level", "tau_slope", "tau_iqr_ratio", "r_level"]

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def data_path(name: str) -> Path:
    """Return an absolute path inside :data:`DATA_DIR`."""
    return DATA_DIR / name


def out_path(name: str) -> Path:
    """Return an absolute path inside :data:`OUT_DIR`, creating the folder."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / name


def require(path: Path) -> Path:
    """Fail loudly instead of silently continuing on a missing input."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Required input not found: {path}\n"
            f"See README.md (Data availability) and data/DATA_DICTIONARY.md "
            f"for how to obtain or regenerate this file."
        )
    return Path(path)
