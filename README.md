# Detecting arterial-waveform tau drift: analysis pipeline

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository contains the analysis code behind *"Detectability of arterial
pressure waveform-derived decay time constant drift in sepsis-associated
hypotension"* (submitted to *Briefings in Bioinformatics*).

The study asks a narrow question: the arterial pressure waveform carries a
Windkessel decay time constant (tau) whose drift over time has been proposed as a
haemodynamic signal, so **can that drift actually be detected at the cohort sizes
available to critical-care research?**

The answer is assembled from three pieces of evidence:

1. **Measurement** -- a free-`P_inf` Windkessel inversion recovers tau from real
   arterial waveforms. On synthetic waveforms with 1 mmHg noise the recovered
   value stays within 5 % of the true value for tau up to about 0.35 s, which
   covers the range this study reports (`06_figures_and_tables/figure_s1_synthetic.py`
   regenerates the recovery experiment).
2. **Association** -- tau drift is compared between stays that did and did not
   develop vasopressor-requiring sepsis, and the drift is also tested as a
   predictor on top of a conventional EHR model.
3. **Detection limit** -- an empirical minimal detectable effect (MDE) is
   estimated from each cohort's own slope distribution, so the study can state
   what effect size it *could* have found, not merely what it did find.

All three are reported for MIMIC-III (indwelling arterial lines in septic ICU
stays) and VitalDB (intra-operative arterial lines in non-cardiac surgery).

## What is reproducible here

Running the scripts in order regenerates **every model, simulation, figure and
table in the manuscript** from the de-identified tables in `data/`:

| Quantity | Where it comes from |
|---|---|
| Cohort counts 1,189 -> 480 -> 441 -> 319 | `01_data_processing/mimic_cohort.py` |
| Table 2 AUC / Brier / calibration | `03_prediction_models/mimic_lightgbm.py` |
| Table 3 ridge, NEWS2, DCA, positive controls | `03_prediction_models/*` |
| +1.91 %/h slope difference and its four sensitivities | `05_statistical_analysis/slope_sensitivity.py` |
| MDE 10.0 %/h and the power curve | `04_mde_simulation/empirical_mde.py`, `power_matrix.py` |
| Method-level MDE matrix (Figure 3) | `04_mde_simulation/method_level_mde.py` |
| VitalDB M1/M2/M3 on the 1,043-case prediction cohort | `03_prediction_models/vitaldb_m1m2m3.py` |
| VitalDB detection limit 51.0 %/h and its short/long split | `04_mde_simulation/vitaldb_mde.py` |
| VitalDB trajectory R² share (Figure 5) | `06_figures_and_tables/figure5_vitaldb.py` |
| Waveform gain share | `05_statistical_analysis/feature_importance.py` |
| Figures 1-5 and Figure S1 | `06_figures_and_tables/*` |

Raw waveform reprocessing (500 Hz -> beats -> tau) is also implemented
(`02_waveform_modeling/`), and needs the source waveforms described under
*Data availability*. It is not required for any of the numbers above: those run
from the shipped, de-identified feature tables.

### Reported values, as recomputed here

| Quantity | Manuscript | This repository |
|---|---|---|
| MIMIC-III A / B / C AUC | 0.656 / 0.650 / 0.483 | 0.6564 / 0.6504 / 0.4829 |
| MIMIC-III ΔAUC (B − A) | −0.006 | −0.0061 [−0.0335, +0.0235] |
| MIMIC-III MDE | 10.0 %/h [6.5, 14.5] | 9.5 %/h [8.0, 10.5] on the 0.5 %/h grid (9.2 on a 0.25 grid) |
| VitalDB M1 / M2 / M3 AUC | 0.600 / 0.6816 / 0.6747 | 0.6003 / 0.6727 / 0.6728 |
| VitalDB ΔAUC (M3 − M2) | −0.0069 | +0.0000 [−0.0155, +0.0149] |
| VitalDB MDE | 51.0 %/h [45.6, 56.2] | 51.0 %/h [48.0, 57.0] (0.85 %/min × 60) |
| VitalDB R² < 0.1 share | 36.7 % | 383/1,043 = 36.7 % |
| Waveform gain share | 22.3 % | 22.6 % |

The M2 / M3 pair sits within 0.01 of the published values, the difference being
attributable to LightGBM version behaviour (floating-point precision and split
thresholds); ΔAUC ≈ 0 is unaffected either way. The VitalDB detection limit is the
manuscript's value: `tau_slope` is measured per minute (`theilslopes(taus, peak_times
/ 60)`, normalised by the median tau), and the reported limit is that same estimate
converted to %/h, so the shipped script prints both forms. Interval endpoints differ
slightly because they are bootstrap percentiles. Item-by-item detail, including the
rows where a quantity is defined on a different cohort, is in
[`outputs/RECOMPUTATION_LOG.md`](outputs/RECOMPUTATION_LOG.md).

## Repository structure

```
ap_detectability_pipeline/
├── README.md
├── requirements.txt
├── data/                          # de-identified analysis tables (see DATA_DICTIONARY.md)
│   ├── DATA_DICTIONARY.md
│   ├── mimic_cohort_480.csv
│   ├── mimic_slope_441.csv
│   ├── mimic_news2_319.csv
│   ├── mimic_mde_bootstrap_1000.csv
│   ├── mde_method_level_matrix.csv
│   ├── vitaldb_trajectories.csv.gz
│   ├── vitaldb_case_outcomes.csv
│   ├── vitaldb_window_slopes.csv
│   ├── vitaldb_case_features.csv
│   └── vitaldb_prediction_set_1043.csv   # 1,043-case prediction cohort
├── outputs/                       # every generated table, figure and log
│   └── RECOMPUTATION_LOG.md       # reported vs recomputed value, item by item
├── 01_data_processing/
│   ├── config.py                  # paths + all analysis constants
│   ├── export_derived_data.py     # builds data/ from the archived analysis objects
│   ├── abp_summary.py             # conventional ABP statistics (shared)
│   ├── mimic_cohort.py            # Sepsis-3 -> 4 h landmark -> 480/441/319
│   ├── mimic_ehr_features.py      # 13 EHR features, 0-4 h window
│   ├── vitaldb_cohort.py          # 6,388 -> risk set, IOH outcome
│   └── vitaldb_features.py        # case-level M1/M2/M3 feature table
├── 02_waveform_modeling/
│   ├── windkessel_lib.py          # simulation, peak detection, free-P_inf fit
│   ├── mimic_tau_extraction.py    # MIMIC segments -> 15-min tau windows
│   ├── vitaldb_tau_extraction.py  # VitalDB 500 Hz -> 1 s median filter -> 5-min windows
│   ├── slope_aggregation.py       # window -> segment -> patient slope
│   └── synthetic_validation.py    # Figure S1 recovery experiment
├── 03_prediction_models/
│   ├── metrics.py                 # AUC, Brier, calibration, DCA, bootstrap
│   ├── mimic_lightgbm.py          # Table 2: A / B / C feature blocks
│   ├── mimic_ridge.py             # Table 3: model-choice sensitivity
│   ├── mimic_news2.py             # Table 3: NEWS2 subset
│   ├── mimic_dca.py               # Table 3: decision curve + positive controls
│   └── vitaldb_m1m2m3.py          # nested clinical -> ABP -> waveform models
├── 04_mde_simulation/
│   ├── empirical_mde.py           # bootstrap MDE on the 0.5 %/h grid
│   ├── power_matrix.py            # power vs cohort size and effect size
│   ├── method_level_mde.py        # beat/window/patient x horizon matrix
│   └── vitaldb_mde.py             # VitalDB limit + short/long split
├── 05_statistical_analysis/
│   ├── slope_sensitivity.py       # +1.91 %/h under IQR and winsorisation
│   ├── feature_importance.py      # gain share (the 22.3 % quantity)
│   └── vitaldb_sensitivity.py     # MAP < 60 / 55 mmHg, slope tails
└── 06_figures_and_tables/
    ├── style.py                   # shared figure style
    ├── figure1_forest.py          # Delta AUC forest
    ├── figure2_power.py           # power curve, MDE vs observed effect
    ├── figure3_heatmap.py         # method-level MDE
    ├── figure4_slopes.py          # slope distribution, two panels
    ├── figure5_vitaldb.py         # tau drift density + trajectory R2
    ├── figure_s1_synthetic.py     # recovery experiment
    └── tables.py                  # manuscript_tables.xlsx
```

## Requirements

Python 3.10 or newer.

```
pip install -r requirements.txt
```

The analysis tables are already in `data/`, so no download is needed for the
reported results. Cloning the repository and running the pipeline end to end takes
about 20 minutes on a laptop; the MDE bootstrap dominates the run time.

## Usage

Set the working directory to the repository root -- all paths are relative
(`./data/`, `./outputs/`).

```bash
# 1. cohort construction (prints the reported counts)
python 01_data_processing/mimic_cohort.py --mode derived

# 2. prediction models (Tables 2 and 3)
python 03_prediction_models/mimic_lightgbm.py
python 03_prediction_models/mimic_ridge.py
python 03_prediction_models/mimic_news2.py
python 03_prediction_models/mimic_dca.py
python 03_prediction_models/vitaldb_m1m2m3.py

# 3. detection limit (Figures 2 and 3)
python 04_mde_simulation/empirical_mde.py
python 04_mde_simulation/power_matrix.py
python 04_mde_simulation/method_level_mde.py
python 04_mde_simulation/vitaldb_mde.py

# 4. statistical analysis (Table 4, Figure 5)
python 05_statistical_analysis/slope_sensitivity.py
python 05_statistical_analysis/feature_importance.py
python 05_statistical_analysis/vitaldb_sensitivity.py --slopes-only
python 05_statistical_analysis/vitaldb_sensitivity.py \
    --prediction data/vitaldb_prediction_set_1043.csv

# 5. figures and tables
python 06_figures_and_tables/figure1_forest.py
python 06_figures_and_tables/figure2_power.py
python 06_figures_and_tables/figure3_heatmap.py
python 06_figures_and_tables/figure4_slopes.py
python 06_figures_and_tables/figure5_vitaldb.py
python 06_figures_and_tables/figure_s1_synthetic.py
python 06_figures_and_tables/tables.py
```

Every script is standalone, prints what it computed, and writes a named CSV or
PNG into `outputs/`. Running one script twice gives the same result: all
randomness is seeded (`config.RANDOM_SEED = 42`).

### Rebuilding the analysis tables from source

Only needed if you want to go back to the raw waveforms. Point the scripts at
your own copies (nothing is downloaded automatically, and no path is hard-coded):

```bash
export APREPO_MIMIC_ROOT=/path/to/mimic-iii-clinical-database-1.4
export APREPO_MIMIC_WAVE_DIR=/path/to/mimic3wdb-matched
export APREPO_VITALDB_ROOT=/path/to/vitaldb
```

```bash
python 01_data_processing/mimic_ehr_features.py
python 02_waveform_modeling/mimic_tau_extraction.py
python 02_waveform_modeling/slope_aggregation.py --traj-dir outputs/mimic_trajectories --out data/mimic_patient_slopes.csv
python 01_data_processing/vitaldb_cohort.py
python 02_waveform_modeling/vitaldb_tau_extraction.py
python 01_data_processing/vitaldb_features.py --vitaldb-root /path/to/vitaldb
```

`vitaldb_features.py` needs `cases.csv` from VitalDB to read the pre-operative
block; the table it produces is already in `data/`, so the downstream scripts do
not depend on that source step.

## Data availability

The raw waveform recordings are not redistributable: MIMIC-III waveform data are
available to credentialed users through PhysioNet
(<https://physionet.org/content/mimic3wdb-matched/1.0/>), and VitalDB is available
from its own repository (<https://vitaldb.net>). Both sources are governed by
data-use agreements that prohibit redistribution, and the arterial waveforms are
individually identifying physiological recordings, so they cannot be attached to a
public code repository.

To make the analysis checkable without those restrictions, `data/` contains the
**de-identified analysis tables** that the pipeline consumes: cohort membership,
derived waveform features (tau, tau-proxy slope, trajectory fit metric), routine
ABP summaries, the 13 EHR features, out-of-fold model predictions, the 1,043-case
prediction cohort, and the per-window slope tables. These tables carry only the
pseudonymous case and subject numbers assigned by the source databases, plus derived
numeric features; no name, date, free-text field, or waveform sample is included,
and the case identifiers are not linkable to a hospital record without the source
database's own access agreement. Every model, simulation, figure and table in the
manuscript runs from these tables with the commands above; where a reported
quantity is defined on a cohort other than the one shipped here,
`outputs/RECOMPUTATION_LOG.md` says so on the line where the two values differ.

Column-by-column definitions, units and provenance are in
[`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md). Reported and recomputed
values are listed side by side, item by item, in
[`outputs/RECOMPUTATION_LOG.md`](outputs/RECOMPUTATION_LOG.md).

## Licence

MIT (see `LICENSE`). The code may be reused freely; the data tables in `data/`
carry the terms of their source databases.

## Citation

If you use this pipeline, please cite the manuscript and the two source databases
(MIMIC-III and VitalDB).
