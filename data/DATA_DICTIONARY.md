# Data dictionary

Every file in `data/` is a de-identified analysis table: it contains derived
quantities only, and the identifiers it carries are the pseudonymous case and
subject numbers assigned by the source databases, plus derived numeric features.
No name, date, free-text field or waveform sample is included, and nothing here is
a raw clinical record.

Units follow the manuscript throughout. Tau drift is **%/min** where the column is a
beat-level Theil-Sen slope against a minute time axis (`tau_slope`), and **%/h** for
the MIMIC slope columns, which are named `_pct_h`; tau is in **seconds**; pressures
in **mmHg**; time in **minutes** unless stated otherwise.

---

## `mimic_cohort_480.csv` — 480 ICU stays with a successful waveform inversion

One row per ICU stay. 145 stays (30.2 %) had vasopressor-requiring sepsis within
the 6-h outcome window.

| Column | Meaning |
|---|---|
| `subject_id` | de-identified patient index (a patient can appear once only; the 480 cohort is stay-level) |
| `icustay_id` | de-identified ICU stay index |
| `event_6h` | 1 if vasopressor-requiring sepsis within 6 h of the landmark |
| `age` | years at admission (ages above 89 are shifted by MIMIC and capped at 300) |
| `gender` | 1 = male |
| `map_mean`, `map_min` | arterial MAP, mean and minimum, 0–4 h |
| `hr_mean`, `hr_max` | heart rate, mean and maximum, 0–4 h |
| `lactate_mean` | serum lactate, mean, 0–4 h |
| `urine_4h` | cumulative urine output, 0–4 h (mL) |
| `mech_vent` | 1 if invasive mechanical ventilation started in 0–4 h |
| `creatinine`, `platelet`, `potassium`, `sodium` | laboratory values, first 4 h |
| `MAP_mean`, `HR_mean`, `PP_mean` | waveform-derived conventional summaries at the landmark |
| `tau_slope` | tau drift per hour, fitted on the tau trajectory |
| `tau_proxy_slope` | tau-proxy drift per hour, the quantity used in the primary slope comparison |
| `last_tau`, `last_tau_proxy` | tau and tau-proxy at the end of the landmark window |
| `pred_ehr` | out-of-fold predicted probability, 13-feature EHR model (Table 2, block A) |
| `pred_both` | out-of-fold predicted probability, 13 EHR + 5 ABP features (block B) |
| `pred_phys` | out-of-fold predicted probability, 5 ABP features only (block C) |

The out-of-fold predictions are stored so that the reported discrimination and
calibration can be recomputed without refitting; `mimic_lightgbm.py --retrain`
refits the same design from the feature columns.

---

## `mimic_slope_441.csv` — 441 stays with a ≥60 min continuous stable period

One row per stay; 132 events. Slope values are the primary input to the slope
comparison, the empirical MDE and Figure 4.

| Column | Meaning |
|---|---|
| `icustay_id`, `subject_id` | identifiers as above |
| `event_6h` | outcome flag |
| `tau_proxy_slope_pct_h` | tau-proxy drift, %/h — the primary comparison |
| `tau_slope_pct_h` | tau drift, %/h |
| `trajectory_fit_metric` | fit quality of the tau trajectory, used where the manuscript refers to a "trajectory-fit" index |

**Note on `trajectory_fit_metric`.** In the archived analysis this column was
labelled `R_slope`. It is a **slope** (range −5.14 to 2.26, 218 negative values),
not an R², even though the manuscript describes the corresponding feature as a
trajectory-fit R². The column has therefore been exported under a neutral name
(`trajectory_fit_metric`) so that the numeric values are preserved exactly while
the naming question is settled. Its distribution reproduces the reported
statement that 84.8 % of values fall below 0.1 with a median of 0.0066.

---

## `mimic_news2_319.csv` — 319 stays with a complete NEWS2 baseline

One row per stay; 77 events. Used only for the NEWS2 secondary analysis in
Table 3, where both models are **refitted inside this subset**.

| Column | Meaning |
|---|---|
| `icustay_id`, `subject_id`, `event_6h` | identifiers and outcome |
| `rr_mean` | respiratory rate, mean |
| `spo2_mean` | peripheral oxygen saturation, mean (%) |
| `temp_mean` | temperature, mean (°C) |
| `sbp_mean` | systolic pressure, mean |
| `hr_mean` | heart rate, mean |
| `gcs_mean` | Glasgow Coma Scale, mean |
| `news2` | aggregate NEWS2 score |

`news2` is a count score, not a probability, so no Brier score is defined for it.

---

## `mimic_mde_bootstrap_1000.csv` — bootstrap distribution of the MDE

| Column | Meaning |
|---|---|
| `mde_boot` | one MDE estimate per patient-level bootstrap replicate (1,000 rows) |

Values fall on the 0.5 %/h simulation grid. The point estimate and its interval
are read from this distribution; `empirical_mde.py` regenerates an equivalent
table.

---

## `mde_method_level_matrix.csv` — method-level detection limit

| Column | Meaning |
|---|---|
| `Cohort` | MIMIC or VitalDB |
| `Granularity` | `ICU-level` / `Case-level` (patient), `Segment-level` / `Track-level`, `Window-level` |
| `60 min` … `300 min` | MDE in %/h for that observation horizon |

Carried through verbatim; see `04_mde_simulation/method_level_mde.py` for why each
cell is horizon-specific.

---

## `vitaldb_trajectories.csv.gz` — 78,257 estimation windows, 3,199 cases

One row per estimation window. This is the window-level substrate behind the
VitalDB figures and the case-level aggregation.

| Column | Meaning |
|---|---|
| `caseid` | de-identified anaesthetic record index |
| `tid` | de-identified track index (the arterial waveform channel) |
| `time_min` | window start, minutes after the anaesthetic record began |
| `MAP`, `SBP`, `DBP`, `HR`, `PP` | window-level conventional pressures and pulse pressure |
| `tau` | Windkessel decay time constant of that window (s) |
| `R` | peripheral-resistance proxy, MAP/(HR·PP) |
| `quality` | `ok` when the window passed QC, otherwise the rejection reason |
| `hypo` | 1 if the window falls inside a sustained hypotension episode |
| `hypo_event` | 1 if that episode qualifies as the outcome |

`n_beats_accepted` is not present in the released window table, so
`vitaldb_case_features.csv` reports `n_beats_accepted_median` as empty rather than
substituting a synthetic value.

---

## `vitaldb_case_outcomes.csv` — one row per case

| Column | Meaning |
|---|---|
| `caseid` | identifier |
| `n_windows` | estimation windows available |
| `n_hypo_events` | number of sustained hypotension episodes |
| `min_MAP`, `mean_MAP` | pressures over the record |
| `n_ok` | windows passing QC |
| `has_hypotension` | 1 if any episode occurred |

3,199 cases, 696 with an episode (21.8 %).

---

## `vitaldb_window_slopes.csv` — 3,015 per-case array slopes

| Column | Meaning |
|---|---|
| `caseid`, `tid` | identifiers |
| `window_start` | start of the fitting horizon, seconds |
| `tau_slope` | slope of tau against time in that horizon (s per second) |
| `R_slope` | slope of the resistance proxy |
| `MAP_slope` | slope of MAP (%/h) |

---

## `vitaldb_case_features.csv` — the M1/M2/M3 model input

One row per case; 3,040 cases with at least one evaluable window.

**Physical and conventional blocks**

| Column | Meaning |
|---|---|
| `n_ok_windows`, `n_eval_windows` | windows passing QC / windows inside the 30-min baseline |
| `n_beats_accepted_median` | median accepted beats per window (empty in this release, see above) |
| `MAP_mean`, `MAP_min`, `SBP_mean`, `DBP_mean`, `HR_mean`, `PP_mean` | conventional ABP summaries of the 30-min baseline block (**M2 block**) |
| `tau_base_mean` | mean tau in the baseline block (**M3 block**) |
| `tau_last` | last tau of the record (**M3**) |
| `tau_slope_pct_h` | tau drift, %/h (**M3**) |
| `tau_slope_rel_pct_h` | tau drift normalised by the median tau level (numerically identical to `tau_slope_pct_h` after that normalisation; retained for traceability) |
| `tau_fit_r2` | R² of the tau-against-time fit (**M3**) — the quantity plotted in Figure 5, right panel |
| `R_base_mean` | mean resistance proxy in the baseline block (**M3**) |
| `R_slope_pct_h` | drift of the resistance proxy, %/h |

**Clinical block (M1)**

| Column | Meaning |
|---|---|
| `caseid`, `subjectid` | identifiers |
| `age` | years |
| `sex` | 1 = male |
| `bmi` | kg/m² |
| `asa` | ASA physical status |
| `preop_htn`, `preop_dm` | history of hypertension / diabetes |
| `preop_hb`, `preop_plt`, `preop_na`, `preop_k`, `preop_gluc`, `preop_alb`, `preop_bun`, `preop_cr` | pre-operative haemoglobin, platelet count, sodium, potassium, glucose, albumin, urea nitrogen, creatinine |

Only pre-operative variables enter M1: intra-operative cumulative quantities are
recorded after the prediction landmark and are excluded by construction.

**Outcome**

| Column | Meaning |
|---|---|
| `event` | 1 if any sustained hypotension episode occurred after the 30-min landmark |

---

## `vitaldb_prediction_set_1043.csv` — the prediction cohort

One row per case; **1,043 cases, 325 events (31.2 %)**. This is the cohort the
prediction arm of the manuscript is computed on: adult non-cardiac surgery with an
arterial line, a Solar8000 `ART_MBP` numeric stream and a `SNUADC/ART` waveform,
no sustained hypotension before the 30-min landmark, and at least 5 min of
evaluable pre-landmark data. It is the default input of
`03_prediction_models/vitaldb_m1m2m3.py`, `04_mde_simulation/vitaldb_mde.py` and
`06_figures_and_tables/figure5_vitaldb.py`.

The wider 3,040-case table above is a descriptive resource covering every case with
at least one evaluable window; it is not the model input, and the two cohorts are
never pooled.

| Column | Meaning |
|---|---|
| `caseid`, `subjectid` | identifiers (1,036 unique patients) |
| `first_ioh_in_pred` | outcome: 1 if the first sustained hypotension episode fell inside the 30-min prediction window |
| `pre_landmark_valid_win` | evaluable pre-landmark minutes; the short/long split is taken at its median (13 min) |
| `age`, `sex`, `bmi`, `asa`, `emop` | pre-operative clinical variables (**M1 block**) |
| `department`, `is_thoracic` | surgical specialty and its thoracic indicator (**M1 block**) |
| `map_mean`, `map_min`, `map_cv`, `map_slope` | MAP level, minimum, coefficient of variation and trend over the baseline (**M2 block**) |
| `hr_mean`, `pp_mean`, `pp_cv`, `pp_slope` | heart rate and pulse-pressure summaries of the baseline (**M2 block**) |
| `tau_level`, `tau_slope`, `tau_iqr_ratio`, `r_level` | mean decay constant, its relative drift (**%/min**), its dispersion and the mean resistance proxy (**M3 block**) |
| `tau_median`, `tau_slope_perbeat_ts`, `tau_r2_5min`, `r_proxy` | the underlying beat-level summaries behind `tau_level` / `tau_slope` / `r_level`, plus the R² of the tau-against-time fit |
| `n_beats_total`, `n_beats_acceptable`, `n_windows_5min` | extraction-volume counters |
| `quality_flag` | waveform QC verdict |

**Units.** `tau_slope` is a *relative* drift **per minute**, because the beat-level
Theil-Sen fit is run against a time axis in minutes:

```
pt      = peak_times / 60.0          # seconds -> minutes
s_beat  = theilslopes(taus, pt)[0] / tau_median * 100
```

A median |slope| of 3.20 %/min at the cohort's median tau of 0.249 s is an absolute
drift of 8.0 ms/min; sustained for an hour that is 0.48 s/h, and the manuscript's
detection limit is the same quantity expressed per hour (0.85 %/min = 51.0 %/h).
`04_mde_simulation/vitaldb_mde.py` runs the simulation on the column as measured and
multiplies by 60 only when reporting, so `vitaldb_mde.csv` carries both forms
(`mde_pct_min` and `mde_pct_h`). `tau_r2_5min` is defined only
for the 768 cases whose tau sequence spans at least two windows; the remaining 275
cases yield no trajectory fit, which is why Figure 5 counts "no computable fit" as
failing the R² < 0.1 threshold (108 + 275 = 383 of 1,043 = 36.7 %).

---

## Regenerating these tables

`01_data_processing/export_derived_data.py` writes every file above from the
archived analysis objects. The raw sources (`APREPO_MIMIC_ROOT`,
`APREPO_VITALDB_ROOT`) are only needed for the waveform-reprocessing scripts in
`02_waveform_modeling/`.
