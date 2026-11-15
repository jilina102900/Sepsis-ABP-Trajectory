# -*- coding: utf-8 -*-
"""Aggregate the 13 EHR baseline features over the 0-4 h landmark window.

Feature blocks (Methods, Table 2):

    demographics   age, gender
    haemodynamics  map_mean, map_min, hr_mean, hr_max       (arterial line)
    labs           lactate_mean, creatinine, platelet, potassium, sodium
    support        mech_vent, urine_4h

Item identifiers are resolved from ``D_ITEMS`` / ``D_LABITEMS`` of MIMIC-III v1.4
and are listed in :data:`ITEMIDS`; each entry was verified against the released
dictionary rather than assumed.

The percentile aggregation mirrors the archived analysis: mean (or min / max
where the column name says so) over every observation inside
``[intime, intime + LANDMARK_H]``.

Usage
-----
    python 01_data_processing/mimic_ehr_features.py \
        --mimic-root /path/to/mimic-iii-clinical-database-1.4 \
        --stays data/mimic_stays.csv --out data/mimic_ehr_features.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# Verified against D_ITEMS.csv.gz / D_LABITEMS.csv.gz (MIMIC-III v1.4)
ITEMIDS = {
    "abp_mean": 220052,       # Arterial Blood Pressure mean (mmHg)
    "hr": 220045,             # Heart Rate (bpm)
    "creatinine": 220615,     # Creatinine (Labs)
    "platelet": 227457,       # Platelet Count (Labs)
    "potassium": 1535,        # Potassium (Chemistry)
    "sodium": 1536,           # Sodium (Chemistry)
}
LAB_ITEMIDS = {"lactate": 50813}
MECH_VENT_ITEMID = 225792     # Invasive Ventilation (procedureevents_mv)

CHART_COLS = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM"]


def _load_stays(path: Path) -> pd.DataFrame:
    stays = pd.read_csv(path)
    cols = {c.lower(): c for c in stays.columns}
    ren = {cols[c]: c for c in cols if c in ("subject_id", "hadm_id", "icustay_id")}
    stays = stays.rename(columns=ren)
    if "intime" in stays.columns:
        stays["intime"] = pd.to_datetime(stays["intime"])
    return stays


def _charted(mimic_root: Path, stay_ids: set, itemids: dict,
             horizon_h: float = config.MIMIC_LANDMARK_H) -> pd.DataFrame:
    """Stream CHARTEVENTS and keep only the requested items / stays / window."""
    inv = {v: k for k, v in itemids.items()}
    chunks = []
    for chunk in pd.read_csv(mimic_root / "CHARTEVENTS.csv.gz", usecols=CHART_COLS,
                             chunksize=2_000_000,
                             dtype={"SUBJECT_ID": "int32", "ICUSTAY_ID": "int32"}):
        chunk = chunk[chunk["ITEMID"].isin(inv) & chunk["ICUSTAY_ID"].isin(stay_ids)]
        if chunk.empty:
            continue
        chunk["feature"] = chunk["ITEMID"].map(inv)
        chunks.append(chunk.drop(columns="ITEMID"))
    if not chunks:
        return pd.DataFrame(columns=["SUBJECT_ID", "ICUSTAY_ID", "CHARTTIME", "VALUENUM", "feature"])
    df = pd.concat(chunks, ignore_index=True)
    df["CHARTTIME"] = pd.to_datetime(df["CHARTTIME"], errors="coerce")
    return df


def build(mimic_root: Path, stays: pd.DataFrame,
          horizon_h: float = config.MIMIC_LANDMARK_H) -> pd.DataFrame:
    """Return one row per ICU stay with the 13 baseline features."""
    stay_ids = set(stays["icustay_id"].dropna().astype(int))
    intime = dict(zip(stays["icustay_id"].astype(int), pd.to_datetime(stays["intime"])))

    charted = _charted(mimic_root, stay_ids, ITEMIDS)
    charted = charted.dropna(subset=["VALUENUM"])
    charted["intime"] = charted["ICUSTAY_ID"].map(intime)
    charted["hours"] = (charted["CHARTTIME"] - charted["intime"]).dt.total_seconds() / 3600.0
    charted = charted[(charted["hours"] >= 0) & (charted["hours"] <= horizon_h)]

    g = charted.groupby(["ICUSTAY_ID", "feature"])["VALUENUM"]
    wide = g.agg(["mean", "min", "max"]).unstack("feature")
    wide.columns = [f"{feat}_{stat}" for stat, feat in wide.columns]

    out = pd.DataFrame({"icustay_id": sorted(stay_ids)})
    out["map_mean"] = wide.get("abp_mean_mean")
    out["map_min"] = wide.get("abp_mean_min")
    out["hr_mean"] = wide.get("hr_mean")
    out["hr_max"] = wide.get("hr_max")
    out["creatinine"] = wide.get("creatinine_mean")
    out["platelet"] = wide.get("platelet_mean")
    out["potassium"] = wide.get("potassium_mean")
    out["sodium"] = wide.get("sodium_mean")

    # ---- lactate (LABEVENTS, first 4 h) ----------------------------------
    lab = pd.read_csv(mimic_root / "LABEVENTS.csv.gz",
                      usecols=["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
                      parse_dates=["CHARTTIME"])
    lab = lab[lab["ITEMID"].isin(LAB_ITEMIDS.values())].dropna(subset=["VALUENUM"])
    lab = lab.merge(stays[["hadm_id", "icustay_id", "intime"]], on="hadm_id", how="inner")
    lab["hours"] = (lab["CHARTTIME"] - lab["intime"]).dt.total_seconds() / 3600.0
    lab = lab[(lab["hours"] >= 0) & (lab["hours"] <= horizon_h)]
    out = out.merge(lab.groupby("icustay_id")["VALUENUM"].mean().rename("lactate_mean"),
                    left_on="icustay_id", right_index=True, how="left")

    # ---- urine output, first 4 h -------------------------------------------
    uo = pd.read_csv(mimic_root / "OUTPUTEVENTS.csv.gz",
                     usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "VALUE"],
                     parse_dates=["CHARTTIME"])
    uo = uo.merge(stays[["icustay_id", "intime"]], on="icustay_id", how="inner")
    uo["hours"] = (uo["CHARTTIME"] - uo["intime"]).dt.total_seconds() / 3600.0
    uo = uo[(uo["hours"] >= 0) & (uo["hours"] <= horizon_h)]
    out = out.merge(uo.groupby("icustay_id")["VALUE"].sum().rename("urine_4h"),
                    left_on="icustay_id", right_index=True, how="left")

    # ---- invasive mechanical ventilation -----------------------------------
    mv = pd.read_csv(mimic_root / "PROCEDUREEVENTS_MV.csv.gz",
                     usecols=["ICUSTAY_ID", "ITEMID", "STARTTIME"],
                     parse_dates=["STARTTIME"])
    mv = mv[mv["ITEMID"] == MECH_VENT_ITEMID]
    vent = (mv.groupby("ICUSTAY_ID")["STARTTIME"].min().rename("vent_time").reset_index()
              .rename(columns={"ICUSTAY_ID": "icustay_id"}))
    out = out.merge(vent, on="icustay_id", how="left")
    out["mech_vent"] = (out["vent_time"].notna()).astype(int)
    out = out.drop(columns="vent_time")

    # ---- demographics -----------------------------------------------------
    pat = pd.read_csv(mimic_root / "PATIENTS.csv.gz", usecols=["SUBJECT_ID", "DOB", "GENDER"],
                     parse_dates=["DOB"])
    adm = pd.read_csv(mimic_root / "ADMISSIONS.csv.gz", usecols=["SUBJECT_ID", "ADMITTIME"],
                      parse_dates=["ADMITTIME"])
    demo = adm.merge(pat, on="SUBJECT_ID", how="left")
    demo["age"] = (demo["ADMITTIME"] - demo["DOB"]).dt.total_seconds() / 86400.0 / 365.25
    demo["age"] = demo["age"].clip(upper=300)          # MIMIC shifts ages > 89 to 300
    demo["gender"] = (demo["GENDER"] == "M").astype(int)
    demo = demo.groupby("SUBJECT_ID")[["age", "gender"]].first().reset_index()
    out = out.merge(stays[["icustay_id", "subject_id"]], on="icustay_id", how="left")
    out = out.merge(demo, left_on="subject_id", right_on="SUBJECT_ID", how="left").drop(
        columns=["SUBJECT_ID"])

    lead = ["icustay_id", "subject_id"] + config.MIMIC_EHR_COLS
    return out[[c for c in lead if c in out.columns]].sort_values("icustay_id")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mimic-root", default=config.MIMIC_ROOT)
    ap.add_argument("--stays", default=str(config.data_path("mimic_stays.csv")),
                    help="CSV with icustay_id, subject_id, hadm_id, intime")
    ap.add_argument("--out", default=str(config.data_path("mimic_ehr_features.csv")))
    args = ap.parse_args()
    if not args.mimic_root:
        raise SystemExit("set --mimic-root or the APREPO_MIMIC_ROOT environment variable")

    stays = _load_stays(config.require(Path(args.stays)))
    feats = build(Path(args.mimic_root), stays)
    feats.to_csv(args.out, index=False)
    print(f"stays: {len(feats)}")
    print(feats[config.MIMIC_EHR_COLS].isna().sum().to_string())
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
