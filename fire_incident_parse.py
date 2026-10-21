"""
Fire Incidents Parser v5
MTF Wildfire Grid Protection Project

Run from project root: python parse_incidents.py
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

INCIDENTS_DIR = Path.cwd() / "fire_incidents"
OUTPUT_CSV    = Path.cwd() / "fire_incidents" / "incidents_clean.csv"

STANDARD_COLS = [
    "utility", "date", "time",
    "latitude", "longitude",
    "material_at_origin", "land_use_at_origin",
    "fire_size", "suppressed_by", "suppressing_agency",
    "facility_id", "other_companies",
    "voltage_v", "equipment_involved", "line_type",
    "was_outage", "outage_date", "outage_time",
    "suspected_initiating_event",
    "equipment_failure", "contact_from_object",
    "facility_contacted", "contributing_factor",
    "hftd_tier", "cpuc_reportable",
    "source_file",
]

ALIAS = {
    # ── Utility name appears in col 0 as merged cell → "Unnamed: 0"
    # in most files, or "Utility Name" in a few
    "unnamed: 0":                            "utility",
    "utility name":                          "utility",
    # ── Fire start date/time
    "date":                                  "date",
    "time":                                  "time",
    # ── Location
    "latitude":                              "latitude",
    "longitude":                             "longitude",
    # ── Fire details
    "material at origin":                    "material_at_origin",
    "material at origin - comments":         "material_at_origin",  # newer PGE files
    "land use at origin":                    "land_use_at_origin",
    "size":                                  "fire_size",
    "suppressed by":                         "suppressed_by",
    "suppressed \nby":                       "suppressed_by",
    "suppressing agency":                    "suppressing_agency",
    "suppressing \nagency":                  "suppressing_agency",
    # ── Facility
    "facility identification":               "facility_id",
    "facility identification/circuit":       "facility_id",
    "other companies":                       "other_companies",
    "voltage\n(volts)":                      "voltage_v",
    "voltage (volts)":                       "voltage_v",
    "voltage":                               "voltage_v",
    "equipment involved with ignition":      "equipment_involved",
    "type":                                  "line_type",
    # ── Outage
    "was there an outage":                   "was_outage",
    "date.1":                                "outage_date",
    "time.1":                                "outage_time",
    # ── Cause
    "suspected initiating event":            "suspected_initiating_event",
    "suspected ignition cause":              "suspected_initiating_event",
    "suspected cause":                       "suspected_initiating_event",
    "revised suspected initiating event":    "suspected_initiating_event",
    "equipment /facility \nfailure":         "equipment_failure",
    "equipment /facility failure":           "equipment_failure",
    "equipment/facility failure":            "equipment_failure",
    "revised equipment /facility failure":   "equipment_failure",
    "contact from \nobject":                 "contact_from_object",
    "contact from object":                   "contact_from_object",
    "facility \ncontacted":                  "facility_contacted",
    "facility contacted":                    "facility_contacted",
    "contributing factor":                   "contributing_factor",
    # ── HFTD / reportable
    "hftd":                                  "hftd_tier",
    "hftd.1":                                "hftd_tier",
    "revised tier":                          "hftd_tier",
    "hfra tier":                             "hftd_tier",
    "cpuc reportable":                       "cpuc_reportable",
}


def clean_col(name) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def normalize_utility(val: str) -> str:
    v = re.sub(r"[^a-z0-9]", "", str(val).strip().lower())
    if "pge" in v or "pacificgas" in v:
        return "PGE"
    if "sdge" in v or "sandiego" in v:
        return "SDGE"
    if v.startswith("sce") or "southerncalifornia" in v:
        return "SCE"
    return ""


def utility_from_filename(fname: str) -> str:
    """Fallback: infer utility from filename."""
    f = fname.lower()
    if f.startswith("pge") or "_pge" in f:
        return "PGE"
    if f.startswith("sdge") or "_sdge" in f:
        return "SDGE"
    if f.startswith("sce") or "_sce" in f:
        return "SCE"
    return ""


def detect_header_row(path: Path) -> int:
    raw = pd.read_excel(path, header=None, nrows=4)
    for i in range(len(raw)):
        vals = raw.iloc[i].astype(str).str.strip().str.lower()
        if vals.isin(["latitude", "longitude"]).any():
            return i
    return 1


def read_file(path: Path) -> tuple[pd.DataFrame, int]:
    hrow = detect_header_row(path)
    df = pd.read_excel(path, header=hrow)
    df = df.dropna(how="all").reset_index(drop=True)

    def is_header_row(row):
        vals = row.astype(str).str.strip().str.lower()
        return vals.isin(["utility name", "date", "latitude"]).sum() >= 2

    df = df[~df.apply(is_header_row, axis=1)].reset_index(drop=True)
    return df, hrow


def standardize(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    rename = {}
    seen_std = set()
    for col in df.columns:
        key = clean_col(col)
        if key in ALIAS:
            std = ALIAS[key]
            if std not in seen_std:
                rename[col] = std
                seen_std.add(std)

    df = df.rename(columns=rename)
    df["source_file"] = source_file

    existing = [c for c in STANDARD_COLS if c in df.columns]
    missing  = [c for c in STANDARD_COLS if c not in df.columns]
    df = df[existing].copy()
    for c in missing:
        df[c] = np.nan

    return df[STANDARD_COLS]


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize utility; fall back to filename inference if empty
    df["utility"] = df["utility"].apply(normalize_utility)
    mask_empty = df["utility"] == ""
    if mask_empty.any():
        df.loc[mask_empty, "utility"] = df.loc[mask_empty, "source_file"].apply(
            utility_from_filename
        )

    # Drop rows that still have no recognizable utility
    df = df[df["utility"].isin(["PGE", "SCE", "SDGE"])].copy()

    # Date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year

    # Coordinates
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Voltage
    def parse_voltage(v):
        s = str(v).replace(",", "").strip()
        kv = re.search(r"(\d+(?:\.\d+)?)\s*kv", s, re.IGNORECASE)
        if kv:
            return float(kv.group(1)) * 1000
        nums = re.findall(r"\d+", s)
        return float(nums[-1]) if nums else np.nan

    df["voltage_v"] = df["voltage_v"].apply(parse_voltage)

    return df.reset_index(drop=True)


def summarize(df: pd.DataFrame):
    print(f"\n  Total clean records    : {len(df):,}")
    print(f"  Date range             : {df['date'].min().date()} – {df['date'].max().date()}")
    print(f"  Records with lat/lon   : {df['latitude'].notna().sum():,}  "
          f"({df['latitude'].notna().mean()*100:.1f}%)")

    lat_ok = df[df["latitude"].notna()]
    if not lat_ok.empty:
        print(f"  Lat range              : {lat_ok['latitude'].min():.4f} – {lat_ok['latitude'].max():.4f}")
        print(f"  Lon range              : {lat_ok['longitude'].min():.4f} – {lat_ok['longitude'].max():.4f}")

    print(f"\n  By utility:")
    print(df["utility"].value_counts().to_string())

    print(f"\n  By year:")
    print(df["year"].value_counts().sort_index().to_string())

    print(f"\n  2019–2022 inflation check:")
    for yr in [2019, 2020, 2021, 2022]:
        n = (df["year"] == yr).sum()
        print(f"    {yr}: {n:,}")

    print(f"\n  Fire size distribution (top 15):")
    print(df["fire_size"].value_counts().head(15).to_string())

    print(f"\n  Equipment involved (top 15):")
    print(df["equipment_involved"].value_counts().head(15).to_string())

    print(f"\n  Suspected initiating event (top 15):")
    print(df["suspected_initiating_event"].value_counts().head(15).to_string())

    print(f"\n  HFTD tier breakdown:")
    print(df["hftd_tier"].value_counts().to_string())

    print(f"\n  Line type breakdown:")
    print(df["line_type"].value_counts().to_string())

    print(f"\n  Null rates — key columns:")
    for col in ["date", "latitude", "longitude", "equipment_involved",
                "suspected_initiating_event", "fire_size", "voltage_v",
                "hftd_tier", "line_type"]:
        pct = df[col].isna().mean() * 100
        print(f"    {col:<35}: {pct:.1f}%")


if __name__ == "__main__":
    xlsx_files = sorted(INCIDENTS_DIR.glob("*.xlsx"))
    print(f"Found {len(xlsx_files)} xlsx files.\n")

    frames = []
    for f in xlsx_files:
        if f.name.startswith("~$"):
            continue
        try:
            raw, hrow = read_file(f)
            std = standardize(raw, f.name)
            lat_ok = std["latitude"].notna().sum()
            frames.append(std)
            print(f"  OK  header_row={hrow}  lat={lat_ok:>4}  {f.name:<65} {len(std):>4} rows")
        except Exception as e:
            print(f"  ERR {f.name}: {e}")

    combined = pd.concat(frames, ignore_index=True)
    combined = coerce_types(combined)

    print("\n" + "=" * 70)
    print("  COMBINED SUMMARY")
    print("=" * 70)
    summarize(combined)

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved → {OUTPUT_CSV}")
