"""
Step 2: Clean incidents + join GridMET fire weather
MTF Wildfire Grid Protection Project

Reads:  fire_incidents/incidents_clean.csv
Reads:  environmental_data/{bi,erc,rmin,vs,tmmx,pr}/*.nc
Writes: fire_incidents/incidents_gridmet.csv

For each incident, extracts the GridMET values at the nearest grid cell
on the day of the incident for: BI, ERC, rmin, wind speed, tmax, precip.

Run from project root: python clean_and_join_gridmet.py
"""

import re
import numpy as np
import pandas as pd
import netCDF4 as nc
from pathlib import Path
from datetime import datetime, timedelta

INCIDENTS_CSV = Path.cwd() / "fire_incidents" / "incidents_clean.csv"
GRIDMET_DIR   = Path.cwd() / "environmental_data"
OUTPUT_CSV    = Path.cwd() / "fire_incidents" / "incidents_gridmet.csv"

# GridMET variable folder → (nc variable name, output column, scale factor)
GRIDMET_VARS = {
    "bi":   ("burning_index_g",           "bi",      1.0),
    "erc":  ("energy_release_component-g","erc",     1.0),
    "rmin": ("relative_humidity",         "rmin",    1.0),
    "vs":   ("wind_speed",                "wind_ms", 1.0),
    "tmmx": ("air_temperature",           "tmax_c",  1.0),   # K → C applied below
    "pr":   ("precipitation_amount",      "precip_mm",1.0),
}

# ── REFERENCE GridMET grid (load once from any file) ─────────────────────────
def load_grid(gridmet_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    sample = next((gridmet_dir / "bi").glob("bi_*.nc"))
    with nc.Dataset(sample) as ds:
        lats = ds.variables["lat"][:]
        lons = ds.variables["lon"][:]
    return np.array(lats), np.array(lons)


def nearest_idx(arr: np.ndarray, val: float) -> int:
    return int(np.argmin(np.abs(arr - val)))


# ── YEAR CACHE: keep one file open per variable per year ─────────────────────
_cache: dict = {}   # (var, year) → opened netCDF4.Dataset

def get_dataset(var: str, year: int, gridmet_dir: Path):
    key = (var, year)
    if key not in _cache:
        path = gridmet_dir / var / f"{var}_{year}.nc"
        if not path.exists():
            return None
        _cache[key] = nc.Dataset(path)
    return _cache[key]


def close_cache():
    for ds in _cache.values():
        try:
            ds.close()
        except Exception:
            pass
    _cache.clear()


def day_of_year(dt) -> int:
    """1-based day index within the year (matches GridMET day dimension)."""
    return (dt - datetime(dt.year, 1, 1)).days   # 0-based index into day dim


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: CLEAN INCIDENTS
# ─────────────────────────────────────────────────────────────────────────────
def clean_incidents(df: pd.DataFrame) -> pd.DataFrame:
    print("  Cleaning incidents...")

    # Drop the year=116 artifact
    df = df[df["year"] >= 2014].copy()

    # Fix positive longitudes (should be negative for California)
    pos_lon = df["longitude"] > 0
    if pos_lon.sum() > 0:
        print(f"    Flipping {pos_lon.sum()} positive longitudes to negative")
        df.loc[pos_lon, "longitude"] = -df.loc[pos_lon, "longitude"]

    # Drop records outside California bounding box
    ca_mask = (
        (df["latitude"]  >= 32.5) & (df["latitude"]  <= 42.0) &
        (df["longitude"] >= -124.5) & (df["longitude"] <= -114.0)
    )
    n_dropped = (~ca_mask).sum()
    if n_dropped:
        print(f"    Dropping {n_dropped} records outside California bbox")
    df = df[ca_mask].copy()

    # Standardize fire_size into clean buckets
    def bucket_size(s):
        s = str(s).strip().lower()
        if any(x in s for x in ["structure only"]):
            return "structure_only"
        if any(x in s for x in ["< 3 meter", "<3 meter", "1 meter", "3 meter - 0.25"]):
            return "under_0.25ac_small"
        if any(x in s for x in ["< 0.25", "<0.25", "less than .25", "less than three"]):
            return "under_0.25ac"
        if any(x in s for x in [".26", "0.26", "0.25 - 10", "26-9", "26 - 9"]):
            return "0.25_to_10ac"
        if any(x in s for x in ["10 - 99", "10 - 100", "10-99"]):
            return "10_to_100ac"
        if any(x in s for x in ["100", "300", "1000", "> 100", ">100"]):
            return "over_100ac"
        return "unknown"

    df["fire_size_bucket"] = df["fire_size"].apply(bucket_size)

    # Standardize suspected_initiating_event
    def bucket_cause(s):
        s = str(s).strip().lower()
        if "contact from object" in s or "contact from" in s:
            return "contact_from_object"
        if "equipment" in s and "failure" in s:
            return "equipment_failure"
        if "equipment" in s and "pg&e" in s:
            return "equipment_failure"
        if "vegetation" in s:
            return "vegetation"
        if "vandal" in s or "theft" in s:
            return "vandalism"
        if "wire" in s and "wire" in s:
            return "wire_wire_contact"
        if "animal" in s or "bird" in s:
            return "animal_contact"
        if "contaminat" in s:
            return "contamination"
        if "unknown" in s:
            return "unknown"
        if "3rd party" in s or "third party" in s:
            return "third_party"
        return "other"

    df["cause_bucket"] = df["suspected_initiating_event"].apply(bucket_cause)

    # Standardize line_type
    def bucket_line(s):
        s = str(s).strip().lower()
        if "overhead" in s or "overbuild" in s:
            return "overhead"
        if "underground" in s or "subsurface" in s:
            return "underground"
        if "pad" in s:
            return "padmounted"
        return "other"

    df["line_type_clean"] = df["line_type"].apply(bucket_line)

    # Flag PGE 2020-2021 inflation period
    df["inflation_flag"] = (
        (df["utility"] == "PGE") &
        (df["year"].isin([2020, 2021]))
    ).astype(int)

    # Is this a confirmed fire spread (not just a fault/near-miss)?
    confirmed = df["fire_size_bucket"].isin(["0.25_to_10ac", "10_to_100ac", "over_100ac"])
    df["confirmed_ignition"] = confirmed.astype(int)

    print(f"    Records after cleaning       : {len(df):,}")
    print(f"    Confirmed ignitions (≥0.25ac): {confirmed.sum():,}")
    print(f"    Inflation-flagged PGE rows   : {df['inflation_flag'].sum():,}")
    print(f"\n    Fire size bucket distribution:")
    print(df["fire_size_bucket"].value_counts().to_string())
    print(f"\n    Cause bucket distribution:")
    print(df["cause_bucket"].value_counts().to_string())

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: JOIN GRIDMET
# ─────────────────────────────────────────────────────────────────────────────
def join_gridmet(df: pd.DataFrame, gridmet_dir: Path) -> pd.DataFrame:
    print("\n  Loading GridMET grid coordinates...")
    lats, lons = load_grid(gridmet_dir)
    print(f"    Grid: {len(lats)} lats × {len(lons)} lons")

    # Pre-compute nearest grid indices for each unique lat/lon pair
    # (much faster than recomputing per row)
    print("  Computing nearest grid cell for each unique location...")
    unique_coords = df[["latitude", "longitude"]].dropna().drop_duplicates()
    coord_to_idx = {}
    for _, row in unique_coords.iterrows():
        li = nearest_idx(lats, row["latitude"])
        lj = nearest_idx(lons, row["longitude"])
        coord_to_idx[(row["latitude"], row["longitude"])] = (li, lj)
    print(f"    Unique locations: {len(coord_to_idx):,}")

    # Initialize output columns
    for _, (_, col, _) in GRIDMET_VARS.items():
        df[col] = np.nan

    print("  Extracting GridMET values per incident...")
    n_total  = len(df)
    n_ok     = 0
    n_nodate = 0
    n_nofile = 0

    # Group by year to minimize file opens
    df["_date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["_year_int"] = df["_date_dt"].dt.year.astype("Int64")

    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"    {idx:>5}/{n_total} processed...", end="\r")

        dt = row["_date_dt"]
        if pd.isna(dt):
            n_nodate += 1
            continue

        lat = row["latitude"]
        lon = row["longitude"]
        if pd.isna(lat) or pd.isna(lon):
            continue

        coord_key = (lat, lon)
        if coord_key not in coord_to_idx:
            # Find nearest on the fly
            li = nearest_idx(lats, lat)
            lj = nearest_idx(lons, lon)
            coord_to_idx[coord_key] = (li, lj)

        li, lj = coord_to_idx[coord_key]
        year = int(dt.year)
        doy  = day_of_year(dt.to_pydatetime())

        any_ok = False
        for var, (nc_var, out_col, scale) in GRIDMET_VARS.items():
            ds = get_dataset(var, year, gridmet_dir)
            if ds is None:
                n_nofile += 1
                continue
            try:
                val = float(ds.variables[nc_var][doy, li, lj])
                # Temperature: K → °C
                if out_col == "tmax_c":
                    val = val - 273.15
                if val > 1e30 or val < -1e10:   # masked/fill value
                    val = np.nan
                df.at[idx, out_col] = val
                any_ok = True
            except Exception:
                pass

        if any_ok:
            n_ok += 1

    print(f"\n    Joined successfully : {n_ok:,}")
    print(f"    No date             : {n_nodate:,}")
    print(f"    Missing GridMET file: {n_nofile:,}")

    # Drop temp columns
    df = df.drop(columns=["_date_dt", "_year_int"])

    close_cache()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: SUMMARY OF JOINED DATA
# ─────────────────────────────────────────────────────────────────────────────
def summarize_joined(df: pd.DataFrame):
    print("\n  GridMET column statistics:")
    gm_cols = ["bi", "erc", "rmin", "wind_ms", "tmax_c", "precip_mm"]
    for col in gm_cols:
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        if valid.empty:
            print(f"    {col:<12}: no data")
            continue
        print(f"    {col:<12}: n={len(valid):,}  "
              f"min={valid.min():.2f}  mean={valid.mean():.2f}  "
              f"p90={valid.quantile(0.9):.2f}  max={valid.max():.2f}")

    # Compare fire weather at confirmed ignitions vs non-confirmed
    confirmed = df[df["confirmed_ignition"] == 1]
    not_confirmed = df[df["confirmed_ignition"] == 0]
    print(f"\n  Fire weather comparison: confirmed ignitions vs faults")
    print(f"  {'Column':<12}  {'Confirmed mean':>16}  {'Fault mean':>12}")
    print(f"  {'-'*12}  {'-'*16}  {'-'*12}")
    for col in ["bi", "erc", "rmin", "wind_ms", "tmax_c"]:
        if col not in df.columns:
            continue
        cm = confirmed[col].mean()
        nm = not_confirmed[col].mean()
        print(f"  {col:<12}  {cm:>16.2f}  {nm:>12.2f}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 1: LOADING INCIDENTS")
    print("=" * 60)
    df = pd.read_csv(INCIDENTS_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    print(f"  Loaded {len(df):,} records")

    print("\n" + "=" * 60)
    print("  STEP 2: CLEANING")
    print("=" * 60)
    df = clean_incidents(df)

    print("\n" + "=" * 60)
    print("  STEP 3: GRIDMET JOIN")
    print("=" * 60)
    df = join_gridmet(df, GRIDMET_DIR)

    print("\n" + "=" * 60)
    print("  STEP 4: SUMMARY")
    print("=" * 60)
    summarize_joined(df)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved → {OUTPUT_CSV}")
    print(f"  Shape  : {df.shape}")
    print(f"  Columns: {list(df.columns)}")