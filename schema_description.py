"""
Data Exploration Script
MTF Wildfire Grid Protection Project

Covers:
  1. NLCD land cover raster (.tif)
  2. GridMET climate variables (.nc)
  3. Fire incidents (.xlsx) from PG&E, SCE, SDGE 2014-2024

Run from project root:
  python explore_data.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path.cwd()  # run from project root

NLCD_TIF      = BASE_DIR / "environmental_data" / "Annual_NLCD_LndCov_2024_CU_C1V1" / "Annual_NLCD_LndCov_2024_CU_C1V1.tif"

GRIDMET_DIR   = BASE_DIR / "environmental_data"
GRIDMET_VARS  = {
    "bi":   "Burning Index",
    "erc":  "Energy Release Component",
    "pr":   "Precipitation",
    "rmin": "Min Relative Humidity",
    "tmmx": "Max Temperature",
    "vs":   "Wind Speed",
}

INCIDENTS_DIR = BASE_DIR / "fire_incidents"
SAMPLE_YEAR   = 2020   # year used for GridMET deep-dive
# ──────────────────────────────────────────────────────────────────────────────


def sep(title=""):
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 1. NLCD RASTER
# ─────────────────────────────────────────────────────────────────────────────
def explore_nlcd(path: Path):
    sep("NLCD LAND COVER RASTER")

    try:
        import rasterio
        import rasterio.windows
    except ImportError:
        print("  rasterio not installed. Run: pip install rasterio")
        return

    if not path.exists():
        print(f"  FILE NOT FOUND: {path}")
        return

    with rasterio.open(path) as src:
        print(f"  File         : {path.name}")
        print(f"  Size on disk : {path.stat().st_size / 1e9:.2f} GB")
        print(f"  Driver       : {src.driver}")
        print(f"  Width x Height : {src.width:,} x {src.height:,} pixels")
        print(f"  Bands        : {src.count}")
        print(f"  Dtype        : {src.dtypes[0]}")
        print(f"  CRS          : {src.crs}")
        print(f"  Resolution   : {src.res[0]:.6f} x {src.res[1]:.6f} (map units)")
        print(f"  Bounds       : {src.bounds}")
        print(f"  NoData value : {src.nodata}")

        window = rasterio.windows.Window(0, 0, min(1000, src.width), min(1000, src.height))
        sample = src.read(1, window=window)
        nodata = src.nodata if src.nodata is not None else 0
        valid  = sample[sample != nodata]
        unique, counts = np.unique(valid, return_counts=True)

        nlcd_classes = {
            11: "Open Water",                     12: "Perennial Ice/Snow",
            21: "Developed, Open Space",          22: "Developed, Low Intensity",
            23: "Developed, Medium Intensity",    24: "Developed, High Intensity",
            31: "Barren Land",
            41: "Deciduous Forest",               42: "Evergreen Forest",
            43: "Mixed Forest",                   51: "Dwarf Scrub",
            52: "Shrub/Scrub",                    71: "Grassland/Herbaceous",
            72: "Sedge/Herbaceous",               73: "Lichens",
            74: "Moss",                           81: "Pasture/Hay",
            82: "Cultivated Crops",               90: "Woody Wetlands",
            95: "Emergent Herbaceous Wetlands",
        }

        fire_relevant = {41, 42, 43, 51, 52, 71, 72}
        total = counts.sum()

        print(f"\n  Sample window : {window.width}x{window.height} pixels  ({total:,} valid)")
        print(f"\n  {'Class':>6}  {'Label':<35}  {'Pixels':>10}  {'%':>7}  Fire Risk")
        print(f"  {'-'*6}  {'-'*35}  {'-'*10}  {'-'*7}  {'-'*9}")
        for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
            label = nlcd_classes.get(int(cls), "Unknown")
            flag  = "YES" if int(cls) in fire_relevant else ""
            print(f"  {int(cls):>6}  {label:<35}  {cnt:>10,}  {cnt/total*100:>6.2f}%  {flag}")

        fr_px = sum(c for cls, c in zip(unique, counts) if int(cls) in fire_relevant)
        print(f"\n  Fire-relevant pixels : {fr_px:,}  ({fr_px/total*100:.1f}% of sample)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. GRIDMET
# ─────────────────────────────────────────────────────────────────────────────
def explore_gridmet_catalog(gridmet_dir: Path, variables: dict):
    sep("GRIDMET - FILE CATALOG")

    try:
        import netCDF4 as nc
    except ImportError:
        print("  netCDF4 not installed. Run: pip install netCDF4")
        return

    rows = []
    for var, label in variables.items():
        var_dir = gridmet_dir / var
        if not var_dir.exists():
            print(f"  [{var}] directory not found: {var_dir}")
            continue

        files = sorted(var_dir.glob(f"{var}_*.nc"))
        if not files:
            print(f"  [{var}] no .nc files found")
            continue

        years    = [int(f.stem.split("_")[-1]) for f in files if f.stem.split("_")[-1].isdigit()]
        total_mb = sum(f.stat().st_size for f in files) / 1e6
        missing  = [y for y in range(1990, 2026) if y not in years]

        rows.append({
            "Var":         var,
            "Description": label,
            "Files":       len(files),
            "Years":       f"{min(years)}-{max(years)}" if years else "?",
            "Size (MB)":   round(total_mb, 1),
            "Missing":     str(missing) if missing else "none",
        })

    if rows:
        print(pd.DataFrame(rows).to_string(index=False))


def explore_gridmet_single(gridmet_dir: Path, var: str, label: str, year: int):
    sep(f"GRIDMET DEEP DIVE - {var.upper()} ({label})  [year={year}]")

    try:
        import netCDF4 as nc
    except ImportError:
        return

    path = gridmet_dir / var / f"{var}_{year}.nc"
    if not path.exists():
        print(f"  File not found: {path}")
        return

    with nc.Dataset(path) as ds:
        print(f"  File : {path.name}  ({path.stat().st_size/1e6:.1f} MB)")

        print(f"\n  Global Attributes:")
        for attr in ds.ncattrs():
            val = str(getattr(ds, attr))
            print(f"    {attr:<28}: {val[:80]}")

        print(f"\n  Dimensions:")
        for dname, dim in ds.dimensions.items():
            print(f"    {dname:<15}: {len(dim)}")

        print(f"\n  Variables:")
        for vname, v in ds.variables.items():
            units     = getattr(v, "units",     "-")
            long_name = getattr(v, "long_name", getattr(v, "description", "-"))
            print(f"    {vname:<22}: shape={str(v.shape):<20} dtype={v.dtype}  units={units}")
            print(f"    {'':22}  {long_name}")

        candidates = [v for v in ds.variables if v not in ("lat", "lon", "day", "time", "crs")]
        if candidates:
            main_var = candidates[0]
            data     = ds.variables[main_var][:]
            flat     = data.compressed() if hasattr(data, "compressed") else data.flatten()
            flat     = flat[np.isfinite(flat.astype(float))]

            print(f"\n  Statistics for '{main_var}' ({len(flat):,} valid cells, full year):")
            print(f"    Min     : {np.min(flat):.4f}")
            print(f"    P10     : {np.percentile(flat, 10):.4f}")
            print(f"    Median  : {np.median(flat):.4f}")
            print(f"    Mean    : {np.mean(flat):.4f}")
            print(f"    P90     : {np.percentile(flat, 90):.4f}")
            print(f"    P99     : {np.percentile(flat, 99):.4f}")
            print(f"    Max     : {np.max(flat):.4f}")
            print(f"    Std Dev : {np.std(flat):.4f}")

            if hasattr(data, "mask"):
                pct_miss = data.mask.sum() / data.size * 100
                print(f"    Missing : {data.mask.sum():,}  ({pct_miss:.2f}%)")

            spatial_mean = np.nanmean(data, axis=(1, 2))
            print(f"\n  Spatial-mean time series (first 15 of {len(spatial_mean)} timesteps):")
            for i, val in enumerate(spatial_mean[:15]):
                print(f"    t={i:>3}  {val:.4f}")


def explore_gridmet_trend(gridmet_dir: Path, var: str, label: str,
                          start: int = 1990, end: int = 2024):
    sep(f"GRIDMET ANNUAL TREND - {var.upper()} ({label})  [{start}-{end}]")

    try:
        import netCDF4 as nc
    except ImportError:
        return

    rows = []
    for year in range(start, end + 1):
        path = gridmet_dir / var / f"{var}_{year}.nc"
        if not path.exists():
            continue
        with nc.Dataset(path) as ds:
            candidates = [v for v in ds.variables if v not in ("lat", "lon", "day", "time", "crs")]
            if not candidates:
                continue
            data = ds.variables[candidates[0]][:]
            flat = data.compressed() if hasattr(data, "compressed") else data.flatten()
            flat = flat[np.isfinite(flat.astype(float))]
            rows.append({
                "Year": year,
                "Mean": round(float(np.mean(flat)),  3),
                "P90":  round(float(np.percentile(flat, 90)), 3),
                "Max":  round(float(np.max(flat)),   3),
            })

    if not rows:
        print("  No files found.")
        return

    df   = pd.DataFrame(rows)
    print(df.to_string(index=False))
    coef = np.polyfit(df["Year"], df["Mean"], 1)
    direction = "increasing" if coef[0] > 0 else "decreasing"
    print(f"\n  Linear trend (Mean): {coef[0]:+.4f} units/year  ({direction})")


# ─────────────────────────────────────────────────────────────────────────────
# 3. FIRE INCIDENTS
# ─────────────────────────────────────────────────────────────────────────────
def explore_fire_incidents(incidents_dir: Path):
    sep("FIRE INCIDENTS - CATALOG")

    try:
        import openpyxl
    except ImportError:
        print("  openpyxl not installed. Run: pip install openpyxl")
        return

    if not incidents_dir.exists():
        print(f"  Directory not found: {incidents_dir}")
        return

    xlsx_files = sorted(incidents_dir.glob("*.xlsx"))
    if not xlsx_files:
        print("  No .xlsx files found.")
        return

    print(f"  Found {len(xlsx_files)} files:\n")
    catalog_rows = []
    all_frames   = []

    for f in xlsx_files:
        try:
            full_df = pd.read_excel(f)
            print(f"  -- {f.name}")
            print(f"     Rows    : {len(full_df):,}")
            print(f"     Columns : {list(full_df.columns)}")
            print(f"     Dtypes  :")
            for col, dtype in full_df.dtypes.items():
                null_pct = full_df[col].isna().mean() * 100
                print(f"       {col:<45} {str(dtype):<12} null: {null_pct:.1f}%")
            print()

            full_df["_source_file"] = f.name
            all_frames.append(full_df)
            catalog_rows.append({
                "File":    f.name,
                "Rows":    len(full_df),
                "Columns": len(full_df.columns),
                "Size KB": round(f.stat().st_size / 1e3, 1),
            })

        except Exception as e:
            print(f"  -- {f.name}  ERROR: {e}")

    if not all_frames:
        return

    sep("FIRE INCIDENTS - COMBINED ANALYSIS")
    combined = pd.concat(all_frames, ignore_index=True)

    print(f"  Total records : {len(combined):,}")
    print(f"  Total columns : {len(combined.columns)}")

    # Utility / company column
    util_candidates = [c for c in combined.columns
                       if any(k in c.lower() for k in ["util", "company", "org", "sce", "pge", "sdge", "utility"])]
    if util_candidates:
        col = util_candidates[0]
        print(f"\n  Utility breakdown ('{col}'):")
        print(combined[col].value_counts().to_string())

    # Year / date column
    date_candidates = [c for c in combined.columns
                       if any(k in c.lower() for k in ["year", "date", "time"])]
    if date_candidates:
        col = date_candidates[0]
        print(f"\n  Records by year ('{col}'):")
        try:
            combined["_year"] = pd.to_datetime(combined[col], errors="coerce").dt.year
            print(combined["_year"].value_counts().sort_index().to_string())
            print(f"\n  2019-2022 comparison (2020-2021 inflation check):")
            for yr in [2019, 2020, 2021, 2022]:
                n = (combined["_year"] == yr).sum()
                print(f"    {yr}: {n:,}")
        except Exception as e:
            print(f"    Could not parse dates: {e}")

    # Cause / equipment columns
    cause_candidates = [c for c in combined.columns
                        if any(k in c.lower() for k in ["cause", "equip", "type", "ignit", "fire"])]
    for col in cause_candidates[:2]:
        print(f"\n  Value counts for '{col}':")
        print(combined[col].value_counts().head(20).to_string())

    # Coordinates
    lat_candidates = [c for c in combined.columns if "lat" in c.lower()]
    lon_candidates = [c for c in combined.columns if any(k in c.lower() for k in ["lon", "lng"])]
    if lat_candidates and lon_candidates:
        lat_col, lon_col = lat_candidates[0], lon_candidates[0]
        print(f"\n  Coordinate coverage ('{lat_col}', '{lon_col}'):")
        print(f"    Lat range : {combined[lat_col].min():.4f} - {combined[lat_col].max():.4f}")
        print(f"    Lon range : {combined[lon_col].min():.4f} - {combined[lon_col].max():.4f}")
        print(f"    Missing   : {combined[lat_col].isna().sum():,} lat,  {combined[lon_col].isna().sum():,} lon")
    else:
        print(f"\n  No lat/lon columns detected — check column names above.")

    sep("FIRE INCIDENTS - FILE SUMMARY TABLE")
    print(pd.DataFrame(catalog_rows).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    explore_nlcd(NLCD_TIF)

    explore_gridmet_catalog(GRIDMET_DIR, GRIDMET_VARS)

    for var, label in GRIDMET_VARS.items():
        explore_gridmet_single(GRIDMET_DIR, var, label, SAMPLE_YEAR)

    for var in ["bi", "erc", "vs", "rmin"]:
        explore_gridmet_trend(GRIDMET_DIR, var, GRIDMET_VARS[var])

    explore_fire_incidents(INCIDENTS_DIR)

    sep("DONE")
    print("  Dependencies: pip install rasterio netCDF4 numpy pandas openpyxl\n")