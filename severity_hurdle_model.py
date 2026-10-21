"""
severity_hurdle_model.py
========================
Status quo vs recommended counterfactual expected-loss model using a hurdle
severity formulation.

Run from project root:
  python severity_hurdle_model.py
"""

from __future__ import annotations

import json
import logging
import math
import re
import zipfile
from pathlib import Path
from typing import Any

import fiona
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import average_precision_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Core constants
DAMAGE_PER_STRUCTURE_USD = 500_000.0
EXPOSURE_MULTIPLIER = 1.0
E_AREA_SMALL_DEFAULT = 5.0
LARGE_FIRE_THRESHOLD_AC = 10.0
NONBURNABLE_E_AREA_SMALL_AC = 2.0

TRAIN_YEARS = {2021, 2022, 2023}
TEST_YEAR = 2024

FRAP_YEAR_MIN = 1992
FRAP_YEAR_MAX = 2024
FRAP_MIN_STRATUM_COUNT = 30
FRAP_BOOTSTRAP_N = 500
FRAP_WINSOR_QUANTILE = 0.99

# Conservative utility-severity anchoring (public CPUC incident records).
UTILITY_SEVERITY_YEAR_MIN = 2021
UTILITY_MIN_STRATUM_COUNT = 20
UTILITY_PRIOR_STRENGTH = 40.0

RECOMMEND_THRESHOLDS = {
    "undergrounding_eal_usd": 2_600_000.0,
    "covered_conductor_eal_usd": 1_000_000.0,
    "hardening_eal_usd": 1_000_000.0,
    "vegetation_eal_usd": 200_000.0,
    "high_density_sqmi": 500.0,
}

# Cost / finance assumptions for annualized cost-effectiveness.
CAPITAL_DISCOUNT_RATE = 0.07
CAPITAL_LIFETIME_YEARS = 30

# Unit-cost bases (USD per circuit-mile unless noted).
# `internal_default` preserves historical assumptions for reproducibility.
# `cpuc_public_2025` uses public CPUC Rule 20 FAQ values for key capex levers.
COST_BASIS_DEFAULT = "cpuc_public_2025"
COST_ASSUMPTION_BASES = {
    "internal_default": {
        "source": "Project defaults (public program-level ranges)",
        "assumptions": {
            "undergrounding": {
                "capex_per_mile": {"low": 2_250_000.0, "mid": 3_300_000.0, "high": 4_500_000.0},
                "annual_opex_per_mile": {"low": 10_000.0, "mid": 20_000.0, "high": 40_000.0},
            },
            "covered_conductor": {
                "capex_per_mile": {"low": 649_000.0, "mid": 800_000.0, "high": 1_200_000.0},
                "annual_opex_per_mile": {"low": 15_000.0, "mid": 30_000.0, "high": 60_000.0},
            },
            "fast_trip_and_hardening": {
                # EPSS-like fast-trip baseline (lower capex than covered conductor).
                "capex_per_mile": {"low": 50_000.0, "mid": 100_000.0, "high": 200_000.0},
                "annual_opex_per_mile": {"low": 10_000.0, "mid": 20_000.0, "high": 40_000.0},
            },
            "vegetation_management": {
                "capex_per_mile": {"low": 0.0, "mid": 0.0, "high": 0.0},
                "annual_opex_per_mile": {"low": 20_000.0, "mid": 40_000.0, "high": 80_000.0},
            },
            "targeted_monitoring": {
                "capex_per_mile": {"low": 0.0, "mid": 0.0, "high": 0.0},
                "annual_opex_per_mile": {"low": 5_000.0, "mid": 15_000.0, "high": 40_000.0},
            },
        },
    },
    "cpuc_public_2025": {
        "source": "CPUC Rule 20 FAQ (public unit-cost examples for electric line undergrounding vs alternatives)",
        "source_url": "https://www.cpuc.ca.gov/industries-and-topics/electrical-energy/infrastructure/electric-rules/rule-20-overview/rule-20-faqs",
        "assumptions": {
            # CPUC FAQ cites median undergrounding near $3M/mile.
            "undergrounding": {
                "capex_per_mile": {"low": 2_250_000.0, "mid": 3_000_000.0, "high": 4_500_000.0},
                "annual_opex_per_mile": {"low": 10_000.0, "mid": 20_000.0, "high": 40_000.0},
            },
            # CPUC FAQ cites covered conductor + steel poles around $480k/mile.
            "covered_conductor": {
                "capex_per_mile": {"low": 400_000.0, "mid": 480_000.0, "high": 700_000.0},
                "annual_opex_per_mile": {"low": 15_000.0, "mid": 30_000.0, "high": 60_000.0},
            },
            # CPUC FAQ cites conventional reconductor around $300k/mile (used as hardening proxy).
            "fast_trip_and_hardening": {
                "capex_per_mile": {"low": 200_000.0, "mid": 300_000.0, "high": 450_000.0},
                "annual_opex_per_mile": {"low": 10_000.0, "mid": 20_000.0, "high": 40_000.0},
            },
            # Retain O&M ranges as scenario proxies where consistent statewide capex per-mile is not published.
            "vegetation_management": {
                "capex_per_mile": {"low": 0.0, "mid": 0.0, "high": 0.0},
                "annual_opex_per_mile": {"low": 20_000.0, "mid": 40_000.0, "high": 80_000.0},
            },
            "targeted_monitoring": {
                "capex_per_mile": {"low": 0.0, "mid": 0.0, "high": 0.0},
                "annual_opex_per_mile": {"low": 5_000.0, "mid": 15_000.0, "high": 40_000.0},
            },
        },
    },
}

# Warner et al. (2025, Nature Climate Change), Table 3 point estimates for
# ignition-frequency reduction (r_freq). We keep LOW/MID/HIGH as sensitivity:
# LOW = 50% of point estimate, MID = point estimate, HIGH = 115% capped at 1.0.
WARNER_EFFECT_SOURCE = "Warner et al. 2025, Nature Climate Change (Table 3)"
WARNER_R_FREQ_POINT = {
    "undergrounding": 1.00,  # Warner et al. (2025), Table 3
    "covered_conductor": 0.62,  # Warner et al. (2025), Table 3
    "fast_trip_and_hardening": 0.72,  # Warner et al. (2025), Table 3
    "vegetation_management": 0.57,  # Warner et al. (2025), Table 3
}


def _scenario_r_freq(point_estimate: float) -> dict[str, float]:
    pe = float(min(max(point_estimate, 0.0), 1.0))
    return {
        "low": 0.5 * pe,
        "mid": pe,
        "high": min(1.0, 1.15 * pe),
    }


EFFECTIVENESS_DEFAULTS = {
    "undergrounding": {
        "r_freq": _scenario_r_freq(WARNER_R_FREQ_POINT["undergrounding"]),
        "r_tail": {"low": 0.0, "mid": 0.0, "high": 0.0},
        "r_spread": {"low": 0.0, "mid": 0.0, "high": 0.0},
    },
    "covered_conductor": {
        "r_freq": _scenario_r_freq(WARNER_R_FREQ_POINT["covered_conductor"]),
        "r_tail": {"low": 0.0, "mid": 0.05, "high": 0.1},
        "r_spread": {"low": 0.0, "mid": 0.0, "high": 0.05},
    },
    "fast_trip_and_hardening": {
        "r_freq": _scenario_r_freq(WARNER_R_FREQ_POINT["fast_trip_and_hardening"]),
        "r_tail": {"low": 0.0, "mid": 0.0, "high": 0.05},
        "r_spread": {"low": 0.0, "mid": 0.0, "high": 0.05},
    },
    "vegetation_management": {
        "r_freq": _scenario_r_freq(WARNER_R_FREQ_POINT["vegetation_management"]),
        "r_tail": {"low": 0.1, "mid": 0.2, "high": 0.3},
        "r_spread": {"low": 0.1, "mid": 0.2, "high": 0.35},
    },
    "targeted_monitoring": {
        "r_freq": {"low": 0.0, "mid": 0.0, "high": 0.0},
        "r_tail": {"low": 0.05, "mid": 0.1, "high": 0.2},
        "r_spread": {"low": 0.05, "mid": 0.1, "high": 0.2},
    },
}

SCENARIOS = ("low", "mid", "high")
TOP_N_PLOT = 20

FRAP_ZIP_CANDIDATES = [
    HERE / "environmental_data" / "frap" / "lpf_2024.zip",
    HERE / "loss_function_data" / "lpf_2024.zip",
]
FRAP_GDB_CANDIDATES = [
    HERE / "loss_function_data" / "frap.gbd",
    HERE / "environmental_data" / "frap.gbd",
]
HFTD_CANDIDATES = [
    HERE / "Electrical_Grid" / "cpuc_hftd" / "CPUC_FireThreatMapTiers2_3_v3r_08_19_2021.gdb",
    HERE / "environmental_data" / "hftd.gdb",
]
LANDFIRE_FBFM40_CANDIDATES = [
    HERE / "environmental_data" / "CONUS" / "Tif" / "LF2024_FBFM40_CONUS.tif",
    HERE / "environmental_data" / "CONUS" / "Tif" / "conus_0k.tif",
]

# Vegetation applicability / scaling by fuel context.
VEG_FACTOR_BY_FUEL = {
    "timber_litter": 1.0,
    "timber_understory": 1.0,
    "shrub": 1.0,
    "grass_shrub": 1.0,
    "slash": 1.0,
    "grass": 0.5,
    "nonburnable": 0.0,
    "unknown": 0.5,
}

# Treated circuit miles per cluster used in cost sensitivity.
# These are physically grounded scenario assumptions replacing the former
# transmission-length-derived proxy that was dominated by a floor clip.
# LOW: targeted short segment treatment
# MID: moderate single high-risk feeder segment
# HIGH: broad treatment of multiple feeder segments
TREATED_MILES_BY_SCENARIO = {
    "low": 1.0,
    "mid": 2.0,
    "high": 5.0,
}


# Feature candidates available in both incidents and panel for severity modeling
FEATURE_NUMERIC_CANDIDATES = [
    "cluster_lat",
    "cluster_lon",
    "elevation_m",
    "slope_deg",
    "bi",
    "erc",
    "rmin",
    "wind_ms",
    "tmax_c",
    "precip_mm",
    "bi_7d_mean",
    "bi_30d_mean",
    "erc_14d_max",
    "precip_30d_sum",
    "days_since_rain_ge_5mm",
    "wind_x_slope",
    "hot_dry_index",
    "erc_x_wind",
]
FEATURE_CATEGORICAL_CANDIDATES = [
    "season",
    "fuel_category",
    "hftd_tier",
]


def _missing_error(name: str, attempted: list[Path], instructions: str) -> FileNotFoundError:
    lines = [f"Missing required input for {name}. Attempted paths:"]
    lines.extend([f"  - {p}" for p in attempted])
    lines.append("Next step:")
    lines.append(f"  {instructions}")
    return FileNotFoundError("\n".join(lines))


def _resolve_existing(name: str, attempted: list[Path], instructions: str) -> Path:
    for p in attempted:
        if p.exists():
            return p
    raise _missing_error(name, attempted, instructions)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _pick_col(cols: list[str], candidates: list[str]) -> str | None:
    lc = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lc:
            return lc[cand.lower()]
    return None


def _parse_date(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if isinstance(dt.dtype, pd.DatetimeTZDtype):
        dt = dt.dt.tz_convert(None)
    return dt.dt.tz_localize(None).dt.normalize()


def _canon_hftd_tier(v: Any) -> str:
    s = str(v).strip().lower().replace("-", "_").replace(" ", "_")
    if s in {"tier_3", "tier3", "3", "very_high", "vhigh"}:
        return "tier_3"
    if s in {"tier_2", "tier2", "2", "high"}:
        return "tier_2"
    if s in {"tier_1", "tier1", "1", "elevated", "moderate"}:
        return "tier_1"
    if s in {"non_hftd", "none", "nan", "na", ""}:
        return "non_hftd"
    # Conservative fallback for unknown coding.
    if "3" in s:
        return "tier_3"
    if "2" in s:
        return "tier_2"
    if "1" in s:
        return "tier_1"
    return "non_hftd"


def choose_frap_gdb(path_like: Path) -> Path:
    if path_like.suffix.lower() == ".gdb" and path_like.exists():
        return path_like
    if path_like.exists() and path_like.is_dir():
        nested = sorted(path_like.rglob("*.gdb"))
        if nested:
            return nested[0]
    raise FileNotFoundError(f"FRAP geodatabase not found under: {path_like}")


def _crf(rate: float, years: int) -> float:
    if rate <= 0:
        return 1.0 / max(years, 1)
    num = rate * (1.0 + rate) ** years
    den = ((1.0 + rate) ** years) - 1.0
    return float(num / den) if den != 0 else 0.0


def _norm_fuel(v: Any) -> str:
    s = str(v).strip().lower().replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z_]", "", s)
    return s if s else "unknown"


def _fuel_category_from_fbfm40(code: Any) -> str:
    try:
        c = int(code)
    except Exception:
        return "unknown"
    if c in (91, 92, 93, 98, 99):
        return "nonburnable"
    if 101 <= c <= 109:
        return "grass"
    if 121 <= c <= 124:
        return "grass_shrub"
    if 141 <= c <= 149:
        return "shrub"
    if 161 <= c <= 165:
        return "timber_understory"
    if 181 <= c <= 189:
        return "timber_litter"
    if 201 <= c <= 204:
        return "slash"
    return "unknown"


def _derive_fire_area_ac(df: pd.DataFrame) -> pd.Series:
    bucket_map = {
        "0.25_to_10ac": 5.0,
        "10_to_100ac": 55.0,
        "over_100ac": 200.0,
    }

    def from_bucket(v: Any) -> float:
        s = str(v).strip().lower()
        if s in bucket_map:
            return bucket_map[s]
        if "10" in s and "100" in s:
            return 55.0
        if "over" in s and "100" in s:
            return 200.0
        if "greater" in s and "100" in s:
            return 200.0
        if "0.25" in s and "10" in s:
            return 5.0
        if ".26" in s and "9.99" in s:
            return 5.0
        return float("nan")

    fire_size_col = _pick_col(list(df.columns), ["fire_size_ac", "fire_size_acres", "fire_size"])
    bucket_col = _pick_col(list(df.columns), ["fire_size_bucket"])

    # Prefer bucket midpoint mapping (existing methodology), then backfill from numeric fire_size.
    if bucket_col is not None:
        area = pd.to_numeric(df[bucket_col].map(from_bucket), errors="coerce")
    else:
        area = pd.Series(np.nan, index=df.index, dtype=float)

    if fire_size_col is not None:
        raw = df[fire_size_col].astype(str)
        num = pd.to_numeric(raw, errors="coerce")
        missing_num = num.isna()
        if missing_num.any():
            parsed = raw.str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
            num = num.fillna(pd.to_numeric(parsed, errors="coerce"))
        area = area.fillna(num)

    area = pd.to_numeric(area, errors="coerce")
    area[(area <= 0) | (area > 500_000)] = np.nan
    return area


def load_inputs() -> dict[str, Any]:
    panel_candidates = [
        OUTDIR / "panel_full.csv",
        OUTDIR / "panel_predictions_test.csv",
    ]
    incidents_candidates = [
        HERE / "fire_incidents" / "incidents_features.csv",
        HERE / "fire_incidents" / "incidents_gridmet.csv",
    ]
    struct_candidates = [
        OUTDIR / "structures_by_cluster.csv",
        OUTDIR / "severity_by_cluster.csv",
    ]
    rec_candidates = [
        OUTDIR / "severity_by_cluster.csv",
    ]

    panel_path = _resolve_existing(
        "location-day panel",
        panel_candidates,
        "Run ignition_probability.py first to generate outputs/panel_full.csv with p_ignition_cal.",
    )
    incidents_path = None
    for p in incidents_candidates:
        if p.exists():
            incidents_path = p
            break
    struct_path = _resolve_existing(
        "cluster structure features",
        struct_candidates,
        "Generate outputs/structures_by_cluster.csv (or outputs/severity_by_cluster.csv with structures_per_sqmi).",
    )

    panel = _read_csv(panel_path)
    incidents = _read_csv(incidents_path) if incidents_path is not None else pd.DataFrame()
    struct_df = _read_csv(struct_path)

    p_col = _pick_col(list(panel.columns), ["p_ignition_cal", "p_ignition", "p_ignition_raw"])
    if p_col is None:
        raise ValueError(f"{panel_path} must contain one of: p_ignition_cal, p_ignition, p_ignition_raw")
    if p_col != "p_ignition_cal":
        panel["p_ignition_cal"] = pd.to_numeric(panel[p_col], errors="coerce")

    panel["date"] = _parse_date(panel["date"])
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    panel = panel[panel["cluster_id"].notna() & panel["date"].notna()].copy()
    panel["cluster_id"] = pd.to_numeric(panel["cluster_id"], errors="coerce").astype("Int64")
    panel = panel[panel["cluster_id"].notna()].copy()
    panel["cluster_id"] = panel["cluster_id"].astype(int)
    panel["year"] = panel["year"].fillna(panel["date"].dt.year).astype(int)
    panel["p_ignition_cal"] = pd.to_numeric(panel["p_ignition_cal"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)

    if not incidents.empty:
        incidents["date"] = _parse_date(incidents["date"])
        incidents["year"] = pd.to_numeric(incidents.get("year"), errors="coerce")
        incidents["year"] = incidents["year"].fillna(incidents["date"].dt.year)
        incidents["year"] = incidents["year"].astype("Int64")
        incidents["confirmed_ignition"] = pd.to_numeric(incidents.get("confirmed_ignition"), errors="coerce").fillna(0).astype(int)
        if "cluster_id" in incidents.columns:
            incidents["cluster_id"] = pd.to_numeric(incidents["cluster_id"], errors="coerce").astype("Int64")

    struct_density_col = _pick_col(list(struct_df.columns), ["structures_per_sqmi", "structure_density_sqmi"])
    if struct_density_col is None:
        raise ValueError(f"{struct_path} must include structures_per_sqmi (or structure_density_sqmi).")
    if struct_density_col != "structures_per_sqmi":
        struct_df["structures_per_sqmi"] = pd.to_numeric(struct_df[struct_density_col], errors="coerce")
    struct_df["cluster_id"] = pd.to_numeric(struct_df["cluster_id"], errors="coerce").astype("Int64")
    struct_df = struct_df[struct_df["cluster_id"].notna()].copy()
    struct_df["cluster_id"] = struct_df["cluster_id"].astype(int)
    struct_df["structures_per_sqmi"] = pd.to_numeric(struct_df["structures_per_sqmi"], errors="coerce")

    rec_df = None
    for p in rec_candidates:
        if p.exists():
            tmp = _read_csv(p)
            if {"cluster_id", "recommended_mitigation"}.issubset(set(tmp.columns)):
                tmp["cluster_id"] = pd.to_numeric(tmp["cluster_id"], errors="coerce").astype("Int64")
                tmp = tmp[tmp["cluster_id"].notna()].copy()
                tmp["cluster_id"] = tmp["cluster_id"].astype(int)
                rec_df = tmp[["cluster_id", "recommended_mitigation"]].drop_duplicates("cluster_id")
                break

    detected = {
        "panel_path": str(panel_path),
        "incidents_path": str(incidents_path) if incidents_path is not None else None,
        "structures_path": str(struct_path),
        "panel_shape": (int(len(panel)), int(len(panel.columns))),
        "incidents_shape": (int(len(incidents)), int(len(incidents.columns))),
        "structures_shape": (int(len(struct_df)), int(len(struct_df.columns))),
        "panel_key_cols": [c for c in ["cluster_id", "date", "year", "p_ignition_cal"] if c in panel.columns],
        "incidents_key_cols": [c for c in ["cluster_id", "date", "year", "confirmed_ignition", "fire_size_bucket", "fire_size"] if c in incidents.columns],
        "structures_key_cols": [c for c in ["cluster_id", "structures_per_sqmi"] if c in struct_df.columns],
    }
    print("\nDetected inputs:")
    print(f"  panel      : {detected['panel_path']} shape={detected['panel_shape']} key_cols={detected['panel_key_cols']}")
    if incidents_path is not None:
        print(f"  incidents  : {detected['incidents_path']} shape={detected['incidents_shape']} key_cols={detected['incidents_key_cols']}")
    else:
        print("  incidents  : not found (not required for FRAP empirical severity lookup)")
    print(f"  structures : {detected['structures_path']} shape={detected['structures_shape']} key_cols={detected['structures_key_cols']}")
    if rec_df is not None:
        print(f"  recommendations source: {rec_candidates[0]}")

    return {"panel": panel, "incidents": incidents, "structures": struct_df, "existing_recs": rec_df, "detected": detected}


def build_centroids(panel: pd.DataFrame) -> pd.DataFrame:
    keep = ["cluster_id", "cluster_lat", "cluster_lon", "fuel_category", "hftd_tier", "elevation_m", "slope_deg"]
    cols = [c for c in keep if c in panel.columns]
    cent = panel[cols].drop_duplicates("cluster_id").copy()
    for c in keep:
        if c not in cent.columns:
            cent[c] = np.nan
    return cent.sort_values("cluster_id").reset_index(drop=True)


def assign_incident_clusters(inc: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    out = inc.copy()
    if "cluster_id" in out.columns and out["cluster_id"].notna().any():
        out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="coerce").astype("Int64")
    else:
        out["cluster_id"] = pd.Series([pd.NA] * len(out), dtype="Int64")

    need = out["cluster_id"].isna()
    lat_col = _pick_col(list(out.columns), ["latitude", "lat"])
    lon_col = _pick_col(list(out.columns), ["longitude", "lon", "lng"])
    if need.any() and (lat_col is not None) and (lon_col is not None):
        c_lat = centroids["cluster_lat"].to_numpy(float)
        c_lon = centroids["cluster_lon"].to_numpy(float)
        c_id = centroids["cluster_id"].to_numpy(int)
        rows = out.loc[need, [lat_col, lon_col]]
        assigned = []
        for lat, lon in rows.itertuples(index=False):
            if pd.isna(lat) or pd.isna(lon):
                assigned.append(np.nan)
                continue
            d2 = (c_lat - float(lat)) ** 2 + (c_lon - float(lon)) ** 2
            assigned.append(int(c_id[int(np.argmin(d2))]))
        out.loc[need, "cluster_id"] = pd.Series(assigned, index=rows.index, dtype="float")

    out = out[out["cluster_id"].notna()].copy()
    out["cluster_id"] = out["cluster_id"].astype(int)
    return out


def _read_frap_zip(zip_path: Path) -> gpd.GeoDataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        shp_files = [n for n in zf.namelist() if n.lower().endswith(".shp")]
    if not shp_files:
        raise FileNotFoundError(f"No .shp found inside FRAP zip: {zip_path}")
    shp_name = shp_files[0]
    return gpd.read_file(f"zip://{zip_path.resolve()}!{shp_name}")


def load_frap_perimeters(use_all_causes: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    attempted: list[Path] = []
    frap_raw: gpd.GeoDataFrame | None = None
    source_detail: dict[str, Any] = {}

    for zp in FRAP_ZIP_CANDIDATES:
        attempted.append(zp)
        if zp.exists() and zipfile.is_zipfile(zp):
            try:
                frap_raw = _read_frap_zip(zp)
                source_detail = {"type": "zip", "path": str(zp)}
                break
            except Exception as ex:
                log.warning("Failed reading FRAP zip %s (%s). Falling back.", zp, ex)

    if frap_raw is None:
        for gp in FRAP_GDB_CANDIDATES:
            attempted.append(gp)
            if not gp.exists():
                continue
            gdb = choose_frap_gdb(gp)
            layers = fiona.listlayers(gdb)
            if not layers:
                continue
            layer = layers[0]
            frap_raw = gpd.read_file(gdb, layer=layer)
            source_detail = {"type": "gdb", "path": str(gdb), "layer": layer}
            break

    if frap_raw is None:
        raise _missing_error(
            "FRAP perimeter data",
            attempted,
            "Add FRAP perimeter data at environmental_data/frap/lpf_2024.zip or loss_function_data/frap.gbd.",
        )

    if frap_raw.crs is None:
        frap_raw = frap_raw.set_crs(epsg=4326, allow_override=True)

    frap_raw = frap_raw[frap_raw.geometry.notna() & ~frap_raw.geometry.is_empty].copy()
    frap_raw["year"] = pd.to_numeric(frap_raw.get("YEAR_"), errors="coerce")
    if "CAUSE" in frap_raw.columns:
        frap_raw["cause"] = pd.to_numeric(frap_raw["CAUSE"], errors="coerce")
    else:
        frap_raw["cause"] = np.nan

    frap_3310 = frap_raw.to_crs(epsg=3310)
    frap_raw["gis_acres"] = pd.to_numeric(frap_raw.get("GIS_ACRES"), errors="coerce")
    frap_raw["gis_acres"] = frap_raw["gis_acres"].fillna(frap_3310.geometry.area / 4046.8564224)
    frap_raw["gis_acres"] = pd.to_numeric(frap_raw["gis_acres"], errors="coerce")

    base = frap_raw[
        frap_raw["year"].between(FRAP_YEAR_MIN, FRAP_YEAR_MAX, inclusive="both")
        & frap_raw["gis_acres"].notna()
        & (frap_raw["gis_acres"] > 0)
    ].copy()
    if not use_all_causes:
        base = base[base["cause"] == 7].copy()

    q99 = float(base["gis_acres"].quantile(FRAP_WINSOR_QUANTILE)) if len(base) else 50_000.0
    if not np.isfinite(q99) or q99 <= LARGE_FIRE_THRESHOLD_AC:
        q99 = 50_000.0
    base["gis_acres_winsor"] = pd.to_numeric(base["gis_acres"], errors="coerce").clip(upper=q99)
    base = base.reset_index(drop=True)
    base["frap_id"] = np.arange(len(base), dtype=int)
    base["is_large"] = (base["gis_acres"] >= LARGE_FIRE_THRESHOLD_AC).astype(int)

    # Prefer explicit ignition lat/lon if available; otherwise polygon centroid.
    lat_col = _pick_col(list(base.columns), ["ignition_lat", "latitude", "lat", "point_y"])
    lon_col = _pick_col(list(base.columns), ["ignition_lon", "longitude", "lon", "lng", "point_x"])
    if lat_col is not None and lon_col is not None:
        lat = pd.to_numeric(base[lat_col], errors="coerce")
        lon = pd.to_numeric(base[lon_col], errors="coerce")
        ok = lat.between(-90, 90) & lon.between(-180, 180)
    else:
        ok = pd.Series(False, index=base.index)

    cent = gpd.GeoSeries(base.to_crs(3310).geometry.centroid, crs=3310).to_crs(4326)
    fire_geom = gpd.GeoSeries(cent, index=base.index, crs=4326)
    if bool(ok.any()):
        fire_geom.loc[ok] = gpd.points_from_xy(lon.loc[ok], lat.loc[ok], crs=4326)
    out = gpd.GeoDataFrame(
        base[["frap_id", "year", "cause", "gis_acres", "gis_acres_winsor", "is_large"]].copy(),
        geometry=fire_geom,
        crs=4326,
    )
    out["latitude"] = out.geometry.y
    out["longitude"] = out.geometry.x

    meta = {
        "source": source_detail,
        "use_all_causes": bool(use_all_causes),
        "year_min": FRAP_YEAR_MIN,
        "year_max": FRAP_YEAR_MAX,
        "winsor_quantile": FRAP_WINSOR_QUANTILE,
        "winsor_cap_acres": float(q99),
        "n_fires_after_filters": int(len(out)),
        "n_with_ignition_points": int(ok.sum()),
        "e_area_small_anchor": float(E_AREA_SMALL_DEFAULT),
    }
    log.info("FRAP events loaded: %s rows | source=%s", f"{len(out):,}", source_detail)
    return out, meta


def _find_hftd_gdb() -> Path:
    for p in HFTD_CANDIDATES:
        if p.exists():
            return p
    raise _missing_error(
        "HFTD geodatabase",
        HFTD_CANDIDATES,
        "Add CPUC HFTD geodatabase at Electrical_Grid/cpuc_hftd/*.gdb or environmental_data/hftd.gdb.",
    )


def _pick_hftd_layer(gdb: Path) -> str:
    layers = fiona.listlayers(gdb)
    if not layers:
        raise RuntimeError(f"No layers found in HFTD geodatabase: {gdb}")
    preferred = [lyr for lyr in layers if ("FireThreat" in lyr) or ("HFTD" in lyr)]
    return preferred[0] if preferred else layers[0]


def _resolve_landfire_tif() -> Path:
    return _resolve_existing(
        "LANDFIRE FBFM40 raster",
        LANDFIRE_FBFM40_CANDIDATES,
        "Add LANDFIRE FBFM40 raster at environmental_data/CONUS/Tif/LF2024_FBFM40_CONUS.tif.",
    )


def _load_hftd_polygons_3310() -> gpd.GeoDataFrame:
    gdb = _find_hftd_gdb()
    layer = _pick_hftd_layer(gdb)
    hftd = gpd.read_file(gdb, layer=layer)
    if hftd.empty or "HFTD" not in hftd.columns:
        raise RuntimeError(f"HFTD layer missing data/field 'HFTD': {gdb} | layer={layer}")
    hftd = hftd[["HFTD", "geometry"]].copy()
    hftd = hftd[hftd.geometry.notna() & ~hftd.geometry.is_empty].copy()
    if hftd.crs is None:
        hftd = hftd.set_crs(epsg=3310, allow_override=True)
    return hftd.to_crs(epsg=3310)


def _assign_points_hftd_tier(points: gpd.GeoDataFrame, id_col: str) -> pd.DataFrame:
    hftd = _load_hftd_polygons_3310()
    pts = points.to_crs(epsg=3310)
    j = gpd.sjoin(pts[[id_col, "geometry"]], hftd, how="left", predicate="within")
    j["hftd_tier"] = j["HFTD"].map(_canon_hftd_tier).fillna("non_hftd")
    rank = {"non_hftd": 0, "tier_1": 1, "tier_2": 2, "tier_3": 3}
    j["rank"] = j["hftd_tier"].map(rank).fillna(0)
    top = j.sort_values([id_col, "rank"], ascending=[True, False]).drop_duplicates(id_col)
    return top[[id_col, "hftd_tier"]].copy()


def annotate_frap_with_hftd(frap_points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    top = _assign_points_hftd_tier(frap_points, "frap_id")
    out = frap_points.merge(top, on="frap_id", how="left")
    out["hftd_tier"] = out["hftd_tier"].fillna("non_hftd")
    return out


def annotate_frap_with_fuel(frap_points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    tif = _resolve_landfire_tif()
    out = frap_points.copy()
    with rasterio.open(tif) as ds:
        pts = out.to_crs(ds.crs)
        codes = np.full(len(pts), np.nan, dtype=float)
        valid_idx = pts.geometry.notna() & ~pts.geometry.is_empty
        idx = np.where(valid_idx.to_numpy())[0]
        coords = [(float(pts.geometry.iloc[i].x), float(pts.geometry.iloc[i].y)) for i in idx]
        nodata = ds.nodata
        for ii, val in zip(idx, ds.sample(coords)):
            v = float(val[0]) if len(val) else np.nan
            if not np.isfinite(v):
                continue
            iv = int(round(v))
            if (nodata is not None and iv == int(nodata)) or iv <= 0 or iv == 32767:
                continue
            codes[ii] = iv
    out["fuel_model_code"] = pd.Series(codes).astype("Int64")
    out["fuel_category"] = out["fuel_model_code"].map(_fuel_category_from_fbfm40).fillna("unknown")
    return out


def annotate_frap_with_strata(frap_points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = annotate_frap_with_hftd(frap_points)
    out = annotate_frap_with_fuel(out)
    out["hftd_tier"] = out["hftd_tier"].map(_canon_hftd_tier).fillna("non_hftd")
    out["fuel_category"] = out["fuel_category"].map(_norm_fuel).fillna("unknown")
    return out


def _stratum_stats(df: pd.DataFrame, e_small_anchor: float) -> dict[str, float]:
    n = int(len(df))
    if n == 0:
        return {
            "p_large": float("nan"),
            "e_area_large": float("nan"),
            "e_area_small": float(e_small_anchor),
            "e_area_total": float("nan"),
            "n_fires": 0,
        }
    raw = pd.to_numeric(df["gis_acres"], errors="coerce")
    win = pd.to_numeric(df.get("gis_acres_winsor"), errors="coerce")
    m = raw.notna() & win.notna()
    raw = raw[m]
    win = win[m]
    large_mask = raw >= LARGE_FIRE_THRESHOLD_AC
    p_large = float(large_mask.mean()) if len(raw) else float("nan")
    e_large = float(win[large_mask].mean()) if bool(large_mask.any()) else float("nan")
    e_small = float(e_small_anchor)
    e_total = float(p_large * e_large + (1.0 - p_large) * e_small) if np.isfinite(p_large) and np.isfinite(e_large) else float("nan")
    return {
        "p_large": p_large,
        "e_area_large": e_large,
        "e_area_small": e_small,
        "e_area_total": e_total,
        "n_fires": n,
    }


def _utility_stratum_stats(df: pd.DataFrame, global_p: float, global_e_large: float, e_small_anchor: float) -> dict[str, float]:
    n = int(len(df))
    if n <= 0:
        return {
            "p_large": float("nan"),
            "e_area_large": float("nan"),
            "e_area_small": float(e_small_anchor),
            "e_area_total": float("nan"),
            "n_fires": 0,
            "n_large": 0,
        }
    area = pd.to_numeric(df["fire_area_ac"], errors="coerce")
    area = area[area.notna() & (area > 0)]
    n = int(len(area))
    if n <= 0:
        return {
            "p_large": float("nan"),
            "e_area_large": float("nan"),
            "e_area_small": float(e_small_anchor),
            "e_area_total": float("nan"),
            "n_fires": 0,
            "n_large": 0,
        }
    is_large = area >= LARGE_FIRE_THRESHOLD_AC
    n_large = int(is_large.sum())
    k_large = float(n_large)
    p_large = (k_large + (UTILITY_PRIOR_STRENGTH * global_p)) / (n + UTILITY_PRIOR_STRENGTH)
    if n_large > 0:
        sum_large = float(area[is_large].sum())
        e_large = (sum_large + (UTILITY_PRIOR_STRENGTH * global_e_large)) / (n_large + UTILITY_PRIOR_STRENGTH)
    else:
        e_large = float(global_e_large)
    e_small = float(e_small_anchor)
    e_total = float(p_large * e_large + (1.0 - p_large) * e_small)
    return {
        "p_large": float(p_large),
        "e_area_large": float(e_large),
        "e_area_small": float(e_small),
        "e_area_total": float(e_total),
        "n_fires": int(n),
        "n_large": int(n_large),
    }


def build_utility_severity_priors(incidents: pd.DataFrame, e_small_anchor: float) -> tuple[dict[str, Any], dict[str, Any]]:
    if incidents is None or incidents.empty:
        return {
            "global": {},
            "direct": {},
            "fuel": {},
            "tier": {},
        }, {
            "n_confirmed_ignitions_used": 0,
            "n_with_geo_hftd_assignment": 0,
            "utility_large_rate_global": None,
            "utility_e_area_large_global": None,
        }

    inc = incidents.copy()
    inc["confirmed_ignition"] = pd.to_numeric(inc.get("confirmed_ignition"), errors="coerce").fillna(0).astype(int)
    inc["year"] = pd.to_numeric(inc.get("year"), errors="coerce")
    inc = inc[(inc["confirmed_ignition"] == 1) & (inc["year"] >= UTILITY_SEVERITY_YEAR_MIN)].copy()

    inc["fire_area_ac"] = _derive_fire_area_ac(inc)
    inc = inc[inc["fire_area_ac"].notna() & (inc["fire_area_ac"] > 0)].copy()
    if inc.empty:
        return {
            "global": {},
            "direct": {},
            "fuel": {},
            "tier": {},
        }, {
            "n_confirmed_ignitions_used": 0,
            "n_with_geo_hftd_assignment": 0,
            "utility_large_rate_global": None,
            "utility_e_area_large_global": None,
        }
    util_q99 = float(pd.to_numeric(inc["fire_area_ac"], errors="coerce").quantile(0.99))
    if np.isfinite(util_q99) and util_q99 > LARGE_FIRE_THRESHOLD_AC:
        inc["fire_area_ac"] = pd.to_numeric(inc["fire_area_ac"], errors="coerce").clip(upper=util_q99)

    if "fuel_category" in inc.columns:
        inc["fuel_category"] = inc["fuel_category"].map(_norm_fuel).fillna("unknown")
    elif "fuel_model_code" in inc.columns:
        inc["fuel_category"] = inc["fuel_model_code"].map(_fuel_category_from_fbfm40).fillna("unknown")
    else:
        inc["fuel_category"] = "unknown"

    hftd_raw = inc.get("hftd_tier")
    has_hftd_values = hftd_raw is not None and hftd_raw.notna().any()
    n_geo_hftd = 0
    if has_hftd_values:
        inc["hftd_tier"] = hftd_raw.map(_canon_hftd_tier).fillna("non_hftd")
    else:
        lat_col = _pick_col(list(inc.columns), ["latitude", "lat"])
        lon_col = _pick_col(list(inc.columns), ["longitude", "lon", "lng"])
        if lat_col is not None and lon_col is not None:
            pts = inc[[lat_col, lon_col]].copy()
            pts[lat_col] = pd.to_numeric(pts[lat_col], errors="coerce")
            pts[lon_col] = pd.to_numeric(pts[lon_col], errors="coerce")
            good = pts[lat_col].between(-90, 90) & pts[lon_col].between(-180, 180)
            inc["hftd_tier"] = "non_hftd"
            if bool(good.any()):
                g = gpd.GeoDataFrame(
                    inc.loc[good, ["fire_area_ac"]].copy().reset_index(),
                    geometry=gpd.points_from_xy(
                        pts.loc[good, lon_col],
                        pts.loc[good, lat_col],
                        crs=4326,
                    ),
                    crs=4326,
                )
                top = _assign_points_hftd_tier(g, "index")
                n_geo_hftd = int(len(top))
                hmap = top.set_index("index")["hftd_tier"]
                inc.loc[hmap.index, "hftd_tier"] = hmap
        else:
            inc["hftd_tier"] = "non_hftd"
    inc["hftd_tier"] = inc["hftd_tier"].map(_canon_hftd_tier).fillna("non_hftd")

    area = pd.to_numeric(inc["fire_area_ac"], errors="coerce")
    is_large = area >= LARGE_FIRE_THRESHOLD_AC
    global_p = float(is_large.mean()) if len(area) else 0.0
    global_large = area[is_large]
    global_e_large = float(global_large.mean()) if len(global_large) else 55.0

    global_stats = _utility_stratum_stats(inc, global_p, global_e_large, e_small_anchor)
    direct_stats = {
        (str(t), str(f)): _utility_stratum_stats(g, global_p, global_e_large, e_small_anchor)
        for (t, f), g in inc.groupby(["hftd_tier", "fuel_category"], dropna=False)
    }
    fuel_stats = {
        str(f): _utility_stratum_stats(g, global_p, global_e_large, e_small_anchor)
        for f, g in inc.groupby("fuel_category", dropna=False)
    }
    tier_stats = {
        str(t): _utility_stratum_stats(g, global_p, global_e_large, e_small_anchor)
        for t, g in inc.groupby("hftd_tier", dropna=False)
    }

    priors = {
        "global": global_stats,
        "direct": direct_stats,
        "fuel": fuel_stats,
        "tier": tier_stats,
    }
    stats = {
        "n_confirmed_ignitions_used": int(len(inc)),
        "n_with_geo_hftd_assignment": int(n_geo_hftd),
        "utility_large_rate_global": float(global_p),
        "utility_e_area_large_global": float(global_e_large),
        "utility_fire_area_winsor_q99": float(util_q99) if np.isfinite(util_q99) else None,
        "utility_year_min": int(UTILITY_SEVERITY_YEAR_MIN),
        "utility_min_stratum_count": int(UTILITY_MIN_STRATUM_COUNT),
        "utility_prior_strength": float(UTILITY_PRIOR_STRENGTH),
    }
    return priors, stats


def build_frap_severity_lookup(
    frap_with_strata: pd.DataFrame, cluster_static: pd.DataFrame, incidents: pd.DataFrame, e_small_anchor: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cluster_ref = cluster_static[["cluster_id", "fuel_category", "hftd_tier"]].drop_duplicates("cluster_id").copy()
    cluster_ref["fuel_category"] = cluster_ref["fuel_category"].map(_norm_fuel).fillna("unknown")
    cluster_ref["hftd_tier"] = cluster_ref["hftd_tier"].map(_canon_hftd_tier).fillna("non_hftd")

    fires = frap_with_strata.copy()
    fires["fuel_category"] = fires["fuel_category"].map(_norm_fuel).fillna("unknown")
    fires["hftd_tier"] = fires["hftd_tier"].map(_canon_hftd_tier).fillna("non_hftd")
    fires = fires[fires["gis_acres"].notna() & fires["gis_acres_winsor"].notna()].copy()

    frap_statewide = _stratum_stats(fires, e_small_anchor)
    frap_tier_stats = {
        t: _stratum_stats(g, e_small_anchor) for t, g in fires.groupby("hftd_tier", dropna=False)
    }
    frap_direct_stats = {
        (t, f): _stratum_stats(g, e_small_anchor)
        for (t, f), g in fires.groupby(["hftd_tier", "fuel_category"], dropna=False)
    }
    utility_priors, utility_prior_stats = build_utility_severity_priors(incidents, e_small_anchor)

    strata = cluster_ref[["hftd_tier", "fuel_category"]].drop_duplicates().sort_values(["hftd_tier", "fuel_category"])
    rows: list[dict[str, Any]] = []
    n_utility_direct = 0
    n_utility_fuel = 0
    n_utility_tier = 0
    n_utility_state = 0
    n_frap_direct = 0
    n_frap_tier_backfill = 0
    n_frap_state_backfill = 0

    for r in strata.itertuples(index=False):
        key = (str(r.hftd_tier), str(r.fuel_category))
        vals: dict[str, Any] | None = None
        level = ""
        n_fires = 0

        # Conservative primary anchor: confirmed utility ignition severity distribution.
        u_direct = utility_priors.get("direct", {}).get(key, {"n_fires": 0})
        if int(u_direct.get("n_fires", 0)) >= UTILITY_MIN_STRATUM_COUNT:
            vals = u_direct
            level = "utility_hftd_fuel"
            n_fires = int(u_direct.get("n_fires", 0))
            n_utility_direct += 1
        else:
            u_fuel = utility_priors.get("fuel", {}).get(key[1], {"n_fires": 0})
            if int(u_fuel.get("n_fires", 0)) >= UTILITY_MIN_STRATUM_COUNT:
                vals = u_fuel
                level = "utility_fuel_only"
                n_fires = int(u_fuel.get("n_fires", 0))
                n_utility_fuel += 1
            else:
                u_tier = utility_priors.get("tier", {}).get(key[0], {"n_fires": 0})
                if int(u_tier.get("n_fires", 0)) >= UTILITY_MIN_STRATUM_COUNT:
                    vals = u_tier
                    level = "utility_hftd_only"
                    n_fires = int(u_tier.get("n_fires", 0))
                    n_utility_tier += 1
                else:
                    u_state = utility_priors.get("global", {})
                    if int(u_state.get("n_fires", 0)) >= UTILITY_MIN_STRATUM_COUNT:
                        vals = u_state
                        level = "utility_statewide"
                        n_fires = int(u_state.get("n_fires", 0))
                        n_utility_state += 1

        # FRAP fallback only if utility samples are insufficient.
        if vals is None:
            d = frap_direct_stats.get(key, {"n_fires": 0})
            if int(d.get("n_fires", 0)) >= FRAP_MIN_STRATUM_COUNT:
                vals = frap_direct_stats[key]
                level = "frap_hftd_fuel"
                n_fires = int(d.get("n_fires", 0))
                n_frap_direct += 1
            else:
                tvals = frap_tier_stats.get(key[0], {"n_fires": 0})
                if int(tvals.get("n_fires", 0)) >= FRAP_MIN_STRATUM_COUNT:
                    vals = tvals
                    level = "frap_hftd_only"
                    n_fires = int(tvals.get("n_fires", 0))
                    n_frap_tier_backfill += 1
                else:
                    vals = frap_statewide
                    level = "frap_statewide"
                    n_fires = int(frap_statewide.get("n_fires", 0))
                    n_frap_state_backfill += 1

        p_large = float(vals["p_large"]) if np.isfinite(vals["p_large"]) else 0.15
        e_large = float(vals["e_area_large"]) if np.isfinite(vals["e_area_large"]) else max(55.0, LARGE_FIRE_THRESHOLD_AC)
        e_small = float(E_AREA_SMALL_DEFAULT)
        e_total = float(p_large * e_large + (1.0 - p_large) * e_small)

        if key[1] == "nonburnable":
            p_large = 0.0
            e_large = 0.0
            e_small = NONBURNABLE_E_AREA_SMALL_AC
            e_total = NONBURNABLE_E_AREA_SMALL_AC

        rows.append(
            {
                "hftd_tier": key[0],
                "fuel_category": key[1],
                "p_large": float(min(max(p_large, 0.0), 1.0)),
                "e_area_large": float(0.0 if key[1] == "nonburnable" else max(e_large, LARGE_FIRE_THRESHOLD_AC)),
                "e_area_small": float(max(e_small, 0.1)),
                "e_area_total": float(max(e_total, 0.1)),
                "n_fires": n_fires,
                "backfill_level": level,
            }
        )

    lookup = pd.DataFrame(rows).sort_values(["hftd_tier", "fuel_category"]).reset_index(drop=True)
    lookup.to_csv(OUTDIR / "frap_severity_lookup.csv", index=False)

    display_cols = ["hftd_tier", "fuel_category", "n_fires", "p_large", "e_area_large", "e_area_total"]
    printable = lookup[display_cols].copy()
    print("\nSeverity Lookup (utility anchored, FRAP fallback)")
    print("-" * 96)
    print(printable.to_string(index=False))
    too_large = printable[printable["e_area_total"] > 10_000]
    if not too_large.empty:
        log.warning(
            "One or more strata has e_area_total > 10,000 acres after winsorization:\n%s",
            too_large.to_string(index=False),
        )

    stats = {
        "n_fires_used_total": int(len(fires)),
        "n_lookup_rows": int(len(lookup)),
        "n_utility_hftd_fuel": int(n_utility_direct),
        "n_utility_fuel_only": int(n_utility_fuel),
        "n_utility_hftd_only": int(n_utility_tier),
        "n_utility_statewide": int(n_utility_state),
        "n_frap_hftd_fuel": int(n_frap_direct),
        "n_frap_hftd_only": int(n_frap_tier_backfill),
        "n_frap_statewide": int(n_frap_state_backfill),
        "utility_prior_stats": utility_prior_stats,
        "statewide_defaults": {
            "p_large": float(frap_statewide["p_large"]) if np.isfinite(frap_statewide["p_large"]) else 1.0,
            "e_area_large": float(frap_statewide["e_area_large"]) if np.isfinite(frap_statewide["e_area_large"]) else 55.0,
            "e_area_small": float(E_AREA_SMALL_DEFAULT),
            "e_area_total": float(frap_statewide["e_area_total"]) if np.isfinite(frap_statewide["e_area_total"]) else float("nan"),
            "n_fires": int(frap_statewide["n_fires"]),
        },
    }
    return lookup, stats


def predict_severity_components_from_lookup(
    panel: pd.DataFrame, lookup: pd.DataFrame, cluster_static: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cl = cluster_static[["cluster_id", "fuel_category", "hftd_tier"]].drop_duplicates("cluster_id").copy()
    cl["fuel_category"] = cl["fuel_category"].map(_norm_fuel).fillna("unknown")
    cl["hftd_tier"] = cl["hftd_tier"].map(_canon_hftd_tier).fillna("non_hftd")

    lut = lookup.copy()
    lut["fuel_category"] = lut["fuel_category"].map(_norm_fuel).fillna("unknown")
    lut["hftd_tier"] = lut["hftd_tier"].map(_canon_hftd_tier).fillna("non_hftd")

    statewide_row = lut.copy()
    statewide_defaults = {
        "p_large": float(pd.to_numeric(statewide_row["p_large"], errors="coerce").mean()),
        "e_area_large": float(pd.to_numeric(statewide_row["e_area_large"], errors="coerce").mean()),
        "e_area_small": float(pd.to_numeric(statewide_row["e_area_small"], errors="coerce").mean()),
    }

    cl = cl.merge(lut, on=["hftd_tier", "fuel_category"], how="left")
    cl["p_large"] = pd.to_numeric(cl["p_large"], errors="coerce").fillna(statewide_defaults["p_large"]).clip(0.0, 1.0)
    cl["e_area_large"] = pd.to_numeric(cl["e_area_large"], errors="coerce").fillna(statewide_defaults["e_area_large"])
    cl["e_area_small"] = pd.to_numeric(cl["e_area_small"], errors="coerce").fillna(statewide_defaults["e_area_small"])

    cl["P_large_base_raw"] = cl["p_large"].clip(0.0, 1.0)
    cl["E_area_large_base_raw"] = cl["e_area_large"].clip(lower=LARGE_FIRE_THRESHOLD_AC)
    cl["P_large_base"] = cl["P_large_base_raw"].copy()
    cl["E_area_large_base"] = cl["E_area_large_base_raw"].copy()
    cl["E_area_small"] = cl["e_area_small"].clip(lower=0.1)

    nonburn = cl["fuel_category"].map(_norm_fuel).eq("nonburnable")
    cl["plausibility_gated_row"] = False
    if bool(nonburn.any()):
        cl.loc[nonburn, "P_large_base"] = 0.0
        cl.loc[nonburn, "E_area_large_base"] = 0.0
        cl.loc[nonburn, "E_area_small"] = NONBURNABLE_E_AREA_SMALL_AC
        cl.loc[nonburn, "plausibility_gated_row"] = True

    cl["E_area_given_ign_base"] = cl["P_large_base"] * cl["E_area_large_base"] + (1.0 - cl["P_large_base"]) * cl["E_area_small"]

    keep = ["cluster_id", "date", "year", "p_ignition_cal"]
    for c in ["tx_len_km_10km"]:
        if c in panel.columns:
            keep.append(c)
    out = panel[keep].merge(
        cl[
            [
                "cluster_id",
                "fuel_category",
                "hftd_tier",
                "P_large_base_raw",
                "E_area_large_base_raw",
                "P_large_base",
                "E_area_large_base",
                "E_area_small",
                "E_area_given_ign_base",
                "plausibility_gated_row",
            ]
        ],
        on="cluster_id",
        how="left",
    )
    return out, cl


def _assign_recommendations(
    by_cluster: pd.DataFrame, existing_recs: pd.DataFrame | None
) -> pd.DataFrame:
    out = by_cluster.copy()
    # Sequential policy: fast-trip baseline for HFTD, then residual-risk escalation.
    # Existing recommendations are used only as fallback if required fields are missing.
    if "hftd_tier" in out.columns:
        out["is_hftd"] = out["hftd_tier"].astype(str).str.lower().ne("non_hftd")
    else:
        out["is_hftd"] = False

    fuel_norm = out["fuel_category"].map(_norm_fuel) if "fuel_category" in out.columns else pd.Series("unknown", index=out.index)
    veg_factor = fuel_norm.map(lambda f: VEG_FACTOR_BY_FUEL.get(f, VEG_FACTOR_BY_FUEL["unknown"]))
    veg_eligible = veg_factor > 0

    rf_fast_mid = EFFECTIVENESS_DEFAULTS["fast_trip_and_hardening"]["r_freq"]["mid"]
    rf_cov_mid = EFFECTIVENESS_DEFAULTS["covered_conductor"]["r_freq"]["mid"]
    rf_veg_mid = EFFECTIVENESS_DEFAULTS["vegetation_management"]["r_freq"]["mid"]

    rec = []
    decision_rule = []
    residual_after_fast = []
    for eal, dens, is_hftd, is_veg in zip(
        out["EAL_base_usd"].fillna(0.0),
        out["structures_per_sqmi"].fillna(0.0),
        out["is_hftd"].fillna(False),
        veg_eligible.fillna(False),
    ):
        residual = float(eal)
        if bool(is_hftd):
            residual *= (1.0 - rf_fast_mid)
        residual_after_fast.append(float(residual))

        if bool(is_hftd) and (residual > RECOMMEND_THRESHOLDS["covered_conductor_eal_usd"]):
            res_cov = residual * (1.0 - rf_cov_mid)
            if (res_cov > RECOMMEND_THRESHOLDS["undergrounding_eal_usd"]) and (dens > RECOMMEND_THRESHOLDS["high_density_sqmi"]):
                rec.append("undergrounding")
                decision_rule.append("hftd_extreme_residual_and_high_density")
            else:
                rec.append("covered_conductor")
                decision_rule.append("hftd_high_residual_escalate_covered_conductor")
        elif bool(is_veg) and (residual > RECOMMEND_THRESHOLDS["vegetation_eal_usd"]):
            # Moderate EL in vegetated fuels where fuel-contact mechanism is relevant.
            res_veg = residual * (1.0 - rf_veg_mid)
            if (res_veg > RECOMMEND_THRESHOLDS["hardening_eal_usd"]) and bool(is_hftd):
                rec.append("covered_conductor")
                decision_rule.append("vegetated_residual_high_in_hftd_escalate_covered_conductor")
            else:
                rec.append("vegetation_management")
                decision_rule.append("vegetated_moderate_residual_vegetation_management")
        elif bool(is_hftd):
            rec.append("fast_trip_and_hardening")
            decision_rule.append("hftd_baseline_fast_trip")
        else:
            rec.append("targeted_monitoring")
            decision_rule.append("non_hftd_low_residual_targeted_monitoring")

    out["recommendation"] = rec
    out["decision_rule"] = decision_rule
    out["residual_eal_after_fast_trip_mid"] = residual_after_fast

    # Optional fallback to existing recommendation if something is unexpectedly missing.
    if existing_recs is not None and not existing_recs.empty:
        ex = existing_recs.rename(columns={"recommended_mitigation": "recommended_existing"})
        out = out.merge(ex, on="cluster_id", how="left")
        out["recommendation"] = out["recommendation"].fillna(out["recommended_existing"]).fillna("targeted_monitoring")
        out = out.drop(columns=["recommended_existing"], errors="ignore")

    # Hard gating: no vegetation recommendation in nonburnable.
    nonburn = fuel_norm == "nonburnable"
    mask = nonburn & (out["recommendation"] == "vegetation_management")
    out.loc[mask, "recommendation"] = "targeted_monitoring"
    out.loc[mask, "decision_rule"] = "nonburnable_fuel_gate_override_to_monitoring"
    return out


def _scenario_reduction(action: str, key: str, scenario: str, fuel_category: Any | None = None) -> float:
    action = str(action)
    if action not in EFFECTIVENESS_DEFAULTS:
        action = "targeted_monitoring"
    base = float(EFFECTIVENESS_DEFAULTS[action][key][scenario])
    if action == "vegetation_management":
        vf = VEG_FACTOR_BY_FUEL.get(_norm_fuel(fuel_category), VEG_FACTOR_BY_FUEL["unknown"])
        base = base * float(vf)
    return float(min(max(base, 0.0), 1.0))


def _get_cost_basis(cost_basis: str) -> dict[str, Any]:
    if cost_basis not in COST_ASSUMPTION_BASES:
        valid = ", ".join(sorted(COST_ASSUMPTION_BASES.keys()))
        raise ValueError(f"Unknown cost_basis='{cost_basis}'. Valid options: {valid}")
    return COST_ASSUMPTION_BASES[cost_basis]


def compute_counterfactuals(
    panel_comp: pd.DataFrame,
    struct_df: pd.DataFrame,
    recommendation_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      panel_loss: location-day panel with base + scenario expected losses
      annual_cluster: cluster-year aggregated losses
    """
    panel = panel_comp.merge(struct_df[["cluster_id", "structures_per_sqmi"]], on="cluster_id", how="left")
    panel = panel.merge(recommendation_map[["cluster_id", "recommendation"]], on="cluster_id", how="left")
    panel["recommendation"] = panel["recommendation"].fillna("targeted_monitoring")
    panel["structures_per_sqmi"] = pd.to_numeric(panel["structures_per_sqmi"], errors="coerce").fillna(0.0)
    if "fuel_category" not in panel.columns:
        panel["fuel_category"] = "unknown"
    if "hftd_tier" not in panel.columns:
        panel["hftd_tier"] = "non_hftd"
    if "tx_len_km_10km" not in panel.columns:
        panel["tx_len_km_10km"] = np.nan
    if "plausibility_gated_row" not in panel.columns:
        panel["plausibility_gated_row"] = False
    panel["is_hftd"] = panel["hftd_tier"].astype(str).str.lower().ne("non_hftd")

    # Base
    panel["structures_impacted_given_ign_base"] = (
        panel["structures_per_sqmi"] * (panel["E_area_given_ign_base"] / 640.0) * EXPOSURE_MULTIPLIER
    )
    panel["loss_given_ign_base_usd"] = panel["structures_impacted_given_ign_base"] * DAMAGE_PER_STRUCTURE_USD
    panel["expected_daily_loss_base_usd"] = panel["p_ignition_cal"] * panel["loss_given_ign_base_usd"]
    if {"P_large_base_raw", "E_area_large_base_raw", "E_area_small"}.issubset(set(panel.columns)):
        panel["E_area_given_ign_base_ungated"] = (
            panel["P_large_base_raw"] * panel["E_area_large_base_raw"]
            + (1.0 - panel["P_large_base_raw"]) * panel["E_area_small"]
        )
        panel["structures_impacted_given_ign_ungated"] = (
            panel["structures_per_sqmi"] * (panel["E_area_given_ign_base_ungated"] / 640.0) * EXPOSURE_MULTIPLIER
        )
        panel["loss_given_ign_base_ungated_usd"] = panel["structures_impacted_given_ign_ungated"] * DAMAGE_PER_STRUCTURE_USD
        panel["expected_daily_loss_base_ungated_usd"] = panel["p_ignition_cal"] * panel["loss_given_ign_base_ungated_usd"]
    else:
        panel["expected_daily_loss_base_ungated_usd"] = panel["expected_daily_loss_base_usd"]

    for s in SCENARIOS:
        # Step 1 (baseline in HFTD): fast-trip
        rf_fast = panel.apply(
            lambda r: _scenario_reduction("fast_trip_and_hardening", "r_freq", s, r["fuel_category"]) if r["is_hftd"] else 0.0,
            axis=1,
        )
        rt_fast = panel.apply(
            lambda r: _scenario_reduction("fast_trip_and_hardening", "r_tail", s, r["fuel_category"]) if r["is_hftd"] else 0.0,
            axis=1,
        )
        rs_fast = panel.apply(
            lambda r: _scenario_reduction("fast_trip_and_hardening", "r_spread", s, r["fuel_category"]) if r["is_hftd"] else 0.0,
            axis=1,
        )

        p_after_fast = panel["p_ignition_cal"] * (1.0 - rf_fast)
        p_large_after_fast = panel["P_large_base"] * (1.0 - rt_fast)
        e_large_after_fast = panel["E_area_large_base"] * (1.0 - rs_fast)

        # Step 2 (cluster-specific escalation action on residual risk)
        rf_sec = panel.apply(lambda r: _scenario_reduction(r["recommendation"], "r_freq", s, r["fuel_category"]), axis=1)
        rt_sec = panel.apply(lambda r: _scenario_reduction(r["recommendation"], "r_tail", s, r["fuel_category"]), axis=1)
        rs_sec = panel.apply(lambda r: _scenario_reduction(r["recommendation"], "r_spread", s, r["fuel_category"]), axis=1)

        # Avoid double-counting fast-trip if it is also the selected recommendation.
        is_fast_only = panel["recommendation"].eq("fast_trip_and_hardening")
        rf_sec = rf_sec.where(~is_fast_only, 0.0)
        rt_sec = rt_sec.where(~is_fast_only, 0.0)
        rs_sec = rs_sec.where(~is_fast_only, 0.0)

        p_cf = p_after_fast * (1.0 - rf_sec)
        p_large_cf = p_large_after_fast * (1.0 - rt_sec)
        e_large_cf = e_large_after_fast * (1.0 - rs_sec)
        e_area_cf = p_large_cf * e_large_cf + (1.0 - p_large_cf) * panel["E_area_small"]
        impacted_cf = panel["structures_per_sqmi"] * (e_area_cf / 640.0) * EXPOSURE_MULTIPLIER
        loss_given_ign_cf = impacted_cf * DAMAGE_PER_STRUCTURE_USD
        daily_cf = p_cf * loss_given_ign_cf

        panel[f"p_ignition_cf_{s}"] = p_cf
        panel[f"P_large_cf_{s}"] = p_large_cf
        panel[f"E_area_large_cf_{s}"] = e_large_cf
        panel[f"loss_given_ign_cf_{s}_usd"] = loss_given_ign_cf
        panel[f"expected_daily_loss_cf_{s}_usd"] = daily_cf

    agg_dict: dict[str, tuple[str, str]] = {
        "mean_daily_p": ("p_ignition_cal", "mean"),
        "P_large_mean": ("P_large_base", "mean"),
        "E_area_large_mean": ("E_area_large_base", "mean"),
        "EAL_base_usd": ("expected_daily_loss_base_usd", "sum"),
        "EAL_base_ungated_usd": ("expected_daily_loss_base_ungated_usd", "sum"),
        "mean_loss_given_ign_base_usd": ("loss_given_ign_base_usd", "mean"),
        "plausibility_gated": ("plausibility_gated_row", "max"),
    }
    for s in SCENARIOS:
        agg_dict[f"EAL_cf_{s}_usd"] = (f"expected_daily_loss_cf_{s}_usd", "sum")
        agg_dict[f"mean_p_cf_{s}"] = (f"p_ignition_cf_{s}", "mean")
        agg_dict[f"mean_loss_given_ign_cf_{s}_usd"] = (f"loss_given_ign_cf_{s}_usd", "mean")

    annual = panel.groupby(["cluster_id", "year", "recommendation", "structures_per_sqmi"], as_index=False).agg(**agg_dict)
    annual["plausibility_gated"] = annual["plausibility_gated"].astype(bool)
    if "fuel_category" in panel.columns:
        fuel_mode = panel.groupby(["cluster_id", "year"])["fuel_category"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown"
        ).reset_index(name="fuel_category")
        annual = annual.merge(fuel_mode, on=["cluster_id", "year"], how="left")
    if "hftd_tier" in panel.columns:
        hftd_mode = panel.groupby(["cluster_id", "year"])["hftd_tier"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else "non_hftd"
        ).reset_index(name="hftd_tier")
        annual = annual.merge(hftd_mode, on=["cluster_id", "year"], how="left")
    if "tx_len_km_10km" in panel.columns:
        tx = panel.groupby(["cluster_id", "year"])["tx_len_km_10km"].mean().reset_index(name="tx_len_km_10km")
        annual = annual.merge(tx, on=["cluster_id", "year"], how="left")
    annual["annualized_prob_base"] = 1.0 - np.power(1.0 - annual["mean_daily_p"].clip(0, 1), 365.0)
    annual["EAL_base_annualized_usd"] = annual["annualized_prob_base"] * annual["mean_loss_given_ign_base_usd"]
    annual["delta_plausibility_gate_usd"] = annual["EAL_base_ungated_usd"] - annual["EAL_base_usd"]
    for s in SCENARIOS:
        annual[f"annualized_prob_cf_{s}"] = 1.0 - np.power(1.0 - annual[f"mean_p_cf_{s}"].clip(0, 1), 365.0)
        annual[f"EAL_cf_{s}_annualized_usd"] = annual[f"annualized_prob_cf_{s}"] * annual[f"mean_loss_given_ign_cf_{s}_usd"]

    return panel, annual


def _fuel_plot_group(v: Any) -> str:
    f = _norm_fuel(v)
    if f in {"timber_litter", "timber_understory"}:
        return "timber"
    if f in {"shrub", "grass_shrub", "slash"}:
        return "shrub"
    if f == "grass":
        return "grass"
    if f == "nonburnable":
        return "nonburnable"
    return "other"


def _hftd_plot_group(v: Any) -> str:
    t = _canon_hftd_tier(v)
    if t == "tier_3":
        return "Tier 3"
    if t == "tier_2":
        return "Tier 2"
    return "Tier 1/non-HFTD"


def make_frap_lookup_chart(lookup: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    work = lookup.copy()
    work["fuel_plot"] = work["fuel_category"].map(_fuel_plot_group)
    work["tier_plot"] = work["hftd_tier"].map(_hftd_plot_group)
    work = work[work["fuel_plot"].isin(["grass", "shrub", "timber", "nonburnable"])].copy()

    plot_df = pd.DataFrame(
        {
            "hftd_tier_plot": work["tier_plot"],
            "fuel_category_plot": work["fuel_plot"],
            "expected_area_ac": pd.to_numeric(work["e_area_total"], errors="coerce"),
            "ci95_low_ac": np.nan,
            "ci95_high_ac": np.nan,
            "n_fires": pd.to_numeric(work["n_fires"], errors="coerce").fillna(0).astype(int),
        }
    )
    tier_order = ["Tier 3", "Tier 2", "Tier 1/non-HFTD"]
    fuel_order = ["grass", "shrub", "timber", "nonburnable"]
    colors = {
        "grass": "#66c2a5",
        "shrub": "#fc8d62",
        "timber": "#8da0cb",
        "nonburnable": "#bdbdbd",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(tier_order), dtype=float)
    w = 0.18
    for i, fuel in enumerate(fuel_order):
        sub = []
        y = []
        for tier in tier_order:
            r = plot_df[(plot_df["hftd_tier_plot"] == tier) & (plot_df["fuel_category_plot"] == fuel)]
            if r.empty:
                yv = np.nan
                n = 0
            else:
                rr = r.iloc[0]
                yv = float(rr["expected_area_ac"])
                n = int(rr["n_fires"])
            sub.append({"tier": tier, "n": n})
            y.append(yv)
        xpos = x + (i - 1.5) * w
        y_plot = np.array([max(v, 0.1) if np.isfinite(v) else np.nan for v in y], dtype=float)
        ax.bar(xpos, y_plot, width=w, label=fuel, color=colors[fuel], edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(tier_order)
    ax.set_yscale("log")
    ax.set_ylabel("Expected acres | ignition (log scale)")
    ax.set_xlabel("HFTD tier")
    ax.set_title("Severity Lookup Used in Model (Utility-Anchored, FRAP Fallback)")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend(title="Fuel category", ncol=2, frameon=False)
    fig.tight_layout()

    out = OUTDIR / "frap_severity_lookup_chart.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out, plot_df


def pareto_summary(values: pd.Series, cuts: tuple[float, ...] = (0.05, 0.10, 0.20, 0.50)) -> dict[str, float]:
    v = pd.to_numeric(values, errors="coerce").fillna(0.0).sort_values(ascending=False).to_numpy()
    total = float(v.sum())
    out: dict[str, float] = {}
    if total <= 0 or len(v) == 0:
        for c in cuts:
            out[f"top_{int(c*100)}pct_share"] = 0.0
        return out
    for c in cuts:
        n = max(1, int(math.ceil(len(v) * c)))
        out[f"top_{int(c*100)}pct_share"] = float(v[:n].sum() / total)
    return out


def bootstrap_portfolio_ci(
    by_cluster: pd.DataFrame, n_boot: int = 1000, seed: int = 42
) -> dict[str, list[float] | int]:
    """
    Cluster bootstrap confidence intervals for headline portfolio metrics.
    Sampling unit is cluster (with replacement) over the cluster-level EAL vector.
    """
    req = {"EAL_base_usd", "delta_mid_usd", "annualized_cost_mid_usd"}
    missing = req.difference(set(by_cluster.columns))
    if missing:
        raise ValueError(f"bootstrap_portfolio_ci missing columns: {sorted(missing)}")

    base = pd.to_numeric(by_cluster["EAL_base_usd"], errors="coerce").fillna(0.0).to_numpy(float)
    delta = pd.to_numeric(by_cluster["delta_mid_usd"], errors="coerce").fillna(0.0).to_numpy(float)
    cost = pd.to_numeric(by_cluster["annualized_cost_mid_usd"], errors="coerce").fillna(0.0).to_numpy(float)

    n = len(base)
    if n == 0:
        raise ValueError("bootstrap_portfolio_ci received empty by_cluster table.")

    rng = np.random.default_rng(seed)
    base_boot = np.empty(n_boot, dtype=float)
    delta_boot = np.empty(n_boot, dtype=float)
    bcr_boot = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        b = float(base[idx].sum())
        d = float(delta[idx].sum())
        c = float(cost[idx].sum())
        base_boot[i] = b
        delta_boot[i] = d
        bcr_boot[i] = (d / c) if c > 0 else np.nan

    def ci(arr: np.ndarray, point: float) -> list[float]:
        clean = arr[np.isfinite(arr)]
        if len(clean) == 0:
            return [float("nan"), float(point), float("nan")]
        lo = float(np.percentile(clean, 2.5))
        hi = float(np.percentile(clean, 97.5))
        return [lo, float(point), hi]

    base_point = float(base.sum())
    delta_point = float(delta.sum())
    cost_point = float(cost.sum())
    bcr_point = float(delta_point / cost_point) if cost_point > 0 else float("nan")

    return {
        "n_bootstrap": int(n_boot),
        "portfolio_base_eal_usd_ci95": ci(base_boot, base_point),
        "portfolio_mid_avoided_eal_usd_ci95": ci(delta_boot, delta_point),
        "portfolio_mid_bcr_ci95": ci(bcr_boot, bcr_point),
    }


def build_validation_summary_table(
    panel: pd.DataFrame,
    model_eval: dict[str, Any],
    by_cluster: pd.DataFrame,
    portfolio_base: float,
    portfolio_delta: dict[str, float],
    portfolio_bcr: dict[str, float],
    bootstrap_ci: dict[str, Any],
) -> tuple[pd.DataFrame, Path]:
    """
    Build one judge-friendly table with key validation and scenario outputs.
    """
    test = panel[panel["year"] == TEST_YEAR].copy()
    ignition_series = test["ignition"] if "ignition" in test.columns else pd.Series(0, index=test.index, dtype=float)
    y = pd.to_numeric(ignition_series, errors="coerce").fillna(0).astype(int).to_numpy()
    p = pd.to_numeric(test.get("p_ignition_cal"), errors="coerce").fillna(0.0).clip(0, 1).to_numpy()
    prevalence = float(y.mean()) if len(y) else float("nan")

    if len(y) and len(np.unique(y)) > 1:
        ign_auc = float(roc_auc_score(y, p))
        ign_pr = float(average_precision_score(y, p))
    else:
        ign_auc = float("nan")
        ign_pr = float("nan")
    ign_pr_lift = float(ign_pr / prevalence) if np.isfinite(ign_pr) and prevalence > 0 else float("nan")

    if len(y):
        k = max(1, int(math.ceil(0.10 * len(y))))
        order = np.argsort(-p)
        hits = int(y[order[:k]].sum())
        top10_capture = float(hits / max(int(y.sum()), 1))
    else:
        top10_capture = float("nan")

    sev_prev = (
        float(model_eval.get("n_large_test_2024", 0) / model_eval.get("n_events_test_2024", 0))
        if model_eval.get("n_events_test_2024", 0) > 0
        else float("nan")
    )

    pareto_top10 = float(pareto_summary(by_cluster["EAL_base_usd"]).get("top_10pct_share", float("nan")))
    ci_base = bootstrap_ci.get("portfolio_base_eal_usd_ci95", [float("nan"), float("nan"), float("nan")])
    ci_delta = bootstrap_ci.get("portfolio_mid_avoided_eal_usd_ci95", [float("nan"), float("nan"), float("nan")])
    ci_bcr = bootstrap_ci.get("portfolio_mid_bcr_ci95", [float("nan"), float("nan"), float("nan")])

    rows = [
        {
            "section": "Ignition model",
            "prevalence": prevalence,
            "auc": ign_auc,
            "pr_auc": ign_pr,
            "pr_lift_over_baseline": ign_pr_lift,
            "top_10pct_capture_rate": top10_capture,
        },
        {
            "section": "Severity model",
            "large_fire_prevalence_test_2024": sev_prev,
            "classifier_auc_test_2024": model_eval.get("classifier_auc_test_2024"),
            "classifier_pr_auc_test_2024": model_eval.get("classifier_pr_auc_test_2024"),
            "severity_model_type": model_eval.get("severity_model_type", "empirical_lookup"),
        },
        {
            "section": "Portfolio",
            "base_eal_usd": float(portfolio_base),
            "pareto_top_10pct_share": pareto_top10,
            "mid_avoided_eal_usd": float(portfolio_delta["mid"]),
            "benefit_cost_ratio_mid": float(portfolio_bcr["mid"]),
            "base_eal_ci95_lower": ci_base[0],
            "base_eal_ci95_point": ci_base[1],
            "base_eal_ci95_upper": ci_base[2],
            "mid_avoided_eal_ci95_lower": ci_delta[0],
            "mid_avoided_eal_ci95_point": ci_delta[1],
            "mid_avoided_eal_ci95_upper": ci_delta[2],
            "mid_bcr_ci95_lower": ci_bcr[0],
            "mid_bcr_ci95_point": ci_bcr[1],
            "mid_bcr_ci95_upper": ci_bcr[2],
        },
        {
            "section": "Robustness",
            "low_avoided_eal_usd": float(portfolio_delta["low"]),
            "high_avoided_eal_usd": float(portfolio_delta["high"]),
        },
    ]
    tbl = pd.DataFrame(rows)
    out = OUTDIR / "model_validation_and_scenario_summary.csv"
    tbl.to_csv(out, index=False)

    print("\nModel Validation and Scenario Summary")
    print("-" * 72)
    print(
        f"Ignition  | prevalence={prevalence:.6f}  AUC={ign_auc:.4f}  PR-AUC={ign_pr:.6f}  "
        f"PR-lift={ign_pr_lift:.2f}x  top10% capture={top10_capture:.4f}"
    )
    print(
        f"Severity  | large-prev={sev_prev:.4f}  AUC={model_eval.get('classifier_auc_test_2024')}  "
        f"PR-AUC={model_eval.get('classifier_pr_auc_test_2024')}"
    )
    print(
        f"Portfolio | base EAL=${portfolio_base:,.0f}  top10 share={pareto_top10:.3f}  "
        f"MID avoided=${portfolio_delta['mid']:,.0f}  MID BCR={portfolio_bcr['mid']:.2f}"
    )
    print(
        f"Robust    | LOW avoided=${portfolio_delta['low']:,.0f}  HIGH avoided=${portfolio_delta['high']:,.0f}"
    )
    print(
        f"95% CI    | base=[{ci_base[0]:,.0f}, {ci_base[1]:,.0f}, {ci_base[2]:,.0f}]  "
        f"MID avoided=[{ci_delta[0]:,.0f}, {ci_delta[1]:,.0f}, {ci_delta[2]:,.0f}]  "
        f"MID BCR=[{ci_bcr[0]:.2f}, {ci_bcr[1]:.2f}, {ci_bcr[2]:.2f}]"
    )
    return tbl, out


def _treated_miles_proxy(scenario: str) -> float:
    if scenario not in TREATED_MILES_BY_SCENARIO:
        valid = ", ".join(sorted(TREATED_MILES_BY_SCENARIO.keys()))
        raise ValueError(f"Unknown scenario '{scenario}' for treated miles. Valid: {valid}")
    return float(TREATED_MILES_BY_SCENARIO[scenario])


def add_cost_effectiveness(by_cluster: pd.DataFrame, cost_basis: str) -> pd.DataFrame:
    out = by_cluster.copy()
    basis = _get_cost_basis(cost_basis)
    cost_assumptions = basis["assumptions"]
    # Keep a MID display column for convenience; scenario-specific values are
    # applied during each LOW/MID/HIGH cost computation.
    out["treated_miles_proxy"] = _treated_miles_proxy("mid")
    crf = _crf(CAPITAL_DISCOUNT_RATE, CAPITAL_LIFETIME_YEARS)
    if "hftd_tier" in out.columns:
        out["is_hftd"] = out["hftd_tier"].astype(str).str.lower().ne("non_hftd")
    else:
        out["is_hftd"] = False

    for s in SCENARIOS:
        treated_miles_s = _treated_miles_proxy(s)
        out[f"treated_miles_proxy_{s}"] = treated_miles_s
        annual_cost = []
        for _, r in out.iterrows():
            action = str(r["recommendation"])
            if action not in cost_assumptions:
                action = "targeted_monitoring"

            # Baseline fast-trip cost on all HFTD clusters.
            total = 0.0
            if bool(r["is_hftd"]):
                unit_fast = cost_assumptions["fast_trip_and_hardening"]
                capex_fast = float(unit_fast["capex_per_mile"][s]) * treated_miles_s
                opex_fast = float(unit_fast["annual_opex_per_mile"][s]) * treated_miles_s
                total += capex_fast * crf + opex_fast

            # Additional escalation action cost (if any).
            if action != "fast_trip_and_hardening":
                unit = cost_assumptions[action]
                capex = float(unit["capex_per_mile"][s]) * treated_miles_s
                opex = float(unit["annual_opex_per_mile"][s]) * treated_miles_s
                total += capex * crf + opex

            annual_cost.append(total)
        out[f"annualized_cost_{s}_usd"] = annual_cost
        out[f"net_benefit_{s}_usd"] = out[f"delta_{s}_usd"] - out[f"annualized_cost_{s}_usd"]
        out[f"benefit_cost_ratio_{s}"] = out[f"delta_{s}_usd"] / out[f"annualized_cost_{s}_usd"].replace(0, np.nan)
        out[f"benefit_cost_ratio_{s}"] = out[f"benefit_cost_ratio_{s}"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out


def make_plot_top20(by_cluster: pd.DataFrame) -> Path:
    top = by_cluster.sort_values("EAL_base_usd", ascending=False).head(TOP_N_PLOT).copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top))
    w = 0.38
    ax.bar(x - w / 2, top["EAL_base_usd"], width=w, label="Status Quo (Base)")
    ax.bar(x + w / 2, top["EAL_cf_mid_usd"], width=w, label="Counterfactual (MID)")
    ax.set_xticks(x)
    ax.set_xticklabels(top["cluster_id"].astype(str), rotation=60, ha="right")
    ax.set_ylabel("Expected Annual Loss (USD)")
    ax.set_title("Top 20 Clusters: Status Quo vs Recommended Counterfactual (MID)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = OUTDIR / "eal_status_quo_vs_counterfactual.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def compare_el_vs_score_prioritization(
    by_cluster: pd.DataFrame, top_n: int = 25, scenario: str = "mid"
) -> dict[str, float | int]:
    """
    Compare EL-ranked selection vs score-ranked selection under equal annualized-cost budget.
    This yields an intuitive "same budget, how much avoided loss do we buy?" metric.
    """
    need = {f"delta_{scenario}_usd", f"annualized_cost_{scenario}_usd", "EAL_base_usd", "mean_daily_p"}
    missing = [c for c in need if c not in by_cluster.columns]
    if missing:
        raise ValueError(f"compare_el_vs_score_prioritization missing columns: {missing}")

    el = by_cluster.sort_values("EAL_base_usd", ascending=False).head(top_n).copy()
    budget = float(pd.to_numeric(el[f"annualized_cost_{scenario}_usd"], errors="coerce").fillna(0.0).sum())

    score = by_cluster.sort_values("mean_daily_p", ascending=False).copy()
    score[f"annualized_cost_{scenario}_usd"] = pd.to_numeric(score[f"annualized_cost_{scenario}_usd"], errors="coerce").fillna(0.0)
    score["_cum_cost"] = score[f"annualized_cost_{scenario}_usd"].cumsum()
    score = score[score["_cum_cost"] <= budget].copy()
    if score.empty:
        score = by_cluster.sort_values("mean_daily_p", ascending=False).head(1).copy()

    el_avoided = float(pd.to_numeric(el[f"delta_{scenario}_usd"], errors="coerce").fillna(0.0).sum())
    el_cost = float(pd.to_numeric(el[f"annualized_cost_{scenario}_usd"], errors="coerce").fillna(0.0).sum())
    score_avoided = float(pd.to_numeric(score[f"delta_{scenario}_usd"], errors="coerce").fillna(0.0).sum())
    score_cost = float(pd.to_numeric(score[f"annualized_cost_{scenario}_usd"], errors="coerce").fillna(0.0).sum())

    el_bcr = float(el_avoided / el_cost) if el_cost > 0 else 0.0
    score_bcr = float(score_avoided / score_cost) if score_cost > 0 else 0.0
    ratio = float(el_bcr / score_bcr) if score_bcr > 0 else float("inf")

    return {
        "scenario": scenario,
        "top_n_el_ranked": int(top_n),
        "el_ranked_budget_usd": budget,
        "el_ranked_n_selected": int(len(el)),
        "score_ranked_n_selected_within_budget": int(len(score)),
        "el_ranked_avoided_usd": el_avoided,
        "score_ranked_avoided_usd": score_avoided,
        "el_ranked_bcr": el_bcr,
        "score_ranked_bcr": score_bcr,
        "bcr_ratio_el_ranked_vs_score_ranked": ratio,
    }


def main() -> None:
    data = load_inputs()
    panel = data["panel"]
    incidents = data["incidents"]
    structures = data["structures"]
    existing_recs = data["existing_recs"]
    detected = data["detected"]
    centroids = build_centroids(panel)

    frap_raw, frap_meta = load_frap_perimeters(use_all_causes=True)
    frap_tagged = annotate_frap_with_strata(frap_raw)
    e_area_small_anchor = E_AREA_SMALL_DEFAULT
    lookup, lookup_stats = build_frap_severity_lookup(frap_tagged, centroids, incidents, e_area_small_anchor)

    frap_chart_path, frap_chart_table = make_frap_lookup_chart(lookup)

    panel_comp, cluster_severity = predict_severity_components_from_lookup(panel, lookup, centroids)

    frap_test = frap_tagged[frap_tagged["year"] == TEST_YEAR].copy()
    severity_eval = {
        "severity_model_type": "empirical_lookup_utility_anchored_with_frap_fallback",
        "n_events_test_2024": int(len(frap_test)),
        "n_large_test_2024": int((pd.to_numeric(frap_test["gis_acres"], errors="coerce") >= LARGE_FIRE_THRESHOLD_AC).sum()),
        "classifier_auc_test_2024": None,
        "classifier_pr_auc_test_2024": None,
    }

    # Initial baseline for recommendation assignment at cluster-year level
    tmp = panel_comp.merge(structures[["cluster_id", "structures_per_sqmi"]], on="cluster_id", how="left")
    tmp["structures_per_sqmi"] = pd.to_numeric(tmp["structures_per_sqmi"], errors="coerce").fillna(0.0)
    tmp["E_area_given_ign_base"] = tmp["E_area_given_ign_base"].fillna(e_area_small_anchor)
    tmp["loss_given_ign_base_usd"] = (
        tmp["structures_per_sqmi"] * (tmp["E_area_given_ign_base"] / 640.0) * EXPOSURE_MULTIPLIER * DAMAGE_PER_STRUCTURE_USD
    )
    tmp["expected_daily_loss_base_usd"] = tmp["p_ignition_cal"] * tmp["loss_given_ign_base_usd"]
    annual_base = (
        tmp.groupby(["cluster_id", "year", "structures_per_sqmi"], as_index=False)
        .agg(
            mean_daily_p=("p_ignition_cal", "mean"),
            EAL_base_usd=("expected_daily_loss_base_usd", "sum"),
            P_large_mean=("P_large_base", "mean"),
            E_area_large_mean=("E_area_large_base", "mean"),
            E_area_given_ign_mean=("E_area_given_ign_base", "mean"),
            mean_loss_given_ign_base_usd=("loss_given_ign_base_usd", "mean"),
        )
    )
    if "fuel_category" in tmp.columns:
        fuel_mode = tmp.groupby(["cluster_id", "year"])["fuel_category"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown"
        ).reset_index(name="fuel_category")
        annual_base = annual_base.merge(fuel_mode, on=["cluster_id", "year"], how="left")
    if "hftd_tier" in tmp.columns:
        hftd_mode = tmp.groupby(["cluster_id", "year"])["hftd_tier"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else "non_hftd"
        ).reset_index(name="hftd_tier")
        annual_base = annual_base.merge(hftd_mode, on=["cluster_id", "year"], how="left")

    target_year = int(panel["year"].max())
    by_cluster_init = annual_base[annual_base["year"] == target_year].copy()
    by_cluster_init = _assign_recommendations(by_cluster_init, existing_recs)
    rec_map = by_cluster_init[["cluster_id", "recommendation"]].copy()
    decision_trace_cols = [
        "cluster_id",
        "recommendation",
        "decision_rule",
        "residual_eal_after_fast_trip_mid",
        "EAL_base_usd",
        "structures_per_sqmi",
        "fuel_category",
        "hftd_tier",
    ]
    decision_trace = by_cluster_init[[c for c in decision_trace_cols if c in by_cluster_init.columns]].copy()
    decision_trace.to_csv(OUTDIR / f"recommendation_decision_trace_{target_year}.csv", index=False)

    panel_loss, annual = compute_counterfactuals(panel_comp, structures, rec_map)
    by_cluster = annual[annual["year"] == target_year].copy()
    by_cluster["delta_low_usd"] = by_cluster["EAL_base_usd"] - by_cluster["EAL_cf_low_usd"]
    by_cluster["delta_mid_usd"] = by_cluster["EAL_base_usd"] - by_cluster["EAL_cf_mid_usd"]
    by_cluster["delta_high_usd"] = by_cluster["EAL_base_usd"] - by_cluster["EAL_cf_high_usd"]
    by_cluster = by_cluster.merge(
        by_cluster_init[["cluster_id", "decision_rule", "residual_eal_after_fast_trip_mid", "E_area_given_ign_mean"]],
        on="cluster_id",
        how="left",
    )
    by_cluster = by_cluster.rename(columns={"E_area_given_ign_mean": "e_area_given_ign_ac"})
    by_cluster["e_loss_given_ign_usd"] = by_cluster["mean_loss_given_ign_base_usd"]
    by_cluster = add_cost_effectiveness(by_cluster, cost_basis=COST_BASIS_DEFAULT)

    # exact requested column names
    out_cols = [
        "cluster_id",
        "EAL_base_usd",
        "EAL_base_ungated_usd",
        "delta_plausibility_gate_usd",
        "recommendation",
        "plausibility_gated",
        "EAL_cf_low_usd",
        "EAL_cf_mid_usd",
        "EAL_cf_high_usd",
        "delta_low_usd",
        "delta_mid_usd",
        "delta_high_usd",
        "decision_rule",
        "residual_eal_after_fast_trip_mid",
        "mean_daily_p",
        "P_large_mean",
        "E_area_large_mean",
        "e_area_given_ign_ac",
        "e_loss_given_ign_usd",
        "structures_per_sqmi",
        "fuel_category",
        "hftd_tier",
        "tx_len_km_10km",
        "treated_miles_proxy",
        "treated_miles_proxy_low",
        "treated_miles_proxy_mid",
        "treated_miles_proxy_high",
        "EAL_base_annualized_usd",
        "EAL_cf_low_annualized_usd",
        "EAL_cf_mid_annualized_usd",
        "EAL_cf_high_annualized_usd",
        "annualized_cost_low_usd",
        "annualized_cost_mid_usd",
        "annualized_cost_high_usd",
        "net_benefit_low_usd",
        "net_benefit_mid_usd",
        "net_benefit_high_usd",
        "benefit_cost_ratio_low",
        "benefit_cost_ratio_mid",
        "benefit_cost_ratio_high",
    ]
    by_cluster = by_cluster[out_cols].sort_values("EAL_base_usd", ascending=False).reset_index(drop=True)

    out_csv = OUTDIR / "severity_hurdle_by_cluster.csv"
    by_cluster.to_csv(out_csv, index=False)
    out_csv_frap = OUTDIR / "severity_by_cluster.csv"
    by_cluster.to_csv(out_csv_frap, index=False)

    # Portfolio + recommendation summaries
    portfolio_base = float(by_cluster["EAL_base_usd"].sum())
    portfolio_cf = {s: float(by_cluster[f"EAL_cf_{s}_usd"].sum()) for s in SCENARIOS}
    portfolio_delta = {s: portfolio_base - portfolio_cf[s] for s in SCENARIOS}
    portfolio_base_ann = float(by_cluster["EAL_base_annualized_usd"].sum())
    portfolio_cf_ann = {s: float(by_cluster[f"EAL_cf_{s}_annualized_usd"].sum()) for s in SCENARIOS}

    rec_counts = by_cluster["recommendation"].value_counts().to_dict()
    rec_base_share = (
        by_cluster.groupby("recommendation")["EAL_base_usd"].sum() / max(portfolio_base, 1e-12)
    ).to_dict()
    rec_mid_share = (
        by_cluster.groupby("recommendation")["EAL_cf_mid_usd"].sum() / max(portfolio_cf["mid"], 1e-12)
    ).to_dict()

    gated_mask = by_cluster["plausibility_gated"].astype(bool)
    base_ungated_total = float(by_cluster["EAL_base_ungated_usd"].sum())
    gate_shift_total = float(by_cluster["delta_plausibility_gate_usd"].sum())
    gate_shift_share = float(gate_shift_total / max(base_ungated_total, 1e-12))
    gate_shift_gated_share = float(
        by_cluster.loc[gated_mask, "delta_plausibility_gate_usd"].sum() / max(gate_shift_total, 1e-12)
    ) if gate_shift_total > 0 else 0.0

    portfolio_cost = {s: float(by_cluster[f"annualized_cost_{s}_usd"].sum()) for s in SCENARIOS}
    portfolio_net = {s: float(by_cluster[f"net_benefit_{s}_usd"].sum()) for s in SCENARIOS}
    portfolio_bcr = {
        s: float(portfolio_delta[s] / portfolio_cost[s]) if portfolio_cost[s] > 0 else 0.0
        for s in SCENARIOS
    }
    bootstrap_ci = bootstrap_portfolio_ci(by_cluster, n_boot=1000, seed=42)
    validation_table, validation_csv = build_validation_summary_table(
        panel=panel,
        model_eval=severity_eval,
        by_cluster=by_cluster,
        portfolio_base=portfolio_base,
        portfolio_delta=portfolio_delta,
        portfolio_bcr=portfolio_bcr,
        bootstrap_ci=bootstrap_ci,
    )
    _ = validation_table  # explicit keep

    plot_path = make_plot_top20(by_cluster)
    prioritization_cmp = compare_el_vs_score_prioritization(by_cluster, top_n=25, scenario="mid")

    metrics = {
        "claim_scope": (
            "Results represent expected-loss-ranked risk prioritization under public-data assumptions and "
            "literature-derived intervention effect sizes. Dollar figures are modeled estimates, not causal "
            "predictions of actual savings. Uncertainty bounds reflect sampling variability in the cluster-level "
            "EAL distribution."
        ),
        "methodology": {
            "risk_decomposition": "ExpectedLoss(s,t)=P_ignition(s,t)*E(Loss|ignition,s,t)",
            "policy_sequence": [
                "Apply fast-trip baseline to all HFTD clusters",
                "Compute residual EAL after fast-trip",
                "Escalate to covered conductor for high residual EAL",
                "Reserve undergrounding for extreme residual EAL + high structure density",
                "Use vegetation management in vegetated-fuel moderate residual EAL clusters",
            ],
            "severity_model": {
                "type": "empirical_lookup_utility_anchored_with_frap_fallback",
                "description": (
                    "Static cluster-level lookup stratified by HFTD tier and LANDFIRE fuel category. "
                    "Primary anchor is confirmed utility ignition severity (2021+) with empirical-Bayes shrinkage; "
                    "FRAP (1992-2024) is used only as sparse-data fallback."
                ),
                "large_fire_threshold_ac": LARGE_FIRE_THRESHOLD_AC,
                "combination": "E(area|ign,s)=P_large(s)*E_area_large(s)+(1-P_large(s))*E_area_small(s)",
                "stratification": ["hftd_tier", "fuel_category"],
                "min_events_per_stratum": {
                    "utility": UTILITY_MIN_STRATUM_COUNT,
                    "frap_fallback": FRAP_MIN_STRATUM_COUNT,
                },
                "backfill_order": [
                    "utility_hftd_fuel",
                    "utility_fuel_only",
                    "utility_hftd_only",
                    "utility_statewide",
                    "frap_hftd_fuel",
                    "frap_hftd_only",
                    "frap_statewide",
                ],
                "frap_filters": {
                    "year_min": FRAP_YEAR_MIN,
                    "year_max": FRAP_YEAR_MAX,
                    "min_acres": 0.0,
                    "winsor_quantile": FRAP_WINSOR_QUANTILE,
                    "all_causes_used": True,
                },
                "frap_lookup_summary": lookup_stats,
            },
            "wildfire_plausibility_gate": {
                "rule": "If fuel_category == nonburnable, force P_large=0, E_area_large=0, E_area_small=2.0 acres.",
                "physical_basis": "FBFM40 nonburnable classes (urban/water/barren) do not support meaningful wildland spread.",
            },
            "loss_mapping": {
                "acres_to_sqmi": "acres/640",
                "expected_structures_impacted": "structures_per_sqmi * sqmi * exposure_multiplier",
                "damage_per_structure_usd": DAMAGE_PER_STRUCTURE_USD,
                "exposure_multiplier": EXPOSURE_MULTIPLIER,
            },
            "annual_loss_methods": {
                "primary": "sum_over_days_of_expected_daily_loss",
                "secondary": "annualized_prob_formula_1_minus_(1-mean_daily_p)^365_times_mean_loss_given_ign",
            },
            "counterfactual_effectiveness": EFFECTIVENESS_DEFAULTS,
            "counterfactual_effectiveness_source": WARNER_EFFECT_SOURCE,
            "effectiveness_anchors": {
                "fast_trip_r_freq_mid": 0.72,
                "vegetation_management_r_freq_mid": 0.57,
                "covered_conductor_r_freq_mid": 0.62,
                "undergrounding_r_freq_mid": 1.0,
                "notes": "LOW=50% of Warner point estimate; MID=Warner point estimate; HIGH=115% capped at 1.0.",
                "source": WARNER_EFFECT_SOURCE,
                "fuel_treatment_tail_spread_reference": "USFS fuel-treatment severity/spread literature synthesis",
            },
            "recommendation_thresholds": RECOMMEND_THRESHOLDS,
            "vegetation_applicability": {
                "rule_nonburnable_no_vegetation_recommendation": True,
                "veg_factor_by_fuel_category": VEG_FACTOR_BY_FUEL,
            },
            "cost_assumptions": {
                "cost_basis_used": COST_BASIS_DEFAULT,
                "cost_basis_source": _get_cost_basis(COST_BASIS_DEFAULT).get("source"),
                "cost_basis_source_url": _get_cost_basis(COST_BASIS_DEFAULT).get("source_url"),
                "capital_discount_rate": CAPITAL_DISCOUNT_RATE,
                "capital_lifetime_years": CAPITAL_LIFETIME_YEARS,
                "cost_ranges": _get_cost_basis(COST_BASIS_DEFAULT)["assumptions"],
                "treated_miles_proxy": {
                    "method": "scenario_fixed_miles_per_cluster",
                    "values": TREATED_MILES_BY_SCENARIO,
                    "note": (
                        "Physically grounded sensitivity values replacing transmission-length floor-clipped proxy. "
                        "LOW=targeted short segment, MID=moderate feeder segment, HIGH=broader multi-segment treatment."
                    ),
                },
            },
        },
        "detected_inputs": detected,
        "frap_source": frap_meta,
        "frap_lookup_table_head": lookup.head(20).to_dict(orient="records"),
        "model_evaluation_test_2024": severity_eval,
        "portfolio_totals_usd_sum_daily_primary": {
            "base": portfolio_base,
            "cf_low": portfolio_cf["low"],
            "cf_mid": portfolio_cf["mid"],
            "cf_high": portfolio_cf["high"],
            "delta_low": portfolio_delta["low"],
            "delta_mid": portfolio_delta["mid"],
            "delta_high": portfolio_delta["high"],
        },
        "portfolio_totals_usd_annualized_secondary": {
            "base": portfolio_base_ann,
            "cf_low": portfolio_cf_ann["low"],
            "cf_mid": portfolio_cf_ann["mid"],
            "cf_high": portfolio_cf_ann["high"],
        },
        "cost_effectiveness_portfolio": {
            "annualized_cost_low": portfolio_cost["low"],
            "annualized_cost_mid": portfolio_cost["mid"],
            "annualized_cost_high": portfolio_cost["high"],
            "net_benefit_low": portfolio_net["low"],
            "net_benefit_mid": portfolio_net["mid"],
            "net_benefit_high": portfolio_net["high"],
            "benefit_cost_ratio_low": portfolio_bcr["low"],
            "benefit_cost_ratio_mid": portfolio_bcr["mid"],
            "benefit_cost_ratio_high": portfolio_bcr["high"],
        },
        "pareto_concentration": {
            "base": pareto_summary(by_cluster["EAL_base_usd"]),
            "cf_mid": pareto_summary(by_cluster["EAL_cf_mid_usd"]),
        },
        "recommendation_counts": rec_counts,
        "recommendation_eal_share_base": {k: float(v) for k, v in rec_base_share.items()},
        "recommendation_eal_share_mid": {k: float(v) for k, v in rec_mid_share.items()},
        "plausibility_gate_summary": {
            "n_clusters_gated": int(gated_mask.sum()),
            "pct_clusters_gated": float(gated_mask.mean() * 100.0),
            "base_eal_ungated_usd": base_ungated_total,
            "base_eal_gated_usd": portfolio_base,
            "eal_shift_usd": gate_shift_total,
            "eal_shift_pct_of_ungated": gate_shift_share,
            "eal_shift_share_from_gated_clusters": gate_shift_gated_share,
        },
        "bootstrap_ci_95": bootstrap_ci,
        "prioritization_comparison_equal_budget_mid": prioritization_cmp,
        "outputs": {
            "severity_hurdle_by_cluster_csv": str(out_csv),
            "severity_by_cluster_csv": str(out_csv_frap),
            "frap_severity_lookup_csv": str(OUTDIR / "frap_severity_lookup.csv"),
            "frap_lookup_chart_png": str(frap_chart_path),
            "plot_status_quo_vs_cf": str(plot_path),
            "validation_summary_csv": str(validation_csv),
            "recommendation_decision_trace_csv": str(OUTDIR / f"recommendation_decision_trace_{target_year}.csv"),
        },
    }
    out_json = OUTDIR / "severity_hurdle_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    out_json_frap = OUTDIR / "severity_frap_metrics.json"
    with open(out_json_frap, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    frap_chart_table.to_csv(OUTDIR / "frap_lookup_bootstrap_summary.csv", index=False)

    old_portfolio_eal = 499_000_000.0
    old_mid_bcr = 3.20
    print("\n" + "=" * 72)
    print("FRAP EMPIRICAL SEVERITY COUNTERFACTUAL COMPLETE")
    print("=" * 72)
    print(f"  Target year (cluster summary): {target_year}")
    print(f"  FRAP fires used total:          {lookup_stats['n_fires_used_total']:,}")
    print(
        "  Strata fill levels:             "
        f"utility_direct={lookup_stats['n_utility_hftd_fuel']}, "
        f"utility_fuel={lookup_stats['n_utility_fuel_only']}, "
        f"utility_tier={lookup_stats['n_utility_hftd_only']}, "
        f"utility_state={lookup_stats['n_utility_statewide']}, "
        f"frap_direct={lookup_stats['n_frap_hftd_fuel']}, "
        f"frap_tier={lookup_stats['n_frap_hftd_only']}, "
        f"frap_state={lookup_stats['n_frap_statewide']}"
    )
    print(f"  Base portfolio EAL (sum daily): ${portfolio_base:,.0f} (old reference: ${old_portfolio_eal:,.0f})")
    print(f"  MID  portfolio EAL (sum daily): ${portfolio_cf['mid']:,.0f}")
    print(f"  MID avoided EAL:                ${portfolio_delta['mid']:,.0f}")
    print(f"  MID BCR:                        {portfolio_bcr['mid']:.2f}x (old reference: {old_mid_bcr:.2f}x)")
    print(f"  Saved: {out_csv}")
    print(f"  Saved: {out_csv_frap}")
    print(f"  Saved: {OUTDIR / 'frap_severity_lookup.csv'}")
    print(f"  Saved: {out_json}")
    print(f"  Saved: {out_json_frap}")
    print(f"  Saved: {frap_chart_path}")
    print(f"  Saved: {plot_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
