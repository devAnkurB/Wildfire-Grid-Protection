# LEGACY: superseded by severity_hurdle_model.py. Retained for reproducibility only.
"""
severity_loss_model.py
======================
Finalist-grade severity/loss layer for California wildfire utility risk.

Core equation:
  ExpectedLoss(s,t) = P_ignition(s,t) * E(Loss | ignition, s,t)

Primary structure-centric target:
  eq_struct_loss = structures_destroyed + 0.25 * structures_damaged

Two-part (hurdle) severity model:
  A) p_any_damage = P(eq_struct_loss > 0 | ignition, features)
  B) mu_damage    = E(eq_struct_loss | eq_struct_loss > 0, ignition, features)
  expected_eq_struct_loss_cond_ign = p_any_damage * mu_damage
  expected_eq_struct_loss          = p_ignition_cal * expected_eq_struct_loss_cond_ign
  expected_dollar_loss             = expected_eq_struct_loss * value_per_structure_proxy

Fallback mode (when DINS linkage is unavailable):
  - Train on FRAP perimeter burned-area overlap by segment-event.
  - Produce expected_burned_acres and convert to proxy structure/dollar loss.
  - Log assumptions explicitly in metrics.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fiona
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# CONFIGURABLE PATHS
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Existing ignition outputs
PANEL_FULL_CSV = OUTDIR / "panel_full.csv"
PANEL_TEST_CSV = OUTDIR / "panel_predictions_test.csv"
SEGMENT_SEASONAL_RISK_CSV = OUTDIR / "segment_seasonal_risk.csv"

# CAL FIRE damage inspection data (DINS) - can be CSV / GeoJSON / GDB / ZIP
DINS_PATH = HERE / "loss_function_data" / "CAL_FIRE_Wildfire_Damage_Inspection_Template.zip"

# FRAP perimeter data - can be direct GDB or parent folder containing nested GDB
FRAP_PATH = HERE / "loss_function_data" / "frap.gbd"

# ACS table exports
ACS_TABLE_PATH = HERE / "Electrical_Grid" / "acs" / "ca_acs_tract_2023.json"

# TIGER polygons (optional, recommended)
# If absent, script uses robust fallback exposure allocation.
CENSUS_TRACTS_PATH = HERE / "loss_function_data" / "tl_2023_06_tract" / "tl_2023_06_tract.shp"
CENSUS_BLOCKGROUPS_PATH = HERE / "loss_function_data" / "tl_2023_06_bg" / "tl_2023_06_bg.shp"

# Optional MTBS severity data
MTBS_PERIM_ZIP = HERE / "loss_function_data" / "mtbs_perimeter_data.zip"
MTBS_POINTS_ZIP = HERE / "loss_function_data" / "mtbs_fod_pts_data.zip"

# Existing structures summary fallback
STRUCTURES_BY_CLUSTER_CSV = OUTDIR / "structures_by_cluster.csv"


# -----------------------------------------------------------------------------
# MODEL / PIPELINE CONSTANTS
# -----------------------------------------------------------------------------

TRAIN_END_YEAR = 2022
VAL_YEAR = 2023
TEST_YEAR = 2024

FIRE_SEASON_MONTHS = (5, 6, 7, 8, 9, 10)  # May-Oct
RANDOM_SEED = 42

EXPOSURE_BUFFER_KM = 10.0
PERIMETER_CLUSTER_BUFFER_KM = 12.0

# If ACS home value is missing, these keep proxy values bounded.
MIN_HOME_VALUE_PROXY = 100_000.0
MAX_HOME_VALUE_PROXY = 3_000_000.0
HOME_VALUE_TO_REPLACEMENT_MULT = 1.10

# Fallback conversion when structure damage labels are unavailable.
BURNED_ACRE_TO_EQ_STRUCT_FACTOR = 0.30

TOPK_SEGMENT_SHARE = 0.10
MITIGATION_REDUCTION_PCTS = (0.10, 0.25, 0.50)

SEVERITY_NUMERIC_FEATURES = [
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
    "housing_units",
    "population",
    "median_home_value",
    "median_income",
    "value_per_structure_proxy",
]

SEVERITY_CATEGORICAL_FEATURES = [
    "season",
    "fuel_category",
    "hftd_tier",
]


@dataclass
class SeverityArtifacts:
    mode: str
    classifier: Pipeline | None
    regressor: Pipeline
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    metrics: dict[str, Any]
    diagnostics: dict[str, Any]
    feature_importance: pd.DataFrame


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------


def normalize_name(value: Any) -> str:
    txt = str(value or "").upper().strip()
    txt = re.sub(r"[^A-Z0-9]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def parse_dt(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if isinstance(dt.dtype, pd.DatetimeTZDtype):  # pragma: no cover (tz-variant by source)
        dt = dt.dt.tz_convert(None)
    return dt.dt.tz_localize(None)


def pick_col(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    m = v.notna() & w.notna() & (w > 0)
    if not m.any():
        return float("nan")
    return float(np.average(v[m], weights=w[m]))


def make_preprocessor(num_features: list[str], cat_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_features,
            ),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer, num_features: list[str], cat_features: list[str]) -> list[str]:
    cat_names = list(
        preprocessor.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(cat_features)
    )
    return list(num_features) + cat_names


def predict_in_chunks(model: Pipeline, X: pd.DataFrame, proba: bool, chunk_size: int = 200_000) -> np.ndarray:
    out: list[np.ndarray] = []
    n = len(X)
    for i in range(0, n, chunk_size):
        sl = X.iloc[i : i + chunk_size]
        if proba:
            out.append(model.predict_proba(sl)[:, 1])
        else:
            out.append(model.predict(sl))
    return np.concatenate(out, axis=0) if out else np.array([])


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i < n_bins - 1:
            m = (p >= lo) & (p < hi)
        else:
            m = (p >= lo) & (p <= hi)
        if not m.any():
            continue
        obs = float(y[m].mean())
        pred = float(p[m].mean())
        frac = float(m.mean())
        ece += frac * abs(obs - pred)
        rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": int(m.sum()), "pred_mean": pred, "obs_rate": obs})
    return float(ece), pd.DataFrame(rows)


def choose_frap_gdb(path_like: Path) -> Path:
    if path_like.suffix.lower() == ".gdb" and path_like.exists():
        return path_like
    if path_like.exists() and path_like.is_dir():
        nested = sorted(path_like.rglob("*.gdb"))
        if nested:
            return nested[0]
    raise FileNotFoundError(f"FRAP geodatabase not found under: {path_like}")


def load_panel_full(path: Path = PANEL_FULL_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required panel file: {path}")
    log.info(f"Loading panel: {path}")
    panel = pd.read_csv(path, low_memory=False)
    panel["date"] = parse_dt(panel["date"]).dt.normalize()
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    panel["month"] = pd.to_numeric(panel["month"], errors="coerce").astype("Int64")
    panel["cluster_id"] = pd.to_numeric(panel["cluster_id"], errors="coerce").astype("Int64")
    panel = panel[panel["cluster_id"].notna() & panel["date"].notna()].copy()
    panel["cluster_id"] = panel["cluster_id"].astype(int)
    log.info(f"Panel loaded: {len(panel):,} rows, {panel['cluster_id'].nunique()} clusters")
    return panel


def extract_centroids(panel_full: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "cluster_id",
        "cluster_lat",
        "cluster_lon",
        "fuel_category",
        "elevation_m",
        "slope_deg",
        "hftd_tier",
    ]
    cols = [c for c in keep_cols if c in panel_full.columns]
    centroids = panel_full[cols].drop_duplicates(subset=["cluster_id"]).copy()
    for c in keep_cols:
        if c not in centroids.columns:
            centroids[c] = np.nan
    return centroids.sort_values("cluster_id").reset_index(drop=True)


# -----------------------------------------------------------------------------
# EXPOSURE BUILD
# -----------------------------------------------------------------------------


def load_acs_table(path: Path = ACS_TABLE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ACS table not found: {path}")

    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            header = raw[0]
            rows = raw[1:]
            acs = pd.DataFrame(rows, columns=header)
        else:
            raise ValueError(f"Unsupported ACS JSON layout in {path}")
    else:
        acs = pd.read_csv(path)

    col_map = {
        "B01003_001E": "population",
        "B19013_001E": "median_income",
        "B25077_001E": "median_home_value",
        "B25002_001E": "housing_units",
    }
    for src, dst in col_map.items():
        if src in acs.columns and dst not in acs.columns:
            acs[dst] = acs[src]

    for c in ["state", "county", "tract"]:
        if c not in acs.columns:
            raise ValueError(f"ACS table missing key field: {c}")
        acs[c] = acs[c].astype(str).str.strip()
    acs["state"] = acs["state"].str.zfill(2)
    acs["county"] = acs["county"].str.zfill(3)
    acs["tract"] = acs["tract"].str.zfill(6)
    acs["geoid"] = acs["state"] + acs["county"] + acs["tract"]

    for c in ["housing_units", "population", "median_home_value", "median_income"]:
        if c not in acs.columns:
            acs[c] = np.nan
        acs[c] = pd.to_numeric(acs[c], errors="coerce")
        acs.loc[acs[c] < 0, c] = np.nan

    log.info(
        "ACS loaded: %s rows | housing non-null %.1f%% | home value non-null %.1f%%",
        len(acs),
        100.0 * acs["housing_units"].notna().mean(),
        100.0 * acs["median_home_value"].notna().mean(),
    )
    return acs


def load_census_polygons() -> gpd.GeoDataFrame | None:
    path = CENSUS_BLOCKGROUPS_PATH if CENSUS_BLOCKGROUPS_PATH.exists() else CENSUS_TRACTS_PATH
    if not path.exists():
        log.warning("No TIGER tract/block-group polygons found. Using exposure fallback mode.")
        return None

    gdf = gpd.read_file(path)
    if gdf.empty:
        log.warning("Census polygon file is empty. Using exposure fallback mode.")
        return None

    geoid_col = pick_col(
        list(gdf.columns),
        ["GEOID", "GEOID10", "GEOID20", "GEOIDFQ", "geoid"],
    )
    if geoid_col is None:
        state_col = pick_col(list(gdf.columns), ["STATEFP", "STATEFP10", "statefp"])
        county_col = pick_col(list(gdf.columns), ["COUNTYFP", "COUNTYFP10", "countyfp"])
        tract_col = pick_col(list(gdf.columns), ["TRACTCE", "TRACTCE10", "tractce", "BLKGRPCE", "blkgrpce"])
        if state_col and county_col and tract_col:
            gdf["geoid"] = (
                gdf[state_col].astype(str).str.zfill(2)
                + gdf[county_col].astype(str).str.zfill(3)
                + gdf[tract_col].astype(str).str.zfill(6)
            )
        else:
            log.warning("Could not derive GEOID from census polygons. Using exposure fallback mode.")
            return None
    else:
        gdf["geoid"] = gdf[geoid_col].astype(str).str.extract(r"(\d{11,12})", expand=False)

    gdf = gdf[gdf["geoid"].notna()].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    return gdf[["geoid", "geometry"]]


def _exposure_from_polygons(
    centroids: pd.DataFrame, acs: pd.DataFrame, census_poly: gpd.GeoDataFrame
) -> pd.DataFrame:
    tracts = census_poly.merge(
        acs[["geoid", "housing_units", "population", "median_home_value", "median_income"]],
        on="geoid",
        how="left",
    )
    tracts = tracts.to_crs(epsg=3310)
    tracts["tract_area_m2"] = tracts.geometry.area
    tracts = tracts[tracts["tract_area_m2"] > 0].copy()

    cent = gpd.GeoDataFrame(
        centroids[["cluster_id", "cluster_lat", "cluster_lon"]].copy(),
        geometry=gpd.points_from_xy(centroids["cluster_lon"], centroids["cluster_lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3310)
    buffers = cent[["cluster_id", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(EXPOSURE_BUFFER_KM * 1000.0)
    buffer_area_ac = math.pi * (EXPOSURE_BUFFER_KM * 1000.0) ** 2 / 4046.8564224

    pairs = gpd.sjoin(
        tracts[["geoid", "housing_units", "population", "median_home_value", "median_income", "tract_area_m2", "geometry"]],
        buffers[["cluster_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    if pairs.empty:
        raise ValueError("No census polygon intersects with cluster buffers.")

    buf_geom = buffers.geometry
    inter_area = []
    for geom, idx_buf in zip(pairs.geometry.values, pairs["index_right"].values):
        inter_area.append(float(geom.intersection(buf_geom.iloc[int(idx_buf)]).area))
    pairs["inter_area_m2"] = inter_area
    pairs = pairs[pairs["inter_area_m2"] > 0].copy()
    pairs["alloc_w"] = (pairs["inter_area_m2"] / pairs["tract_area_m2"]).clip(lower=0.0, upper=1.0)
    pairs["alloc_housing"] = pairs["housing_units"] * pairs["alloc_w"]
    pairs["alloc_population"] = pairs["population"] * pairs["alloc_w"]

    grp = pairs.groupby("cluster_id", as_index=False).agg(
        housing_units=("alloc_housing", "sum"),
        population=("alloc_population", "sum"),
    )

    hm = (
        pairs.groupby("cluster_id")
        .apply(lambda d: weighted_mean(d["median_home_value"], d["alloc_housing"]))
        .rename("median_home_value")
        .reset_index()
    )
    im = (
        pairs.groupby("cluster_id")
        .apply(lambda d: weighted_mean(d["median_income"], d["alloc_population"]))
        .rename("median_income")
        .reset_index()
    )

    out = centroids[["cluster_id"]].merge(grp, on="cluster_id", how="left")
    out = out.merge(hm, on="cluster_id", how="left").merge(im, on="cluster_id", how="left")
    out["structures_per_acre_proxy"] = out["housing_units"] / float(buffer_area_ac)
    out["exposure_method"] = "acs_polygon_overlap"
    return out


def _exposure_fallback(centroids: pd.DataFrame, acs: pd.DataFrame) -> pd.DataFrame:
    out = centroids[["cluster_id"]].copy()

    if STRUCTURES_BY_CLUSTER_CSV.exists():
        st = pd.read_csv(STRUCTURES_BY_CLUSTER_CSV)
        st["cluster_id"] = pd.to_numeric(st["cluster_id"], errors="coerce").astype("Int64")
        st = st[st["cluster_id"].notna()].copy()
        st["cluster_id"] = st["cluster_id"].astype(int)
        st["structures_per_sqmi"] = pd.to_numeric(st["structures_per_sqmi"], errors="coerce")
        out = out.merge(st[["cluster_id", "structures_per_sqmi"]], on="cluster_id", how="left")
        w = out["structures_per_sqmi"].fillna(out["structures_per_sqmi"].median())
    else:
        out["structures_per_sqmi"] = np.nan
        w = pd.Series(np.ones(len(out)), index=out.index)

    w = w.clip(lower=0)
    if float(w.sum()) <= 0:
        w = pd.Series(np.ones(len(out)), index=out.index)
    w = w / float(w.sum())

    total_housing = float(pd.to_numeric(acs["housing_units"], errors="coerce").sum(skipna=True))
    total_pop = float(pd.to_numeric(acs["population"], errors="coerce").sum(skipna=True))
    state_home = weighted_mean(acs["median_home_value"], acs["housing_units"])
    state_income = weighted_mean(acs["median_income"], acs["population"])

    out["housing_units"] = total_housing * w
    out["population"] = total_pop * w
    out["median_home_value"] = state_home
    out["median_income"] = state_income

    # Approximate structures per acre from 1 sqmi = 640 acres.
    out["structures_per_acre_proxy"] = out["structures_per_sqmi"].fillna(out["housing_units"] / 640.0) / 640.0
    out["exposure_method"] = "acs_statewide_weighted_fallback"
    return out


def build_segment_exposure(panel_full: pd.DataFrame) -> pd.DataFrame:
    """
    Build segment exposure table.

    Returns at minimum:
      cluster_id, housing_units, population, median_home_value, median_income,
      value_per_structure_proxy, segment_exposure_value_proxy
    """
    log.info("STEP 1: Build segment exposure table")
    centroids = extract_centroids(panel_full)
    acs = load_acs_table(ACS_TABLE_PATH)
    census_poly = load_census_polygons()

    if census_poly is not None:
        try:
            exposure = _exposure_from_polygons(centroids, acs, census_poly)
        except Exception as e:
            log.warning(f"Polygon-based exposure failed ({e}); switching to fallback.")
            exposure = _exposure_fallback(centroids, acs)
    else:
        exposure = _exposure_fallback(centroids, acs)

    exposure["housing_units"] = pd.to_numeric(exposure["housing_units"], errors="coerce")
    exposure["population"] = pd.to_numeric(exposure["population"], errors="coerce")
    exposure["median_home_value"] = pd.to_numeric(exposure["median_home_value"], errors="coerce")
    exposure["median_income"] = pd.to_numeric(exposure["median_income"], errors="coerce")

    global_home = weighted_mean(acs["median_home_value"], acs["housing_units"])
    if not np.isfinite(global_home):
        global_home = 550_000.0
    exposure["median_home_value"] = exposure["median_home_value"].fillna(global_home)
    exposure["median_home_value"] = exposure["median_home_value"].clip(
        lower=MIN_HOME_VALUE_PROXY,
        upper=MAX_HOME_VALUE_PROXY,
    )
    exposure["median_income"] = exposure["median_income"].fillna(weighted_mean(acs["median_income"], acs["population"]))

    exposure["value_per_structure_proxy"] = exposure["median_home_value"] * HOME_VALUE_TO_REPLACEMENT_MULT
    exposure["segment_exposure_value_proxy"] = exposure["housing_units"] * exposure["value_per_structure_proxy"]
    exposure["cluster_id"] = pd.to_numeric(exposure["cluster_id"], errors="coerce").astype("Int64")
    exposure = exposure[exposure["cluster_id"].notna()].copy()
    exposure["cluster_id"] = exposure["cluster_id"].astype(int)

    coverage = {
        "housing_units_nonnull_pct": 100.0 * exposure["housing_units"].notna().mean(),
        "population_nonnull_pct": 100.0 * exposure["population"].notna().mean(),
        "home_value_nonnull_pct": 100.0 * exposure["median_home_value"].notna().mean(),
        "value_proxy_nonnull_pct": 100.0 * exposure["value_per_structure_proxy"].notna().mean(),
    }
    log.info(f"Exposure coverage: {coverage}")
    return exposure.sort_values("cluster_id").reset_index(drop=True)


# -----------------------------------------------------------------------------
# FIRE DATA LOAD + LINKAGE
# -----------------------------------------------------------------------------


def load_frap_perimeters() -> gpd.GeoDataFrame:
    gdb = choose_frap_gdb(FRAP_PATH)
    layers = fiona.listlayers(gdb)
    layer = layers[0]
    log.info(f"Loading FRAP perimeters: {gdb.name} | layer={layer}")
    frap = gpd.read_file(gdb, layer=layer)

    frap["year"] = pd.to_numeric(frap.get("YEAR_"), errors="coerce").astype("Int64")
    if "ALARM_DATE" in frap.columns:
        frap["event_date"] = parse_dt(frap["ALARM_DATE"]).dt.normalize()
    else:
        frap["event_date"] = pd.NaT
    frap["event_date"] = frap["event_date"].fillna(
        pd.to_datetime(frap["year"].astype(str) + "-07-01", errors="coerce")
    )

    frap = frap[(frap["year"] >= 2014) & (frap["year"] <= 2024)].copy()
    frap["gis_acres"] = pd.to_numeric(frap.get("GIS_ACRES"), errors="coerce")
    frap["gis_acres"] = frap["gis_acres"].fillna((frap.geometry.to_crs(3310).area / 4046.8564224))
    frap["fire_name"] = frap.get("FIRE_NAME", "").astype(str)
    frap["fire_name_norm"] = frap["fire_name"].map(normalize_name)
    frap["frap_id"] = np.arange(len(frap), dtype=int)
    if frap.crs is None:
        frap = frap.set_crs(epsg=3857, allow_override=True)
    frap = frap.to_crs(epsg=3310)

    cols = ["frap_id", "year", "event_date", "fire_name", "fire_name_norm", "gis_acres", "geometry"]
    frap = frap[cols].copy()
    log.info(f"FRAP rows retained (2014-2024): {len(frap):,}")
    return frap


def assign_perimeters_to_clusters_by_overlap(
    frap: gpd.GeoDataFrame, centroids: pd.DataFrame
) -> pd.DataFrame:
    cent = gpd.GeoDataFrame(
        centroids[["cluster_id", "cluster_lat", "cluster_lon"]].copy(),
        geometry=gpd.points_from_xy(centroids["cluster_lon"], centroids["cluster_lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3310)
    buffers = cent[["cluster_id", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(PERIMETER_CLUSTER_BUFFER_KM * 1000.0)

    pairs = gpd.sjoin(
        frap[["frap_id", "year", "event_date", "fire_name", "fire_name_norm", "gis_acres", "geometry"]],
        buffers[["cluster_id", "geometry"]],
        how="left",
        predicate="intersects",
    )

    overlap = pairs[pairs["index_right"].notna()].copy()
    if not overlap.empty:
        buf_geom = buffers.geometry
        inter_areas = []
        for geom, idxb in zip(overlap.geometry.values, overlap["index_right"].values):
            inter_areas.append(float(geom.intersection(buf_geom.iloc[int(idxb)]).area))
        overlap["inter_area_m2"] = inter_areas
        overlap = overlap[overlap["inter_area_m2"] > 0].copy()
        overlap["cluster_id"] = overlap["cluster_id"].astype(int)

        tot = overlap.groupby("frap_id")["inter_area_m2"].transform("sum")
        overlap["overlap_share"] = np.where(tot > 0, overlap["inter_area_m2"] / tot, 0.0)
        overlap["allocated_burned_acres"] = overlap["gis_acres"] * overlap["overlap_share"]
    else:
        overlap = pd.DataFrame(columns=["frap_id", "cluster_id", "overlap_share", "allocated_burned_acres"])

    covered_ids = set(overlap["frap_id"].tolist())
    missing = frap[~frap["frap_id"].isin(covered_ids)].copy()
    if not missing.empty:
        # nearest-cluster fallback for perimeters with no buffer overlap
        cxy = centroids[["cluster_id", "cluster_lat", "cluster_lon"]].copy()
        c_lat = cxy["cluster_lat"].to_numpy(float)
        c_lon = cxy["cluster_lon"].to_numpy(float)
        rows = []
        missing4326 = missing.to_crs(epsg=4326)
        for r in missing4326.itertuples():
            rp = r.geometry.representative_point()
            d2 = (c_lat - float(rp.y)) ** 2 + (c_lon - float(rp.x)) ** 2
            cid = int(cxy.iloc[int(np.argmin(d2))]["cluster_id"])
            rows.append(
                {
                    "frap_id": int(r.frap_id),
                    "cluster_id": cid,
                    "overlap_share": 1.0,
                    "allocated_burned_acres": float(r.gis_acres),
                }
            )
        overlap = pd.concat([overlap, pd.DataFrame(rows)], ignore_index=True)

    # Attach FRAP event metadata without suffix collisions.
    meta = frap.set_index("frap_id")[["year", "event_date", "fire_name", "fire_name_norm", "gis_acres"]]
    overlap["year"] = overlap["frap_id"].map(meta["year"])
    overlap["event_date"] = overlap["frap_id"].map(meta["event_date"])
    overlap["fire_name"] = overlap["frap_id"].map(meta["fire_name"])
    overlap["fire_name_norm"] = overlap["frap_id"].map(meta["fire_name_norm"])
    overlap["gis_acres"] = overlap["frap_id"].map(meta["gis_acres"])
    overlap = overlap.drop(columns=[c for c in ["geometry", "index_right"] if c in overlap.columns], errors="ignore")
    log.info(
        "FRAP->cluster allocation rows: %s (unique events=%s)",
        f"{len(overlap):,}",
        f"{overlap['frap_id'].nunique():,}",
    )
    return overlap


def _extract_zip_once(zip_path: Path, target_root: Path) -> Path:
    target_root.mkdir(parents=True, exist_ok=True)
    marker = target_root / ".extracted_ok"
    if not marker.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(target_root)
        marker.write_text("ok")
    return target_root


def _load_dins_raw(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        log.warning(f"DINS path not found: {path}")
        return gpd.GeoDataFrame()

    p = path
    if path.suffix.lower() == ".zip":
        extracted = _extract_zip_once(path, OUTDIR / "_dins_extract")
        gdbs = sorted(extracted.rglob("*.gdb"))
        if not gdbs:
            log.warning("DINS zip extracted but no .gdb found.")
            return gpd.GeoDataFrame()
        p = gdbs[0]

    if p.suffix.lower() == ".gdb":
        layers = fiona.listlayers(p)
        layer = next((ly for ly in layers if not ly.endswith("__ATTACH")), layers[0] if layers else None)
        if layer is None:
            return gpd.GeoDataFrame()
        gdf = gpd.read_file(p, layer=layer)
        if not isinstance(gdf, gpd.GeoDataFrame):
            return gpd.GeoDataFrame(gdf)
        return gdf

    if p.suffix.lower() in {".geojson", ".json", ".gpkg", ".shp"}:
        return gpd.read_file(p)

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        lat_col = pick_col(list(df.columns), ["latitude", "lat", "y"])
        lon_col = pick_col(list(df.columns), ["longitude", "lon", "lng", "x"])
        if lat_col and lon_col:
            return gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(pd.to_numeric(df[lon_col], errors="coerce"), pd.to_numeric(df[lat_col], errors="coerce")),
                crs="EPSG:4326",
            )
        return gpd.GeoDataFrame(df)

    log.warning(f"Unsupported DINS source type: {p}")
    return gpd.GeoDataFrame()


def _assign_points_to_clusters(gdf: gpd.GeoDataFrame, centroids: pd.DataFrame) -> pd.Series:
    c_lat = centroids["cluster_lat"].to_numpy(float)
    c_lon = centroids["cluster_lon"].to_numpy(float)
    c_id = centroids["cluster_id"].to_numpy(int)
    assigned = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            assigned.append(np.nan)
            continue
        rp = geom.representative_point()
        d2 = (c_lat - float(rp.y)) ** 2 + (c_lon - float(rp.x)) ** 2
        assigned.append(int(c_id[int(np.argmin(d2))]))
    return pd.Series(assigned, index=gdf.index, dtype="float")


def load_dins_aggregated(centroids: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = _load_dins_raw(DINS_PATH)
    stats: dict[str, Any] = {"dins_rows_raw": int(len(raw))}
    if raw.empty:
        stats["dins_available"] = False
        return pd.DataFrame(columns=["cluster_id", "event_date", "year", "incident_name_norm", "structures_destroyed", "structures_damaged"]), stats

    raw = raw.copy()
    if raw.crs is None:
        raw = raw.set_crs(epsg=3857, allow_override=True)
    raw = raw.to_crs(epsg=4326)

    cols = list(raw.columns)
    name_col = pick_col(cols, ["INCIDENTNAME", "FIRENAME", "INCIDENT_NAME", "FIRE_NAME", "INCIDENT"])
    date_col = pick_col(cols, ["INCIDENTSTARTDATE", "INSPECTIONDATE", "DATE", "EVENT_DATE", "ALARM_DATE"])
    damage_col = pick_col(cols, ["DAMAGE", "DAMAGE_STATUS", "DAMAGESTATE", "damage"])
    destroyed_col = pick_col(cols, ["STRUCTURES_DESTROYED", "NO_DESTROYED", "DESTROYED"])
    damaged_col = pick_col(cols, ["STRUCTURES_DAMAGED", "NO_DAMAGED", "DAMAGED"])

    if date_col is None:
        raw["event_date"] = pd.NaT
    else:
        raw["event_date"] = parse_dt(raw[date_col]).dt.normalize()

    raw["year"] = pd.to_numeric(raw["event_date"].dt.year, errors="coerce").astype("Int64")
    raw = raw[(raw["year"] >= 2014) & (raw["year"] <= 2024)].copy()

    if name_col is None:
        raw["incident_name_norm"] = ""
    else:
        raw["incident_name_norm"] = raw[name_col].map(normalize_name)

    if destroyed_col and damaged_col:
        raw["structures_destroyed"] = pd.to_numeric(raw[destroyed_col], errors="coerce").fillna(0.0)
        raw["structures_damaged"] = pd.to_numeric(raw[damaged_col], errors="coerce").fillna(0.0)
    else:
        txt = raw[damage_col].astype(str).str.lower() if damage_col else pd.Series("", index=raw.index)
        raw["structures_destroyed"] = txt.str.contains(r"destroy", regex=True, na=False).astype(float)
        raw["structures_damaged"] = txt.str.contains(r"damag|major|minor|affected", regex=True, na=False).astype(float)

    raw["cluster_id"] = _assign_points_to_clusters(raw, centroids)
    raw = raw[raw["cluster_id"].notna() & raw["event_date"].notna()].copy()
    raw["cluster_id"] = raw["cluster_id"].astype(int)

    agg = (
        raw.groupby(["cluster_id", "event_date", "year", "incident_name_norm"], as_index=False)[
            ["structures_destroyed", "structures_damaged"]
        ]
        .sum()
    )
    agg["eq_struct_loss"] = agg["structures_destroyed"] + 0.25 * agg["structures_damaged"]
    stats["dins_available"] = True
    stats["dins_rows_after_clean"] = int(len(raw))
    stats["dins_event_cluster_rows"] = int(len(agg))
    stats["dins_positive_eq_rows"] = int((agg["eq_struct_loss"] > 0).sum())
    return agg, stats


def link_dins_to_frap_and_allocate(
    dins_agg: pd.DataFrame, frap_cluster: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Preferred geometry strategy:
      - match DINS incidents to FRAP perimeters via normalized name + year
      - distribute DINS structure loss to clusters by FRAP overlap share
    Fallback:
      - if no FRAP match, keep DINS nearest-cluster assignment
    """
    stats: dict[str, Any] = {}
    if dins_agg.empty:
        stats["dins_link_rate_pct"] = 0.0
        return dins_agg, stats

    events = (
        frap_cluster[["frap_id", "year", "event_date", "fire_name_norm", "gis_acres"]]
        .drop_duplicates("frap_id")
        .copy()
    )
    events["fire_name_norm"] = events["fire_name_norm"].fillna("").map(normalize_name)

    work = dins_agg.reset_index(drop=False).rename(columns={"index": "dins_row_id"})
    cand = work.merge(
        events,
        left_on=["incident_name_norm", "year"],
        right_on=["fire_name_norm", "year"],
        how="left",
    )
    if not cand.empty:
        cand["date_diff_days"] = (
            (cand["event_date_x"] - cand["event_date_y"]).dt.days.abs().fillna(9999)
        )
        cand = cand.sort_values(["dins_row_id", "date_diff_days", "gis_acres"], ascending=[True, True, False])
        best = cand.drop_duplicates("dins_row_id", keep="first").copy()
    else:
        best = work.copy()
        best["frap_id"] = np.nan

    linked = best["frap_id"].notna()
    stats["dins_link_rate_pct"] = float(100.0 * linked.mean()) if len(best) else 0.0
    stats["dins_linked_rows"] = int(linked.sum())
    stats["dins_unlinked_rows"] = int((~linked).sum())

    # Matched rows: distribute by FRAP perimeter overlap shares
    dist_rows = []
    matched = best[best["frap_id"].notna()].copy()
    if not matched.empty:
        alloc = matched.merge(
            frap_cluster[["frap_id", "cluster_id", "overlap_share"]],
            on="frap_id",
            how="left",
        )
        alloc["overlap_share"] = alloc["overlap_share"].fillna(1.0)
        alloc["structures_destroyed"] = alloc["structures_destroyed"] * alloc["overlap_share"]
        alloc["structures_damaged"] = alloc["structures_damaged"] * alloc["overlap_share"]
        alloc["event_date"] = alloc["event_date_x"].fillna(alloc["event_date_y"])
        dist_rows.append(
            alloc[["cluster_id", "event_date", "year", "structures_destroyed", "structures_damaged"]]
        )

    # Unmatched rows: keep nearest-cluster assignment from DINS points
    un = best[best["frap_id"].isna()].copy()
    if not un.empty:
        un["event_date"] = un["event_date_x"]
        dist_rows.append(un[["cluster_id", "event_date", "year", "structures_destroyed", "structures_damaged"]])

    if dist_rows:
        out = pd.concat(dist_rows, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["cluster_id", "event_date", "year", "structures_destroyed", "structures_damaged"])

    out = (
        out.groupby(["cluster_id", "event_date", "year"], as_index=False)[["structures_destroyed", "structures_damaged"]]
        .sum()
    )
    out["eq_struct_loss"] = out["structures_destroyed"] + 0.25 * out["structures_damaged"]
    return out, stats


def _attach_event_features(events: pd.DataFrame, panel_full: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "cluster_id",
        "date",
        "year",
        "month",
        "season",
        "cluster_lat",
        "cluster_lon",
        "fuel_category",
        "hftd_tier",
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
    keep = [c for c in keep if c in panel_full.columns]
    base = panel_full[keep].copy()
    out = events.merge(
        base,
        left_on=["cluster_id", "event_date"],
        right_on=["cluster_id", "date"],
        how="left",
    )
    if "date" in out.columns:
        out = out.drop(columns=["date"])
    if "year" not in out.columns:
        if "year_x" in out.columns:
            out["year"] = out["year_x"]
        elif "event_date" in out.columns:
            out["year"] = out["event_date"].dt.year
    if "year_y" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").fillna(pd.to_numeric(out["year_y"], errors="coerce"))
    out = out.drop(columns=[c for c in ["year_x", "year_y"] if c in out.columns], errors="ignore")

    # Fill feature misses with cluster-year means (numeric) and cluster mode (categorical).
    num_cols = [c for c in SEVERITY_NUMERIC_FEATURES if c in out.columns]
    cat_cols = [c for c in SEVERITY_CATEGORICAL_FEATURES if c in out.columns]

    cluster_year_means = panel_full.groupby(["cluster_id", "year"])[num_cols].mean().reset_index()
    out = out.merge(cluster_year_means, on=["cluster_id", "year"], how="left", suffixes=("", "_cy"))
    for c in num_cols:
        cy = f"{c}_cy"
        if cy in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(pd.to_numeric(out[cy], errors="coerce"))
            out = out.drop(columns=[cy])

    for c in cat_cols:
        mode_map = (
            panel_full.groupby("cluster_id")[c]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
            .to_dict()
        )
        out[c] = out[c].fillna(out["cluster_id"].map(mode_map))
        out[c] = out[c].fillna("unknown")

    if "season" in out.columns:
        out["season"] = out["season"].fillna(out["event_date"].dt.month.map(
            {1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall", 12: "winter"}
        ))
    return out


def build_severity_training_table(
    panel_full: pd.DataFrame,
    segment_exposure: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build incident-segment training rows with targets:
      structures_destroyed, structures_damaged, eq_struct_loss
    plus ignition-feature columns and exposure columns.
    """
    log.info("STEP 2: Build severity training table")
    centroids = extract_centroids(panel_full)

    frap = load_frap_perimeters()
    frap_cluster = assign_perimeters_to_clusters_by_overlap(frap, centroids)

    dins_agg, dins_stats = load_dins_aggregated(centroids)
    linked_stats: dict[str, Any] = {}
    mode = "hurdle_struct_damage"

    if not dins_agg.empty and (dins_agg["eq_struct_loss"] > 0).sum() > 0:
        damage_rows, linked_stats = link_dins_to_frap_and_allocate(dins_agg, frap_cluster)
        base = (
            frap_cluster.groupby(["cluster_id", "event_date", "year"], as_index=False)
            .agg(
                burned_acres_overlap=("allocated_burned_acres", "sum"),
                n_frap_events=("frap_id", "nunique"),
            )
        )
        train = base.merge(
            damage_rows[["cluster_id", "event_date", "year", "structures_destroyed", "structures_damaged", "eq_struct_loss"]],
            on=["cluster_id", "event_date", "year"],
            how="left",
        )
        train["structures_destroyed"] = train["structures_destroyed"].fillna(0.0)
        train["structures_damaged"] = train["structures_damaged"].fillna(0.0)
        train["eq_struct_loss"] = train["eq_struct_loss"].fillna(
            train["structures_destroyed"] + 0.25 * train["structures_damaged"]
        )
    else:
        mode = "fallback_burned_area_proxy"
        base = (
            frap_cluster.groupby(["cluster_id", "event_date", "year"], as_index=False)
            .agg(
                burned_acres_overlap=("allocated_burned_acres", "sum"),
                n_frap_events=("frap_id", "nunique"),
            )
        )
        train = base.copy()
        train["structures_destroyed"] = np.nan
        train["structures_damaged"] = np.nan
        train["eq_struct_loss"] = np.nan
        log.warning("DINS linkage unavailable or non-positive. Falling back to burned-area proxy mode.")

    train = _attach_event_features(train, panel_full)
    train = train.merge(segment_exposure, on="cluster_id", how="left")

    # Proxy eq_struct_loss in fallback mode.
    if mode == "fallback_burned_area_proxy":
        train["structures_per_acre_proxy"] = pd.to_numeric(train["structures_per_acre_proxy"], errors="coerce").fillna(0.0)
        train["eq_struct_loss"] = (
            pd.to_numeric(train["burned_acres_overlap"], errors="coerce").fillna(0.0)
            * train["structures_per_acre_proxy"]
            * BURNED_ACRE_TO_EQ_STRUCT_FACTOR
        )
        train["structures_destroyed"] = train["eq_struct_loss"]
        train["structures_damaged"] = 0.0

    train["eq_struct_loss"] = pd.to_numeric(train["eq_struct_loss"], errors="coerce").fillna(0.0)
    train["structures_destroyed"] = pd.to_numeric(train["structures_destroyed"], errors="coerce").fillna(0.0)
    train["structures_damaged"] = pd.to_numeric(train["structures_damaged"], errors="coerce").fillna(0.0)
    train["severity_mode"] = mode

    link_summary = {
        "severity_mode": mode,
        "dins_stats": dins_stats,
        "link_stats": linked_stats,
        "n_training_rows": int(len(train)),
        "n_positive_eq_struct_loss": int((train["eq_struct_loss"] > 0).sum()),
    }
    log.info(f"Severity linkage summary: {link_summary}")
    train.attrs["linkage_summary"] = link_summary
    return train


# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------


def _split_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] <= TRAIN_END_YEAR].copy()
    val = df[df["year"] == VAL_YEAR].copy()
    test = df[df["year"] == TEST_YEAR].copy()
    log.info(
        "Split rows: train=%s val=%s test=%s",
        f"{len(train):,}",
        f"{len(val):,}",
        f"{len(test):,}",
    )
    return train, val, test


def _metrics_hurdle(
    y_true: np.ndarray,
    y_any_true: np.ndarray,
    p_any: np.ndarray,
    mu_pred: np.ndarray,
    label: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {"label": label}
    if len(np.unique(y_any_true)) > 1:
        out["auc"] = float(roc_auc_score(y_any_true, p_any))
        out["pr_auc"] = float(average_precision_score(y_any_true, p_any))
    else:
        out["auc"] = None
        out["pr_auc"] = None
    ece, bins = expected_calibration_error(y_any_true, p_any, n_bins=10)
    out["ece"] = float(ece)
    out["calibration_bins"] = bins.to_dict(orient="records")

    pos = y_true > 0
    if pos.any():
        pred_pos = mu_pred[pos]
        obs_pos = y_true[pos]
        out["mae_pos"] = float(mean_absolute_error(obs_pos, pred_pos))
        out["rmse_pos"] = float(math.sqrt(mean_squared_error(obs_pos, pred_pos)))
    else:
        out["mae_pos"] = None
        out["rmse_pos"] = None
    return out


def _feature_importance_from_pipeline(pipe: Pipeline, part: str, feature_names: list[str]) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["model_part", "feature", "importance"])
    imp = model.feature_importances_
    return pd.DataFrame(
        {
            "model_part": part,
            "feature": feature_names,
            "importance": imp,
        }
    ).sort_values("importance", ascending=False)


def train_severity_hurdle_model(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> SeverityArtifacts:
    """
    Train hurdle model:
      A) binary any-damage classifier
      B) positive-loss regressor (log1p target)

    If structure-loss positives are insufficient, fallback to burned-area proxy
    regressor mode and log this explicitly.
    """
    mode = str(train_df["severity_mode"].iloc[0]) if not train_df.empty else "fallback_burned_area_proxy"
    num_cols = [c for c in SEVERITY_NUMERIC_FEATURES if c in train_df.columns]
    cat_cols = [c for c in SEVERITY_CATEGORICAL_FEATURES if c in train_df.columns]
    feature_cols = num_cols + cat_cols
    for df in [train_df, val_df, test_df]:
        for c in feature_cols:
            if c not in df.columns:
                df[c] = np.nan

    y_tr = pd.to_numeric(train_df["eq_struct_loss"], errors="coerce").fillna(0.0).to_numpy()
    y_va = pd.to_numeric(val_df["eq_struct_loss"], errors="coerce").fillna(0.0).to_numpy()
    y_te = pd.to_numeric(test_df["eq_struct_loss"], errors="coerce").fillna(0.0).to_numpy()
    y_any_tr = (y_tr > 0).astype(int)
    y_any_va = (y_va > 0).astype(int)
    y_any_te = (y_te > 0).astype(int)

    n_pos = int(y_any_tr.sum())
    log.info(f"Severity train positives (eq_struct_loss>0): {n_pos:,}")

    diagnostics: dict[str, Any] = {}
    metrics: dict[str, Any] = {"mode": mode}
    importance_frames: list[pd.DataFrame] = []

    # Fallback model if positives are too sparse or fallback mode already selected.
    if mode == "fallback_burned_area_proxy" or n_pos < 25:
        if mode != "fallback_burned_area_proxy":
            log.warning("Positive structure-loss rows too few for hurdle model; switching to burned-area proxy fallback.")
        mode = "fallback_burned_area_proxy"
        target_col = "burned_acres_overlap" if "burned_acres_overlap" in train_df.columns else "eq_struct_loss"
        ytr = np.log1p(pd.to_numeric(train_df[target_col], errors="coerce").fillna(0.0).to_numpy())
        yva = pd.to_numeric(val_df[target_col], errors="coerce").fillna(0.0).to_numpy()
        yte = pd.to_numeric(test_df[target_col], errors="coerce").fillna(0.0).to_numpy()

        pre = make_preprocessor(num_cols, cat_cols)
        reg = RandomForestRegressor(
            n_estimators=350,
            max_depth=12,
            min_samples_leaf=6,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        reg_pipe = Pipeline([("pre", pre), ("model", reg)])
        reg_pipe.fit(train_df[feature_cols], ytr)

        pred_va = np.expm1(predict_in_chunks(reg_pipe, val_df[feature_cols], proba=False))
        pred_te = np.expm1(predict_in_chunks(reg_pipe, test_df[feature_cols], proba=False))
        metrics["val"] = {
            "mae_target": float(mean_absolute_error(yva, pred_va)) if len(val_df) else None,
            "rmse_target": float(math.sqrt(mean_squared_error(yva, pred_va))) if len(val_df) else None,
        }
        metrics["test"] = {
            "mae_target": float(mean_absolute_error(yte, pred_te)) if len(test_df) else None,
            "rmse_target": float(math.sqrt(mean_squared_error(yte, pred_te))) if len(test_df) else None,
        }

        f_names = get_feature_names(pre, num_cols, cat_cols)
        importance_frames.append(_feature_importance_from_pipeline(reg_pipe, "regressor_fallback_burned_area", f_names))
        diagnostics["p_any_test"] = np.ones(len(test_df), dtype=float)
        diagnostics["y_any_test"] = (y_te > 0).astype(int)
        diagnostics["pred_eq_cond_ign_test"] = pred_te

        return SeverityArtifacts(
            mode=mode,
            classifier=None,
            regressor=reg_pipe,
            feature_columns=feature_cols,
            numeric_features=num_cols,
            categorical_features=cat_cols,
            metrics=metrics,
            diagnostics=diagnostics,
            feature_importance=pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame(),
        )

    # Standard hurdle mode.
    pre_clf = make_preprocessor(num_cols, cat_cols)
    clf = RandomForestClassifier(
        n_estimators=350,
        max_depth=10,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf_pipe = Pipeline([("pre", pre_clf), ("model", clf)])
    clf_pipe.fit(train_df[feature_cols], y_any_tr)

    pos_train = train_df[y_any_tr == 1].copy()
    pre_reg = make_preprocessor(num_cols, cat_cols)
    reg = RandomForestRegressor(
        n_estimators=350,
        max_depth=12,
        min_samples_leaf=6,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    reg_pipe = Pipeline([("pre", pre_reg), ("model", reg)])
    reg_pipe.fit(pos_train[feature_cols], np.log1p(pos_train["eq_struct_loss"].to_numpy()))

    p_any_va = predict_in_chunks(clf_pipe, val_df[feature_cols], proba=True)
    p_any_te = predict_in_chunks(clf_pipe, test_df[feature_cols], proba=True)
    mu_va = np.expm1(predict_in_chunks(reg_pipe, val_df[feature_cols], proba=False))
    mu_te = np.expm1(predict_in_chunks(reg_pipe, test_df[feature_cols], proba=False))

    metrics["val"] = _metrics_hurdle(y_va, y_any_va, p_any_va, mu_va, "val")
    metrics["test"] = _metrics_hurdle(y_te, y_any_te, p_any_te, mu_te, "test")

    f_clf = get_feature_names(pre_clf, num_cols, cat_cols)
    f_reg = get_feature_names(pre_reg, num_cols, cat_cols)
    importance_frames.append(_feature_importance_from_pipeline(clf_pipe, "classifier_any_damage", f_clf))
    importance_frames.append(_feature_importance_from_pipeline(reg_pipe, "regressor_positive_damage", f_reg))

    diagnostics["p_any_test"] = p_any_te
    diagnostics["y_any_test"] = y_any_te
    diagnostics["pred_eq_cond_ign_test"] = p_any_te * mu_te
    diagnostics["obs_eq_test"] = y_te

    return SeverityArtifacts(
        mode="hurdle_struct_damage",
        classifier=clf_pipe,
        regressor=reg_pipe,
        feature_columns=feature_cols,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        metrics=metrics,
        diagnostics=diagnostics,
        feature_importance=pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame(),
    )


# -----------------------------------------------------------------------------
# SCORING
# -----------------------------------------------------------------------------


def score_expected_loss_panel(
    panel_full: pd.DataFrame,
    segment_exposure: pd.DataFrame,
    severity_models: SeverityArtifacts,
) -> pd.DataFrame:
    """
    Score expected loss on every cluster-day panel row.
    """
    log.info("STEP 4: Score expected loss on full panel")
    out = panel_full.copy()
    out = out.merge(
        segment_exposure[
            [
                "cluster_id",
                "housing_units",
                "population",
                "median_home_value",
                "median_income",
                "value_per_structure_proxy",
                "structures_per_acre_proxy",
            ]
        ],
        on="cluster_id",
        how="left",
    )

    for c in severity_models.feature_columns:
        if c not in out.columns:
            out[c] = np.nan

    X = out[severity_models.feature_columns].copy()
    if severity_models.mode == "hurdle_struct_damage" and severity_models.classifier is not None:
        p_any = predict_in_chunks(severity_models.classifier, X, proba=True)
        mu = np.expm1(predict_in_chunks(severity_models.regressor, X, proba=False))
        eq_cond = p_any * mu
    else:
        pred_burned = np.expm1(predict_in_chunks(severity_models.regressor, X, proba=False))
        spa = pd.to_numeric(out["structures_per_acre_proxy"], errors="coerce").fillna(0.0).to_numpy()
        p_any = np.ones(len(out), dtype=float)
        eq_cond = pred_burned * spa * BURNED_ACRE_TO_EQ_STRUCT_FACTOR
        out["expected_burned_acres_cond_ign"] = pred_burned

    out["p_any_damage_cond_ign"] = p_any
    out["expected_eq_struct_loss_cond_ign"] = np.maximum(eq_cond, 0.0)
    out["expected_eq_struct_loss"] = (
        pd.to_numeric(out["p_ignition_cal"], errors="coerce").fillna(0.0)
        * out["expected_eq_struct_loss_cond_ign"]
    )
    out["value_per_structure_proxy"] = pd.to_numeric(out["value_per_structure_proxy"], errors="coerce").fillna(0.0)
    out["expected_dollar_loss"] = out["expected_eq_struct_loss"] * out["value_per_structure_proxy"]
    return out


# -----------------------------------------------------------------------------
# AGGREGATION + MITIGATION
# -----------------------------------------------------------------------------


def _aggregate_loss_outputs(panel_loss: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_loss = panel_loss.copy()
    panel_loss["season_bucket"] = np.where(panel_loss["month"].isin(FIRE_SEASON_MONTHS), "fire_season", "off_season")

    seasonal = (
        panel_loss.groupby(["cluster_id", "year", "season_bucket"], as_index=False)
        .agg(
            sum_expected_dollar_loss=("expected_dollar_loss", "sum"),
            sum_expected_structure_loss=("expected_eq_struct_loss", "sum"),
            max_daily_expected_dollar_loss=("expected_dollar_loss", "max"),
            mean_daily_p_ignition=("p_ignition_cal", "mean"),
        )
    )
    seasonal["rank_within_year_season"] = (
        seasonal.groupby(["year", "season_bucket"])["sum_expected_dollar_loss"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    annual = (
        panel_loss.groupby(["cluster_id", "year"], as_index=False)
        .agg(
            sum_expected_dollar_loss=("expected_dollar_loss", "sum"),
            sum_expected_structure_loss=("expected_eq_struct_loss", "sum"),
            max_daily_expected_dollar_loss=("expected_dollar_loss", "max"),
            mean_daily_p_ignition=("p_ignition_cal", "mean"),
            hftd_tier=("hftd_tier", lambda s: s.mode().iloc[0] if len(s.mode()) else "unknown"),
            cluster_lat=("cluster_lat", "first"),
            cluster_lon=("cluster_lon", "first"),
        )
    )
    annual["rank_within_year"] = annual.groupby("year")["sum_expected_dollar_loss"].rank(ascending=False, method="min").astype(int)
    return seasonal, annual


def simulate_mitigation(panel_loss: pd.DataFrame, target_segments: list[int], ignition_reduction_pct: float) -> dict[str, float]:
    """
    Reduce p_ignition_cal for selected segments and compute avoided expected loss.
    """
    base_eq = float(panel_loss["expected_eq_struct_loss"].sum())
    base_usd = float(panel_loss["expected_dollar_loss"].sum())

    p_adj = pd.to_numeric(panel_loss["p_ignition_cal"], errors="coerce").fillna(0.0).to_numpy(copy=True)
    mask = panel_loss["cluster_id"].isin(target_segments).to_numpy()
    p_adj[mask] = p_adj[mask] * max(0.0, 1.0 - float(ignition_reduction_pct))

    eq_cond = pd.to_numeric(panel_loss["expected_eq_struct_loss_cond_ign"], errors="coerce").fillna(0.0).to_numpy()
    vps = pd.to_numeric(panel_loss["value_per_structure_proxy"], errors="coerce").fillna(0.0).to_numpy()

    new_eq = float((p_adj * eq_cond).sum())
    new_usd = float((p_adj * eq_cond * vps).sum())

    return {
        "baseline_expected_structures": base_eq,
        "baseline_expected_dollars": base_usd,
        "new_expected_structures": new_eq,
        "new_expected_dollars": new_usd,
        "avoided_expected_structures": base_eq - new_eq,
        "avoided_expected_dollars": base_usd - new_usd,
        "avoided_dollar_pct": (base_usd - new_usd) / base_usd if base_usd > 0 else 0.0,
    }


# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------


def plot_reliability(y_true: np.ndarray, p_pred: np.ndarray) -> Path:
    if len(y_true) == 0:
        return OUTDIR / "severity_reliability_any_damage_test_2024.png"
    prob_true, prob_pred = calibration_curve(y_true, p_pred, n_bins=10, strategy="quantile")
    ece, _ = expected_calibration_error(y_true, p_pred, n_bins=10)
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", color="#e63946", lw=2, label="p_any_damage")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Severity Any-Damage Reliability (Test 2024)\nECE={ece:.5f}")
    ax.legend()
    fig.tight_layout()
    out = OUTDIR / "severity_reliability_any_damage_test_2024.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pred_vs_obs_positive(obs: np.ndarray, pred: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    if len(obs):
        x_max = float(np.quantile(obs, 0.98))
        y_max = float(np.quantile(pred, 0.98))
        m = max(x_max, y_max, 1.0)
        ax.scatter(obs, pred, s=14, alpha=0.4, color="#457b9d")
        ax.plot([0, m], [0, m], "k--", lw=1)
        ax.set_xlim(0, m)
        ax.set_ylim(0, m)
    ax.set_xlabel("Observed eq_struct_loss (positive cases)")
    ax.set_ylabel("Predicted eq_struct_loss (positive cases)")
    ax.set_title("Severity Positive-Loss Fit (Test 2024)")
    fig.tight_layout()
    out = OUTDIR / "severity_pred_vs_obs_positive_test_2024.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_top_segment_map(annual: pd.DataFrame, year: int) -> Path:
    d = annual[annual["year"] == year].copy()
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    sc = ax.scatter(
        d["cluster_lon"],
        d["cluster_lat"],
        c=d["sum_expected_dollar_loss"],
        cmap="inferno",
        s=28,
        alpha=0.9,
    )
    n_top = max(1, int(math.ceil(len(d) * TOPK_SEGMENT_SHARE)))
    top = d.nsmallest(n_top, "rank_within_year")
    ax.scatter(
        top["cluster_lon"],
        top["cluster_lat"],
        facecolors="none",
        edgecolors="#00d1ff",
        s=65,
        linewidths=1.2,
        label=f"Top {int(TOPK_SEGMENT_SHARE*100)}% ({n_top})",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Top Expected-Loss Segments ({year})")
    fig.colorbar(sc, ax=ax, label="Annual expected dollar loss")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out = OUTDIR / f"severity_top_segments_map_{year}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_mitigation_bar(scenarios: pd.DataFrame) -> Path:
    d = scenarios.copy()
    d["label"] = d["scenario"] + " | -" + (d["ignition_reduction_pct"] * 100).round(0).astype(int).astype(str) + "%"
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.bar(d["label"], d["avoided_expected_dollars"], color="#e63946", alpha=0.85)
    ax.set_ylabel("Avoided expected dollar loss (2024)")
    ax.set_title("Mitigation Scenario Comparison")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    out = OUTDIR / "mitigation_avoided_loss_bar_2024.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    panel = load_panel_full(PANEL_FULL_CSV)
    exposure = build_segment_exposure(panel)
    exposure.to_csv(OUTDIR / "segment_exposure_table.csv", index=False)
    log.info(f"Saved: {OUTDIR / 'segment_exposure_table.csv'}")

    severity_train = build_severity_training_table(panel, exposure)
    severity_train.to_csv(OUTDIR / "severity_training_table.csv", index=False)
    log.info(f"Saved: {OUTDIR / 'severity_training_table.csv'}")

    train_df, val_df, test_df = _split_time(severity_train)
    artifacts = train_severity_hurdle_model(train_df, val_df, test_df)

    panel_loss = score_expected_loss_panel(panel, exposure, artifacts)

    # Save daily loss panel (required deliverable)
    daily_cols = [
        "cluster_id",
        "date",
        "year",
        "month",
        "season",
        "cluster_lat",
        "cluster_lon",
        "hftd_tier",
        "p_ignition_cal",
        "p_any_damage_cond_ign",
        "expected_eq_struct_loss_cond_ign",
        "expected_eq_struct_loss",
        "value_per_structure_proxy",
        "expected_dollar_loss",
    ]
    if "expected_burned_acres_cond_ign" in panel_loss.columns:
        daily_cols.append("expected_burned_acres_cond_ign")
    daily = panel_loss[daily_cols].copy()
    daily.to_csv(OUTDIR / "segment_expected_loss_daily.csv", index=False)

    seasonal, annual = _aggregate_loss_outputs(panel_loss)
    seasonal.to_csv(OUTDIR / "segment_expected_loss_seasonal.csv", index=False)
    annual.to_csv(OUTDIR / "segment_expected_loss_annual.csv", index=False)

    # Mitigation-ready top segments summary
    annual_2024 = annual[annual["year"] == TEST_YEAR].copy().sort_values("sum_expected_dollar_loss", ascending=False)
    n_target = max(1, int(math.ceil(annual_2024["cluster_id"].nunique() * TOPK_SEGMENT_SHARE)))
    top_2024 = annual_2024.head(max(20, n_target)).copy()
    top_2024["mitigation_priority"] = np.where(
        top_2024["rank_within_year"] <= n_target,
        "high_priority_intervention",
        "monitor_or_targeted_maintenance",
    )
    top_2024.to_csv(OUTDIR / "top_segments_expected_dollar_loss_2024.csv", index=False)

    # Mitigation scenarios
    top_segments = annual_2024.head(n_target)["cluster_id"].astype(int).tolist()
    hftd_segments = (
        annual_2024[annual_2024["hftd_tier"].isin(["tier3", "tier2"])]
        .head(n_target)["cluster_id"].astype(int).tolist()
    )
    if len(hftd_segments) == 0:
        hftd_segments = top_segments[:]
    all_segments = annual_2024["cluster_id"].astype(int).tolist()
    rnd_segments = random.sample(all_segments, k=min(n_target, len(all_segments)))

    panel_2024 = panel_loss[panel_loss["year"] == TEST_YEAR].copy()
    scenario_rows: list[dict[str, Any]] = []
    for scen_name, segs in [
        ("top10_expected_loss", top_segments),
        ("top10_hftd_segments", hftd_segments),
        ("top10_random_baseline", rnd_segments),
    ]:
        for pct in MITIGATION_REDUCTION_PCTS:
            res = simulate_mitigation(panel_2024, segs, pct)
            scenario_rows.append(
                {
                    "scenario": scen_name,
                    "n_segments_targeted": len(segs),
                    "ignition_reduction_pct": float(pct),
                    **res,
                }
            )
    scenarios = pd.DataFrame(scenario_rows)
    scenarios.to_csv(OUTDIR / "mitigation_scenarios_2024.csv", index=False)

    # Feature importance
    fi = artifacts.feature_importance.copy()
    fi.to_csv(OUTDIR / "severity_feature_importance.csv", index=False)

    # Plots
    reliability_path = plot_reliability(
        np.asarray(artifacts.diagnostics.get("y_any_test", np.array([]))),
        np.asarray(artifacts.diagnostics.get("p_any_test", np.array([]))),
    )
    y_test = np.asarray(artifacts.diagnostics.get("obs_eq_test", np.array([])))
    y_pred_test = np.asarray(artifacts.diagnostics.get("pred_eq_cond_ign_test", np.array([])))
    pos = y_test > 0
    pred_vs_obs_path = plot_pred_vs_obs_positive(y_test[pos], y_pred_test[pos])
    map_path = plot_top_segment_map(annual, TEST_YEAR)
    mitigation_plot_path = plot_mitigation_bar(scenarios)

    linkage_summary = severity_train.attrs.get("linkage_summary", {})
    metrics = {
        "severity_mode": artifacts.mode,
        "time_split": {"train_end_year": TRAIN_END_YEAR, "val_year": VAL_YEAR, "test_year": TEST_YEAR},
        "linkage_summary": linkage_summary,
        "model_metrics": artifacts.metrics,
        "counts": {
            "n_panel_rows": int(len(panel)),
            "n_training_rows": int(len(severity_train)),
            "n_clusters": int(panel["cluster_id"].nunique()),
        },
        "files": {
            "segment_expected_loss_daily_csv": str(OUTDIR / "segment_expected_loss_daily.csv"),
            "segment_expected_loss_seasonal_csv": str(OUTDIR / "segment_expected_loss_seasonal.csv"),
            "segment_expected_loss_annual_csv": str(OUTDIR / "segment_expected_loss_annual.csv"),
            "top_segments_expected_dollar_loss_2024_csv": str(OUTDIR / "top_segments_expected_dollar_loss_2024.csv"),
            "severity_feature_importance_csv": str(OUTDIR / "severity_feature_importance.csv"),
            "mitigation_scenarios_2024_csv": str(OUTDIR / "mitigation_scenarios_2024.csv"),
            "reliability_plot_png": str(reliability_path),
            "pred_vs_obs_plot_png": str(pred_vs_obs_path),
            "top_segments_map_png": str(map_path),
            "mitigation_bar_png": str(mitigation_plot_path),
        },
        "assumptions": {
            "eq_struct_loss_definition": "structures_destroyed + 0.25 * structures_damaged",
            "home_value_to_replacement_multiplier": HOME_VALUE_TO_REPLACEMENT_MULT,
            "fallback_burned_acre_to_eq_struct_factor": BURNED_ACRE_TO_EQ_STRUCT_FACTOR,
        },
    }
    with open(OUTDIR / "severity_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print("SEVERITY LOSS MODEL COMPLETE")
    print("=" * 72)
    print(f"  Mode                      : {artifacts.mode}")
    print(f"  Training rows             : {len(severity_train):,}")
    print(f"  Panel rows scored         : {len(panel_loss):,}")
    print(f"  Daily output              : {(OUTDIR / 'segment_expected_loss_daily.csv').resolve()}")
    print(f"  Annual output             : {(OUTDIR / 'segment_expected_loss_annual.csv').resolve()}")
    print(f"  Mitigation scenarios      : {(OUTDIR / 'mitigation_scenarios_2024.csv').resolve()}")
    print(f"  Metrics JSON              : {(OUTDIR / 'severity_model_metrics.json').resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
