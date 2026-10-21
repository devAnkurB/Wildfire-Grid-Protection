"""
ignition_probability.py  —  v3
===============================
Ignition probability model: P(confirmed_ignition | location-day, weather)

PANEL DESIGN
────────────
Denominator: Every spatial cluster × every calendar day 2014–2024.
Positives:   Confirmed ignitions (fire spread ≥ 0.25 acres) from 2021–2024.

NOTE ON PRE-2021 DATA
─────────────────────
CPUC fire size reporting changed format in ~2021:
  Pre-2021: linear spread distance  ("< 3 meters", "3 meters – 0.25 Acres")
  Post-2021: acreage                ("Less Than .25 Acres", ".26 – 9.99 Acres")

Because the acreage threshold (≥ 0.25 acres) cannot be applied to linear
measurements, pre-2021 incident rows carry confirmed_ignition = 0 regardless
of actual fire size.  Those rows still contribute valid denominator (non-ignition)
days and add fire-weather signal.  Only 2021–2024 rows are used as labeled
positives (n = 344).

TIME SPLIT (no temporal leakage)
─────────────────────────────────
  Train : 2021–2022
  Val   : 2023
  Test  : 2024

  (Optional) Pre-2021 rows are kept only for feature engineering (rolling weather) but are excluded from training by default.

METHODOLOGY
───────────
  • Spatial clustering: K-means, configurable segments (currently 1000; circuit-segment analogy)
  • Model: Random Forest (Warner et al. 2025)
  • Class imbalance: undersample negatives, Pozzolo (2015) calibration
  • Features: BI, ERC, rmin, wind, tmax, precip, month, cluster coords, LANDFIRE fuels (fuel category + FBFM40 code), topography (elevation, slope), HFTD tier (CPUC regulatory fire-threat designation)
  • Sensitivity ablation only: transmission-line proximity proxy
"""

from __future__ import annotations
import json, logging, warnings, re, math, os
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import geopandas as gpd
import netCDF4 as nc_lib
import rasterio
from rasterio.windows import Window
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                              average_precision_score, roc_curve,
                              precision_recall_curve)
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# ── OPTIONAL: SHAP FEATURE IMPORTANCE ────────────────────────────────────────
try:
    import shap  # type: ignore
except Exception:
    shap = None

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
    category=UserWarning,
)
os.environ.setdefault("PYTHONWARNINGS", "ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

HERE        = Path(__file__).resolve().parent
INPUT_CSV   = HERE / "fire_incidents" / "incidents_features.csv"
FALLBACK_CSV = HERE / "fire_incidents" / "incidents_gridmet.csv"
GRIDMET_DIR = HERE / "environmental_data"
TOPO_DIR    = GRIDMET_DIR / "topo"
TX_LINES_SHP = HERE / "Electrical_Grid" / "Transmission_Line" / "TransmissionLine_CEC.shp"
HFTD_CSV     = HERE / "outputs" / "cluster_grid_features.csv"  # produced by join_grid_features.py
OUTDIR      = HERE / "outputs"; OUTDIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 1000

# --- Feature column definitions ---
GRIDMET_VARS = {
    "bi":   ("burning_index_g",            "bi"),
    "erc":  ("energy_release_component-g", "erc"),
    "rmin": ("relative_humidity",          "rmin"),
    "vs":   ("wind_speed",                 "wind_ms"),
    "tmmx": ("air_temperature",            "tmax_c"),
    "pr":   ("precipitation_amount",       "precip_mm"),
}

WEATHER_COLS = ["bi", "erc", "rmin", "wind_ms", "tmax_c", "precip_mm"]

# Rolling weather / moisture-memory features (computed from daily GridMET per cluster)
ROLLING_COLS = [
    "bi_7d_mean", "bi_30d_mean",
    "erc_14d_max",
    "precip_30d_sum",
    "days_since_rain_ge_5mm",
]

# Simple physics-motivated interactions (computed after topo + weather joins)
INTERACTION_COLS = [
    "wind_x_slope",
    "hot_dry_index",
    "erc_x_wind",
]

# Ignition history / memory features (computed per cluster using only prior days)
HISTORY_COLS = [
    "ign_30d",
    "ign_180d",
    "ign_365d",
    "days_since_ign",
    "ign_rate_smooth",
]

BASE_NUMERIC = WEATHER_COLS + ROLLING_COLS + INTERACTION_COLS + HISTORY_COLS + ["month", "year"]
COORD_COLS   = ["cluster_lat", "cluster_lon"]
FUEL_COLS    = ["fuel_category", "fuel_model_code"]
TOPO_COLS    = ["elevation_m", "slope_deg"]
# Grid proxy from CEC transmission lines. This is used as an infrastructure
# exposure proxy (not causal attribution to transmission equipment).
TX_PROXY_COLS = ["tx_dist_km", "tx_len_km_10km", "tx_count_10km"]
HFTD_COLS     = ["hftd_tier"]   # CPUC regulatory designation, added as categorical
BASE_CATEGORICAL = ["season"]
LABEL_COL    = "ignition"
TX_BUFFER_KM = 10.0
FIRE_SEASON_MONTHS = (5, 6, 7, 8, 9, 10)  # May-Oct
EXTREME_WEATHER_Q = 0.90

HFTD_NOTE = (
    "HFTD tier (CPUC Decision 17-12-024) is the official regulatory fire-threat designation "
    "for overhead utility infrastructure. Tier 3 = extreme risk, Tier 2 = elevated risk. "
    "It encodes terrain, wind exposure, and vegetation factors independent of daily weather, "
    "grounding ignition probability estimates in actual grid risk geography."
)

TX_PROXY_NOTE = (
    "Transmission-line proximity is used as a public-data infrastructure-density proxy: "
    "distance to nearest transmission corridor, total transmission length within 10 km, "
    "and count of segments within 10 km. This approximates where broader grid assets "
    "are concentrated, while not asserting that transmission circuits are the primary "
    "direct ignition source."
)

SEASON_MAP = {1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",
              6:"summer",7:"summer",8:"summer",9:"fall",10:"fall",
              11:"fall",12:"winter"}


def terrain_from_3x3(arr: np.ndarray, cell_m: float = 10.0) -> tuple[float, float]:
    arr = arr.astype(float)
    arr[(arr < -500) | (arr > 9000)] = np.nan
    elev = float(arr[1, 1]) if arr.shape == (3, 3) and np.isfinite(arr[1, 1]) else np.nan
    if arr.shape != (3, 3) or np.isnan(arr).any():
        return elev, np.nan
    dz_dx = ((arr[0, 2] + 2 * arr[1, 2] + arr[2, 2]) - (arr[0, 0] + 2 * arr[1, 0] + arr[2, 0])) / (8 * cell_m)
    dz_dy = ((arr[2, 0] + 2 * arr[2, 1] + arr[2, 2]) - (arr[0, 0] + 2 * arr[0, 1] + arr[0, 2])) / (8 * cell_m)
    slope = float(np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))))
    return elev, slope


def add_topography_to_centroids(centroids: pd.DataFrame) -> pd.DataFrame:
    out = centroids.copy()
    out["elevation_m"] = np.nan
    out["slope_deg"] = np.nan

    tif_paths = sorted(TOPO_DIR.glob("*.tif")) if TOPO_DIR.exists() else []
    if not tif_paths:
        log.warning("No DEM tiles found; topography columns left NaN.")
        return out

    pat = re.compile(r"USGS_13_n(\d+)w(\d+)", re.IGNORECASE)
    idx = {}
    for p in tif_paths:
        m = pat.search(p.name)
        if m:
            idx[(int(m.group(1)), int(m.group(2)))] = p

    # Filter to only tiles needed by cluster centroids.
    by_tile: dict[tuple[int, int], list[int]] = {}
    for i, r in out.iterrows():
        key = (int(math.ceil(float(r["cluster_lat"]))), int(math.ceil(-float(r["cluster_lon"]))))
        by_tile.setdefault(key, []).append(i)

    needed = {k: idx[k] for k in by_tile if k in idx}
    log.info(f"Topography extraction: {len(needed)}/{len(idx)} DEM tiles needed for {len(out)} clusters")
    for key, rows in by_tile.items():
        path = needed.get(key)
        if path is None:
            continue
        with rasterio.open(path) as ds:
            nodata = ds.nodata
            for i in rows:
                lat = float(out.at[i, "cluster_lat"])
                lon = float(out.at[i, "cluster_lon"])
                rr, cc = ds.index(lon, lat)
                if not (0 <= rr < ds.height and 0 <= cc < ds.width):
                    continue
                arr = ds.read(
                    1,
                    window=Window(cc - 1, rr - 1, 3, 3),
                    boundless=True,
                    fill_value=np.nan,
                ).astype(float)
                if nodata is not None:
                    arr[arr == nodata] = np.nan
                elev, slope = terrain_from_3x3(arr)
                out.at[i, "elevation_m"] = elev
                out.at[i, "slope_deg"] = slope

    return out


def add_transmission_proxy_to_centroids(centroids: pd.DataFrame, buffer_km: float = TX_BUFFER_KM) -> pd.DataFrame:
    """
    Add electric-grid proxy features from CEC transmission line geometry.

    Features:
      tx_dist_km      distance from cluster centroid to nearest transmission line
      tx_len_km_10km  total transmission line length within `buffer_km`
      tx_count_10km   number of transmission line geometries intersecting buffer

    These are proxy indicators of grid infrastructure concentration for research
    settings with public data only.
    """
    out = centroids.copy()
    out["tx_dist_km"] = np.nan
    out["tx_len_km_10km"] = np.nan
    out["tx_count_10km"] = np.nan

    if not TX_LINES_SHP.exists():
        log.warning(f"Transmission shapefile not found: {TX_LINES_SHP}; proxy features left NaN.")
        return out

    tx = gpd.read_file(TX_LINES_SHP)
    tx = tx[tx.geometry.notna() & ~tx.geometry.is_empty].copy()
    if tx.empty:
        log.warning("Transmission shapefile loaded but has no valid geometry; proxy features left NaN.")
        return out

    # Use a metric CRS for distances/lengths.
    if tx.crs is None:
        tx = tx.set_crs(epsg=4326, allow_override=True)
    tx = tx.to_crs(epsg=3310)

    pts = gpd.GeoDataFrame(
        out[["cluster_id", "cluster_lat", "cluster_lon"]].copy(),
        geometry=gpd.points_from_xy(out["cluster_lon"], out["cluster_lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3310)

    sidx = tx.sindex
    tx_union = tx.geometry.unary_union
    buffer_m = float(buffer_km) * 1000.0

    dists_km: list[float] = []
    lens_km: list[float] = []
    counts: list[int] = []

    for geom in pts.geometry:
        dists_km.append(float(geom.distance(tx_union) / 1000.0))
        buf = geom.buffer(buffer_m)
        cand_idx = list(sidx.query(buf, predicate="intersects"))
        if not cand_idx:
            lens_km.append(0.0)
            counts.append(0)
            continue
        cand = tx.iloc[cand_idx]
        intersect = cand.geometry.intersection(buf)
        total_km = float(intersect.length.sum() / 1000.0)
        lens_km.append(total_km)
        counts.append(int((intersect.length > 0).sum()))

    out["tx_dist_km"] = dists_km
    out["tx_len_km_10km"] = lens_km
    out["tx_count_10km"] = counts

    log.info(
        f"Transmission proxy extracted for {len(out)} clusters "
        f"(buffer={buffer_km:.1f} km, source={TX_LINES_SHP.name})"
    )
    return out


def add_hftd_to_centroids(centroids: pd.DataFrame) -> pd.DataFrame:
    """
    Join CPUC HFTD tier to each cluster centroid.

    Source: cluster_grid_features.csv produced by join_grid_features.py,
    which performs a point-in-polygon spatial join of cluster centroids
    against the CPUC HFTD v3 (2021) shapefile (CPUC Decision 17-12-024).

    Tier definitions:
      tier3     Extreme risk — mandatory enhanced mitigation required
      tier2     Elevated risk — enhanced mitigation strongly encouraged
      tier1_hhz High Hazard Zone — legacy designation, limited scope
      non_hftd  Outside all HFTD boundaries

    Why HFTD tier instead of transmission line proximity:
      • CPUC incident data is overwhelmingly caused by distribution lines,
        not transmission lines. Proximity to 115–500kV steel towers does
        not measure ignition risk for 12kV wooden-pole distribution circuits.
      • HFTD tier directly encodes regulatory fire risk for ALL overhead
        utility infrastructure, including distribution lines.
      • HFTD tier has a clear physical basis (terrain, wind, vegetation)
        and is independently validated by CPUC rulemaking.
      • Transmission line density from the CEC shapefile often equals NaN
        for clusters without the file present, silently degrading the model.
    """
    out = centroids.copy()
    out["hftd_tier"] = "non_hftd"   # default

    if not HFTD_CSV.exists():
        log.warning(
            f"HFTD features file not found: {HFTD_CSV}\n"
            "  Run join_grid_features.py first to generate it.\n"
            "  hftd_tier will default to 'non_hftd' for all clusters."
        )
        return out

    hftd = pd.read_csv(HFTD_CSV)
    if "hftd_tier" not in hftd.columns or "cluster_id" not in hftd.columns:
        log.warning("cluster_grid_features.csv missing required columns; hftd_tier defaulting to non_hftd.")
        return out

    out = out.merge(
        hftd[["cluster_id", "hftd_tier"]],
        on="cluster_id", how="left", suffixes=("", "_joined")
    )
    if "hftd_tier_joined" in out.columns:
        out["hftd_tier"] = out["hftd_tier_joined"].fillna("non_hftd")
        out = out.drop(columns=["hftd_tier_joined"])
    else:
        out["hftd_tier"] = out["hftd_tier"].fillna("non_hftd")

    tier_counts = out["hftd_tier"].value_counts()
    log.info(f"HFTD tier distribution across {len(out)} clusters:")
    for tier, cnt in tier_counts.items():
        log.info(f"  {tier:<12}: {cnt} ({cnt/len(out)*100:.0f}%)")

    return out


def load_incidents(path: Path) -> pd.DataFrame:
    if not path.exists() and FALLBACK_CSV.exists():
        log.warning(f"{path.name} not found; falling back to {FALLBACK_CSV.name}")
        path = FALLBACK_CSV
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Keep rows with valid date + coordinates; exclude PGE inflation years
    df = df[df["date"].notna() & df["latitude"].notna()
            & df["longitude"].notna() & (df["inflation_flag"] == 0)].copy()
    # Fuel columns are optional (present if add_fuels_topo.py has been run)
    if "fuel_category" not in df.columns:
        df["fuel_category"] = "unknown"
    if "fuel_model_code" not in df.columns:
        df["fuel_model_code"] = np.nan
    log.info(f"Incidents after filtering: {len(df):,}")
    log.info(f"  Confirmed ignitions (2021+): "
             f"{df[df['confirmed_ignition']==1]['year'].value_counts().sort_index().to_dict()}")
    return df


def cluster_locations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coords = df[["latitude", "longitude"]].values
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df = df.copy()
    df["cluster_id"] = km.fit_predict(coords)
    # Aggregate static fuel features per cluster (mode for category, mode for code)
    def _mode(series):
        s = series.dropna()
        if len(s) == 0:
            return np.nan
        return s.value_counts().index[0]

    fuel_cat_by_cluster = df.groupby("cluster_id")["fuel_category"].apply(_mode)
    fuel_code_by_cluster = df.groupby("cluster_id")["fuel_model_code"].apply(_mode)

    centroids = pd.DataFrame(km.cluster_centers_,
                             columns=["cluster_lat", "cluster_lon"])
    centroids["cluster_id"] = range(N_CLUSTERS)
    centroids["fuel_category"] = centroids["cluster_id"].map(fuel_cat_by_cluster).fillna("unknown")
    centroids["fuel_model_code"] = centroids["cluster_id"].map(fuel_code_by_cluster)
    log.info(f"Clustered {len(df):,} incidents into {N_CLUSTERS} spatial segments")
    return df, centroids

# ── STEP 2: BUILD PANEL ───────────────────────────────────────────────────────

def build_panel(df_inc: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    """
    Full denominator panel: every cluster × every day 2014-01-01 → 2024-12-31.
    Ignition = 1 only where a confirmed ignition (≥0.25 ac) occurred in 2021+.
    Pre-2021 cluster-days: always 0 (valid non-ignition denominator).
    """
    all_days = pd.date_range("2014-01-01", "2024-12-31", freq="D")
    cluster_ids = centroids["cluster_id"].values

    log.info(f"Building panel: {len(cluster_ids)} clusters × "
             f"{len(all_days):,} days = {len(cluster_ids)*len(all_days):,} location-days")

    panel = pd.DataFrame({
        "cluster_id": np.repeat(cluster_ids, len(all_days)),
        "date":       np.tile(all_days,      len(cluster_ids)),
    })
    panel["ignition"] = 0

    # Overlay confirmed ignitions (2021+ only, given reporting format change)
    confirmed = df_inc[
        (df_inc["confirmed_ignition"] == 1) & (df_inc["year"] >= 2021)
    ].copy()
    confirmed["date_day"] = confirmed["date"].dt.normalize()
    ignition_days = (confirmed[["cluster_id", "date_day"]]
                     .drop_duplicates()
                     .rename(columns={"date_day": "date"}))
    ignition_days["ignition"] = 1

    panel = panel.merge(ignition_days, on=["cluster_id", "date"],
                        how="left", suffixes=("", "_ev"))
    panel["ignition"] = panel["ignition_ev"].fillna(panel["ignition"]).astype(int)
    panel = panel.drop(columns=["ignition_ev"])

    panel = panel.merge(centroids, on="cluster_id", how="left")
    panel["year"]   = panel["date"].dt.year
    panel["month"]  = panel["date"].dt.month
    panel["season"] = panel["month"].map(SEASON_MAP)
    # Treat fuel model code as categorical (string) for OneHotEncoder
    if "fuel_model_code" in panel.columns:
        panel["fuel_model_code"] = panel["fuel_model_code"].astype("Int64").astype(str).replace("<NA>", "unknown")
    pos = int(panel["ignition"].sum())
    log.info(f"Panel: {len(panel):,} rows | {pos:,} ignitions "
             f"({pos/len(panel)*100:.4f}% base rate)")
    return panel

# ── STEP 3: JOIN GRIDMET ─────────────────────────────────────────────────────

def load_gridmet_grid():
    sample = next((GRIDMET_DIR / "bi").glob("bi_*.nc"))
    with nc_lib.Dataset(sample) as ds:
        return np.array(ds.variables["lat"][:]), np.array(ds.variables["lon"][:])


def nearest_idx(arr, val):
    return int(np.argmin(np.abs(arr - val)))


def join_gridmet(panel: pd.DataFrame) -> pd.DataFrame:
    lats_gm, lons_gm = load_gridmet_grid()
    log.info("GridMET grid loaded.")

    cluster_grid = {}
    for _, r in panel[["cluster_id","cluster_lat","cluster_lon"]].drop_duplicates().iterrows():
        cluster_grid[int(r["cluster_id"])] = (
            nearest_idx(lats_gm, r["cluster_lat"]),
            nearest_idx(lons_gm, r["cluster_lon"]),
        )

    for col in WEATHER_COLS:
        panel[col] = np.nan

    open_ds: dict = {}
    def get_ds(var, yr):
        k = (var, yr)
        if k not in open_ds:
            p = GRIDMET_DIR / var / f"{var}_{yr}.nc"
            open_ds[k] = nc_lib.Dataset(p) if p.exists() else None
        return open_ds[k]

    years = sorted(panel["year"].unique())
    log.info(f"Extracting GridMET for {len(years)} years × {N_CLUSTERS} clusters …")

    for yr in years:
        yr_mask  = panel["year"] == yr
        yr_idx   = panel.index[yr_mask]
        yr_panel = panel.loc[yr_mask]

        # Load all variables for this year into numpy arrays
        var_arrays = {}
        for var_folder, (nc_varname, out_col) in GRIDMET_VARS.items():
            ds = get_ds(var_folder, int(yr))
            if ds is None:
                var_arrays[out_col] = None
                continue
            arr = np.array(ds.variables[nc_varname][:])
            if out_col == "tmax_c":
                arr = arr - 273.15
            arr[(arr > 1e30) | (arr < -1e5)] = np.nan
            var_arrays[out_col] = arr

        for cid, (li, lj) in cluster_grid.items():
            c_mask  = yr_panel["cluster_id"] == cid
            if not c_mask.any():
                continue
            c_idx   = yr_idx[c_mask.values]
            c_dates = panel.loc[c_idx, "date"]
            doys    = (c_dates - pd.Timestamp(f"{int(yr)}-01-01")).dt.days

            for out_col, arr in var_arrays.items():
                if arr is None:
                    continue
                try:
                    panel.loc[c_idx, out_col] = arr[doys.values, li, lj]
                except Exception:
                    pass

        log.info(f"  Year {yr} done")

    for ds in open_ds.values():
        if ds is not None:
            try: ds.close()
            except: pass

    joined = panel[panel[WEATHER_COLS].notna().all(axis=1)].reset_index(drop=True)
    log.info(f"Panel after GridMET join: {len(joined):,} rows "
             f"({len(panel)-len(joined):,} dropped for missing weather)")
    return joined


# --- Rolling weather and interaction features ---
def add_rolling_weather_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add per-cluster rolling weather features.

    Assumes daily panel rows with columns in WEATHER_COLS and keys (cluster_id, date).
    Uses only past/current days (no leakage) via rolling windows on sorted dates.
    """
    out = panel.sort_values(["cluster_id", "date"]).copy()

    # Defensive: ensure numeric
    for c in WEATHER_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    g = out.groupby("cluster_id", sort=False)

    # Rolling means / max / sums
    out["bi_7d_mean"] = g["bi"].transform(lambda s: s.rolling(window=7, min_periods=3).mean())
    out["bi_30d_mean"] = g["bi"].transform(lambda s: s.rolling(window=30, min_periods=10).mean())
    out["erc_14d_max"] = g["erc"].transform(lambda s: s.rolling(window=14, min_periods=5).max())
    out["precip_30d_sum"] = g["precip_mm"].transform(lambda s: s.rolling(window=30, min_periods=10).sum())

    # Days since meaningful rain (>= 5mm). Compute from precipitation time series.
    def _days_since_rain_ge_5mm(precip: pd.Series) -> pd.Series:
        # precip is already sorted by date within cluster
        rain = (precip.fillna(0.0) >= 5.0).to_numpy(dtype=bool)
        days = np.empty(len(rain), dtype=float)
        last = -1
        for i, hit in enumerate(rain):
            if hit:
                last = i
                days[i] = 0.0
            else:
                days[i] = float(i - last) if last >= 0 else float(i + 1)
        return pd.Series(days, index=precip.index)

    out["days_since_rain_ge_5mm"] = g["precip_mm"].transform(_days_since_rain_ge_5mm)

    # Fill any remaining NaNs defensively (early days in series)
    for c in ROLLING_COLS:
        if c in out.columns:
            out[c] = out[c].fillna(out[c].median())

    return out


def add_interaction_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add simple interaction features. Requires topo columns if enabled."""
    out = panel.copy()

    # wind * slope (upslope spread proxy). If slope missing, treat as 0.
    slope = pd.to_numeric(out.get("slope_deg"), errors="coerce").fillna(0.0)
    wind = pd.to_numeric(out.get("wind_ms"), errors="coerce").fillna(0.0)
    out["wind_x_slope"] = wind * slope

    # hot-dry index: tmax * (100 - rmin)
    tmax = pd.to_numeric(out.get("tmax_c"), errors="coerce").fillna(0.0)
    rmin = pd.to_numeric(out.get("rmin"), errors="coerce").fillna(0.0)
    out["hot_dry_index"] = tmax * (100.0 - rmin)

    # erc * wind
    erc = pd.to_numeric(out.get("erc"), errors="coerce").fillna(0.0)
    out["erc_x_wind"] = erc * wind

    return out

# --- Ignition history / memory features ---
def add_ignition_history_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add per-cluster ignition history ("memory") features with no leakage.

    Requires columns: cluster_id, date, ignition (0/1).
    All windowed features exclude the current day using shift(1).
    """
    out = panel.sort_values(["cluster_id", "date"]).copy()

    if "ignition" not in out.columns:
        raise KeyError("add_ignition_history_features: required column 'ignition' not found")

    g = out.groupby("cluster_id", sort=False)

    # Prior-day ignition so today does not leak into rolling windows
    out["ignition_lag1"] = g["ignition"].shift(1).fillna(0).astype(int)

    # Rolling counts over past windows (exclude today)
    out["ign_30d"] = g["ignition_lag1"].transform(lambda s: s.rolling(30, min_periods=1).sum())
    out["ign_180d"] = g["ignition_lag1"].transform(lambda s: s.rolling(180, min_periods=1).sum())
    out["ign_365d"] = g["ignition_lag1"].transform(lambda s: s.rolling(365, min_periods=1).sum())

    # Days since last ignition (based on lagged ignition)
    def _days_since_last(s: pd.Series) -> pd.Series:
        arr = s.to_numpy(dtype=int)
        out_days = np.empty(len(arr), dtype=float)
        last = -1
        for i, v in enumerate(arr):
            if v == 1:
                last = i
                out_days[i] = 0.0
            else:
                out_days[i] = float(i - last) if last >= 0 else float(i + 1)
        return pd.Series(out_days, index=s.index)

    out["days_since_ign"] = g["ignition_lag1"].transform(_days_since_last)

    # Smoothed long-run ignition rate (Bayesian smoothing)
    # prior: alpha ignitions over beta days
    alpha, beta = 1.0, 2000.0
    out["cum_ign_lag"] = g["ignition_lag1"].cumsum()
    out["cum_days"] = g.cumcount() + 1  # 1..T
    out["ign_rate_smooth"] = (out["cum_ign_lag"] + alpha) / (out["cum_days"] + beta)

    # Cleanup intermediates
    out = out.drop(columns=["cum_ign_lag", "cum_days", "ignition_lag1"])

    # Defensive fill (should be none/very few)
    for c in HISTORY_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out

# ── STEP 4: POZZOLO CALIBRATION ──────────────────────────────────────────────

def pozzolo_calibrate(p: np.ndarray, beta: float) -> np.ndarray:
    """Correct for undersampled negatives. Pozzolo et al. (2015)."""
    return p / (p + (1.0 - p) / beta)

# ── STEP 5: SPLIT + UNDERSAMPLE ──────────────────────────────────────────────

def time_split(panel: pd.DataFrame):
    """Time split with no leakage.

    Train: 2021–2022
    Val  : 2023
    Test : 2024

    Note: Pre-2021 rows exist only to support rolling-weather feature engineering
    and denominator construction, but are excluded from training by default because
    the >=0.25 acre positive label is not comparable pre-2021.
    """
    train = panel[(panel["year"] >= 2021) & (panel["year"] <= 2022)]
    val   = panel[panel["year"] == 2023]
    test  = panel[panel["year"] == 2024]

    for name, split in [("train", train), ("val", val), ("test", test)]:
        pos = int(split[LABEL_COL].sum())
        log.info(f"  {name}: {len(split):,} rows | {pos:,} ignitions "
                 f"({pos/max(len(split),1)*100:.4f}%)")

    return train, val, test


def undersample(df: pd.DataFrame, ratio: float = 200.0, seed: int = 42):
    """
    Keep all positives; sample ratio:1 negatives.
    Requires at least 1 positive row (guard against reporting format gaps).
    """
    pos = df[df[LABEL_COL] == 1]
    neg = df[df[LABEL_COL] == 0]

    if len(pos) == 0:
        raise ValueError(
            "No positive (confirmed ignition) rows in training split. "
            "Check that 2021-2022 data is present and confirmed_ignition=1 "
            "rows exist after filtering."
        )

    n_neg_keep = min(len(neg), int(len(pos) * ratio))
    if n_neg_keep == 0:
        raise ValueError("No negative rows available for undersampling.")

    neg_s = neg.sample(n=n_neg_keep, random_state=seed)
    beta  = len(pos) / n_neg_keep
    out   = pd.concat([pos, neg_s]).sample(frac=1, random_state=seed)
    log.info(f"Undersampled: {len(pos)} pos + {n_neg_keep} neg | β = {beta:.5f}")
    return out.reset_index(drop=True), beta

# ── STEP 6: PREPROCESS + TRAIN ───────────────────────────────────────────────

def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp",   SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_features),
    ])


def get_feature_names(prep, numeric_features, categorical_features):
    cat_names = list(prep.named_transformers_["cat"]
                     .named_steps["ohe"]
                     .get_feature_names_out(categorical_features))
    return list(numeric_features) + cat_names


def train_rf(X, y):
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)
    return rf


def log_feature_diagnostics(df: pd.DataFrame, numeric_features: list[str],
                            categorical_features: list[str], label: str):
    """Basic checks to verify requested inputs are present and non-constant."""
    feats = numeric_features + categorical_features
    miss = [c for c in feats if c not in df.columns]
    if miss:
        raise KeyError(f"{label}: missing feature columns: {miss}")

    log.info(f"Feature diagnostics [{label}]")
    for c in numeric_features:
        s = pd.to_numeric(df[c], errors="coerce")
        nuniq = int(s.dropna().nunique())
        log.info(f"  num {c:<16} missing={s.isna().mean():.2%} unique={nuniq}")
    for c in categorical_features:
        s = df[c].astype(str)
        nuniq = int(s.nunique(dropna=False))
        top = s.value_counts(dropna=False).index[0]
        log.info(f"  cat {c:<16} missing={df[c].isna().mean():.2%} unique={nuniq} top={top}")

# ── STEP 7: EVALUATE + PLOT ──────────────────────────────────────────────────

def evaluate(model, X, y, beta, label):
    p_raw = model.predict_proba(X)[:, 1]
    p_cal = pozzolo_calibrate(p_raw, beta)
    auc   = roc_auc_score(y, p_cal)
    brier = brier_score_loss(y, p_cal)
    pr    = average_precision_score(y, p_cal)
    corr  = float(np.corrcoef(p_raw, p_cal)[0, 1])
    log.info(f"  [{label}] AUC={auc:.4f}  PR-AUC={pr:.4f}  "
             f"Brier={brier:.6f}  corr(raw,cal)={corr:.4f}")
    return {"auc": auc, "pr_auc": pr, "brier": brier, "corr_raw_cal": corr}, p_cal, p_raw


def save_plots(y_te, p_raw, p_cal, model, X_val, y_val,
               feat_names, test_panel):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Ignition Probability Model — Location-Day Panel\n"
        r"P(confirmed_ignition $\geq$ 0.25 ac | location-day, weather)",
        fontsize=12, fontweight="bold"
    )

    # ROC
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_te, p_cal)
    ax.plot(fpr, tpr, color="#e63946", lw=2,
            label=f"AUC = {roc_auc_score(y_te, p_cal):.4f}")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set(title="ROC Curve (test 2024)", xlabel="FPR", ylabel="TPR")
    ax.legend()

    # PR
    ax = axes[0, 1]
    prec, rec, _ = precision_recall_curve(y_te, p_cal)
    pr_auc = average_precision_score(y_te, p_cal)
    ax.plot(rec, prec, color="#457b9d", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.axhline(y_te.mean(), color="gray", ls="--", lw=1, label="Baseline (prevalence)")
    ax.set(title="Precision-Recall (test 2024)", xlabel="Recall", ylabel="Precision")
    ax.legend(fontsize=8)

    # P(ignition) vs BI — binned means
    ax = axes[0, 2]
    tmp = pd.DataFrame({"bi": test_panel["bi"].values, "p": p_cal, "y": y_te})
    for col_name, color, label in [("p","#e63946","Predicted"),("y","#457b9d","Observed")]:
        grp = tmp.groupby(pd.cut(tmp["bi"], 15), observed=True)[col_name].mean()
        ax.plot([iv.mid for iv in grp.index], grp.values, "o-",
                color=color, lw=2, label=label, markersize=4)
    ax.set(title="P(Ignition) vs Burning Index", xlabel="BI", ylabel="P(ignition)")
    ax.legend(fontsize=8)

    # P(ignition) vs ERC
    ax = axes[1, 0]
    tmp2 = pd.DataFrame({"erc": test_panel["erc"].values, "p": p_cal, "y": y_te})
    for col_name, color, label in [("p","#e63946","Predicted"),("y","#457b9d","Observed")]:
        grp = tmp2.groupby(pd.cut(tmp2["erc"], 15), observed=True)[col_name].mean()
        ax.plot([iv.mid for iv in grp.index], grp.values, "o-",
                color=color, lw=2, label=label, markersize=4)
    ax.set(title="P(Ignition) vs ERC", xlabel="ERC", ylabel="P(ignition)")
    ax.legend(fontsize=8)

    # Score distribution
    ax = axes[1, 1]
    ax.hist(p_cal[y_te==0], bins=60, alpha=0.6, color="#457b9d",
            label="Non-ignition days", density=True)
    ax.hist(p_cal[y_te==1], bins=30, alpha=0.8, color="#e63946",
            label="Ignition days", density=True)
    ax.set(title="Score Distribution (test 2024)",
           xlabel="Predicted P(ignition)", ylabel="Density")
    ax.legend(fontsize=8)

    # Permutation feature importance (top 10)
    ax = axes[1, 2]
    perm = permutation_importance(model, X_val, y_val, n_repeats=8,
                                  random_state=42, n_jobs=1, scoring="roc_auc")
    top10 = np.argsort(perm.importances_mean)[-10:]
    vals  = perm.importances_mean[top10]
    names = [feat_names[i] for i in top10]
    colors = ["#e63946" if v == vals.max() else "#457b9d" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.set(title="Feature Importance\n(permutation AUC, val 2023)",
           xlabel="ΔAUC-ROC")
    ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    path = OUTDIR / "ignition_model_panel.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path}")


def compute_topk_capture_lift(y_true: np.ndarray, p_cal: np.ndarray,
                              top_fracs: tuple[float, ...] = (0.001, 0.005, 0.01)) -> dict:
    """Operational metrics for rare-event screening."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_cal, dtype=float)
    n = len(y)
    n_pos = int(y.sum())
    base_rate = float(y.mean()) if n > 0 else float("nan")
    order = np.argsort(-p)

    rows = []
    for frac in top_fracs:
        k = max(1, int(math.ceil(n * frac)))
        idx = order[:k]
        hits = int(y[idx].sum())
        precision = float(hits / k) if k > 0 else float("nan")
        recall = float(hits / n_pos) if n_pos > 0 else float("nan")
        lift = float(precision / base_rate) if base_rate > 0 else float("nan")
        rows.append({
            "top_fraction": float(frac),
            "top_k_rows": int(k),
            "hits": int(hits),
            "capture_recall": recall,
            "precision_in_bucket": precision,
            "lift_vs_base_rate": lift,
        })

    return {
        "n_rows": int(n),
        "n_ignitions": int(n_pos),
        "base_rate": base_rate,
        "topk": rows,
    }


def expected_calibration_error(y_true: np.ndarray, p_cal: np.ndarray, n_bins: int = 12) -> tuple[float, list[dict]]:
    """ECE using equal-width bins on [0,1]."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_cal, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows: list[dict] = []

    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i < n_bins - 1:
            m = (p >= lo) & (p < hi)
        else:
            m = (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        obs = float(y[m].mean())
        pred = float(p[m].mean())
        w = float(m.mean())
        ece += w * abs(obs - pred)
        rows.append({
            "bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
            "pred_mean": pred, "obs_rate": obs,
        })
    return float(ece), rows


def save_reliability_plot(y_true: np.ndarray, p_cal: np.ndarray, label: str = "test_2024") -> tuple[float, Path]:
    """Save calibration / reliability curve."""
    prob_true, prob_pred = calibration_curve(y_true, p_cal, n_bins=12, strategy="quantile")
    ece, _ = expected_calibration_error(y_true, p_cal, n_bins=12)

    plt.figure(figsize=(6.8, 5.6))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    plt.plot(prob_pred, prob_true, marker="o", lw=2, color="#e63946", label="Model (calibrated)")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed ignition rate")
    plt.title(f"Reliability Curve ({label})\nECE={ece:.6f}")
    plt.legend()
    plt.tight_layout()
    path = OUTDIR / f"reliability_curve_{label}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {path}")
    return ece, path


def summarize_segment_seasonal_risk(panel_with_probs: pd.DataFrame,
                                    fire_months: tuple[int, ...] = FIRE_SEASON_MONTHS,
                                    extreme_q: float = EXTREME_WEATHER_Q) -> tuple[pd.DataFrame, float]:
    """Aggregate daily risk to cluster-year seasonal risk + rank shifts."""
    fs = panel_with_probs[panel_with_probs["month"].isin(fire_months)].copy()
    if fs.empty:
        raise ValueError("Fire-season panel is empty; cannot build seasonal segment risk.")

    # Extreme vs normal day classification by composite weather severity.
    comps = []
    for c in ["bi", "erc", "wind_ms", "tmax_c"]:
        s = pd.to_numeric(fs[c], errors="coerce")
        sd = float(s.std(ddof=0))
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        comps.append((s - s.mean()) / sd)
    dry = 100.0 - pd.to_numeric(fs["rmin"], errors="coerce")
    dry_sd = float(dry.std(ddof=0))
    if not np.isfinite(dry_sd) or dry_sd == 0:
        dry_sd = 1.0
    comps.append((dry - dry.mean()) / dry_sd)

    wx = sum(comps)
    wx = wx.replace([np.inf, -np.inf], np.nan)
    fs["wx_severity_score"] = wx.fillna(float(wx.median()))
    threshold = float(fs["wx_severity_score"].quantile(extreme_q))
    fs["is_extreme_weather_day"] = fs["wx_severity_score"] >= threshold

    grp = ["cluster_id", "year"]
    agg = fs.groupby(grp, as_index=False).agg(
        seasonal_sum_risk=("p_ignition_cal", "sum"),
        seasonal_mean_risk=("p_ignition_cal", "mean"),
        seasonal_max_risk=("p_ignition_cal", "max"),
        fire_season_days=("p_ignition_cal", "size"),
        extreme_days=("is_extreme_weather_day", "sum"),
    )

    ext = (
        fs[fs["is_extreme_weather_day"]]
        .groupby(grp)["p_ignition_cal"]
        .mean()
        .rename("extreme_mean_risk")
        .reset_index()
    )
    nrm = (
        fs[~fs["is_extreme_weather_day"]]
        .groupby(grp)["p_ignition_cal"]
        .mean()
        .rename("normal_mean_risk")
        .reset_index()
    )
    agg = agg.merge(ext, on=grp, how="left").merge(nrm, on=grp, how="left")
    agg["extreme_mean_risk"] = agg["extreme_mean_risk"].fillna(0.0)
    agg["normal_mean_risk"] = agg["normal_mean_risk"].fillna(0.0)

    cent = panel_with_probs[["cluster_id", "cluster_lat", "cluster_lon"]].drop_duplicates("cluster_id")
    agg = agg.merge(cent, on="cluster_id", how="left")

    agg["rank_seasonal_sum"] = (
        agg.groupby("year")["seasonal_sum_risk"].rank(ascending=False, method="min").astype(int)
    )
    agg["rank_extreme_mean"] = (
        agg.groupby("year")["extreme_mean_risk"].rank(ascending=False, method="min").astype(int)
    )
    agg["rank_normal_mean"] = (
        agg.groupby("year")["normal_mean_risk"].rank(ascending=False, method="min").astype(int)
    )
    agg["rank_shift_extreme_minus_normal"] = agg["rank_extreme_mean"] - agg["rank_normal_mean"]

    return agg.sort_values(["year", "rank_seasonal_sum"]).reset_index(drop=True), threshold


def save_top_risk_segment_map(seasonal: pd.DataFrame, year: int) -> Path:
    m = seasonal[seasonal["year"] == year].copy()
    if m.empty:
        raise ValueError(f"No seasonal rows available for year={year}")

    plt.figure(figsize=(8.2, 6.0))
    sc = plt.scatter(
        m["cluster_lon"], m["cluster_lat"],
        c=m["seasonal_sum_risk"], cmap="inferno", s=26, alpha=0.9
    )
    n_top = max(1, int(math.ceil(len(m) * 0.10)))
    top = m.nsmallest(n_top, "rank_seasonal_sum")
    plt.scatter(top["cluster_lon"], top["cluster_lat"],
                facecolors="none", edgecolors="#00d1ff", s=60, linewidths=1.2,
                label=f"Top 10% segments ({n_top})")
    plt.colorbar(sc, label="Fire-season risk (sum of daily P(ignition))")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Top-Risk Segments Map ({year} Fire Season)")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    path = OUTDIR / f"segment_risk_map_{year}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {path}")
    return path

# ── OPTIONAL: SHAP FEATURE IMPORTANCE ────────────────────────────────────────

def save_shap_plots(model, X_ref, feat_names, label: str = "val_2023", max_samples: int = 5000):
    """Compute and save SHAP summary plots for the trained tree model.

    Notes:
      - Works best for tree ensembles (RandomForest, XGBoost, LightGBM).
      - `X_ref` should be the *preprocessed* feature matrix (numpy array).
      - Uses a subsample to keep runtime reasonable.
    """
    if shap is None:
        log.warning("SHAP not available (package 'shap' not installed). Skipping SHAP plots.")
        return

    try:
        # Subsample for speed / memory
        n = X_ref.shape[0]
        if n == 0:
            log.warning("SHAP: empty reference matrix; skipping.")
            return
        if n > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=max_samples, replace=False)
            Xs = X_ref[idx]
        else:
            Xs = X_ref

        # TreeExplainer handles sklearn RandomForestClassifier
        explainer = shap.TreeExplainer(model)

        # For binary classification, SHAP may return:
        #   - list of arrays (per class)  OR
        #   - a single array
        sv = explainer.shap_values(Xs)
        if isinstance(sv, list) and len(sv) >= 2:
            shap_vals = sv[1]
        else:
            shap_vals = sv

        # Plot: bar (global importance)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, Xs, feature_names=feat_names, plot_type="bar", show=False, max_display=25)
        out_bar = OUTDIR / f"shap_summary_bar_{label}.png"
        plt.tight_layout()
        plt.savefig(out_bar, dpi=150, bbox_inches="tight")
        plt.close()

        # Plot: beeswarm (distribution)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, Xs, feature_names=feat_names, show=False, max_display=25)
        out_swarm = OUTDIR / f"shap_summary_swarm_{label}.png"
        plt.tight_layout()
        plt.savefig(out_swarm, dpi=150, bbox_inches="tight")
        plt.close()

        log.info(f"Saved SHAP plots: {out_bar}")
        log.info(f"Saved SHAP plots: {out_swarm}")

    except Exception as e:
        log.warning(f"SHAP failed; skipping SHAP plots. Reason: {e}")

# ── MAIN ─────────────────────────────────────────────────────────────────────

def run_experiment(panel: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                   name: str, numeric_features: list[str], categorical_features: list[str],
                   ratio: float = 200.0):
    log.info(f"\n--- EXPERIMENT: {name} ---")
    feats = numeric_features + categorical_features

    train_us, beta = undersample(train, ratio=ratio)
    log_feature_diagnostics(train_us, numeric_features, categorical_features, f"{name} train_us")

    prep = build_preprocessor(numeric_features, categorical_features)
    X_tr  = prep.fit_transform(train_us[feats])
    X_val = prep.transform(val[feats])
    X_te  = prep.transform(test[feats])

    y_tr  = train_us[LABEL_COL].values
    y_val = val[LABEL_COL].values
    y_te  = test[LABEL_COL].values

    feat_names = get_feature_names(prep, numeric_features, categorical_features)

    model = train_rf(X_tr, y_tr)

    val_m,  _, _ = evaluate(model, X_val, y_val, beta, f"{name} | val 2023")
    test_m, _, _ = evaluate(model, X_te,  y_te,  beta, f"{name} | test 2024")

    return {
        "name": name,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "beta": float(beta),
        "val_2023": val_m,
        "test_2024": test_m,
    }


def main():
    log.info("STEP 1: Load incidents & cluster locations")
    df_inc, centroids = cluster_locations(load_incidents(INPUT_CSV))
    log.info("STEP 1b: Add topography to cluster centroids (filtered DEM tiles)")
    centroids = add_topography_to_centroids(centroids)
    log.info("STEP 1c: Add electric-grid transmission proxy to cluster centroids")
    centroids = add_transmission_proxy_to_centroids(centroids, buffer_km=TX_BUFFER_KM)
    log.info("STEP 1d: Add HFTD tier to cluster centroids")
    centroids = add_hftd_to_centroids(centroids)

    log.info("STEP 2: Build location-day panel")
    panel = build_panel(df_inc, centroids)

    log.info("STEP 3: Join GridMET fire weather")
    panel = join_gridmet(panel)

    log.info("STEP 3b: Rolling weather + interaction features")
    panel = add_rolling_weather_features(panel)
    panel = add_interaction_features(panel)
    log.info("STEP 3c: Ignition history features")
    panel = add_ignition_history_features(panel)

    log.info("STEP 4: Time split")
    train, val, test = time_split(panel)

    # ── EXPERIMENTS (ablation) ──────────────────────────────────────────────
    log.info("STEP 5: Run ablations")
    exp_results = []

    # (1) Weather-only
    exp_results.append(run_experiment(
        panel, train, val, test,
        name="weather_only",
        numeric_features=BASE_NUMERIC,
        categorical_features=BASE_CATEGORICAL,
        ratio=200.0,
    ))

    # (2) Weather + coords (tests whether geography is a proxy)
    exp_results.append(run_experiment(
        panel, train, val, test,
        name="weather_plus_coords",
        numeric_features=BASE_NUMERIC + COORD_COLS,
        categorical_features=BASE_CATEGORICAL,
        ratio=200.0,
    ))

    # (3) Weather + coords + fuels + topography
    full_numeric = BASE_NUMERIC + COORD_COLS + TOPO_COLS
    full_categorical = BASE_CATEGORICAL + FUEL_COLS
    exp_results.append(run_experiment(
        panel, train, val, test,
        name="weather_coords_fuels_topo",
        numeric_features=full_numeric,
        categorical_features=full_categorical,
        ratio=200.0,
    ))

    # (4) Add HFTD tier — policy-grounded final covariate.
    full_categorical_hftd = full_categorical + HFTD_COLS
    exp_results.append(run_experiment(
        panel, train, val, test,
        name="weather_coords_fuels_topo_hftd",
        numeric_features=full_numeric,
        categorical_features=full_categorical_hftd,
        ratio=200.0,
    ))

    # (5) Sensitivity only: add transmission proxy and compare.
    full_numeric_tx = full_numeric + TX_PROXY_COLS
    exp_results.append(run_experiment(
        panel, train, val, test,
        name="weather_coords_fuels_topo_hftd_gridproxy",
        numeric_features=full_numeric_tx,
        categorical_features=full_categorical_hftd,
        ratio=200.0,
    ))

    # Print a compact comparison table
    print("\n" + "="*72)
    print("ABLATION SUMMARY (test 2024)")
    print("="*72)
    for r in exp_results:
        t = r["test_2024"]
        print(f"  {r['name']:<22} AUC={t['auc']:.4f}  PR-AUC={t['pr_auc']:.4f}  Brier={t['brier']:.6f}")
    print("="*72)

    # ── Fit FULL model again for plots + outputs (keeps existing artifacts) ─
    log.info("STEP 6: Fit full model for plots + outputs")
    train_us, beta = undersample(train, ratio=200.0)

    # Final model is frozen without transmission proxy.
    prep = build_preprocessor(full_numeric, full_categorical_hftd)
    feats_full = full_numeric + full_categorical_hftd
    X_tr  = prep.fit_transform(train_us[feats_full])
    X_val = prep.transform(val[feats_full])
    X_te  = prep.transform(test[feats_full])
    y_tr  = train_us[LABEL_COL].values
    y_val = val[LABEL_COL].values
    y_te  = test[LABEL_COL].values
    feat_names = get_feature_names(prep, full_numeric, full_categorical_hftd)

    model = train_rf(X_tr, y_tr)

    log.info("STEP 7: Evaluate full model")
    val_m,  p_val_cal, p_val_raw = evaluate(model, X_val, y_val, beta, "full | val 2023")
    test_m, p_te_cal,  p_te_raw  = evaluate(model, X_te,  y_te,  beta, "full | test 2024")

    log.info("STEP 8: Plots")
    save_plots(y_te, p_te_raw, p_te_cal, model,
               X_val, y_val, feat_names,
               test.reset_index(drop=True))

    log.info("STEP 8b: SHAP feature importance (optional)")
    save_shap_plots(model, X_val, feat_names, label="val_2023")

    log.info("STEP 8c: Actionable test metrics + calibration")
    topk_metrics = compute_topk_capture_lift(y_te, p_te_cal)
    pd.DataFrame(topk_metrics["topk"]).to_csv(OUTDIR / "topk_capture_lift_test_2024.csv", index=False)
    ece_test, rel_path = save_reliability_plot(y_te, p_te_cal, label="test_2024")
    _, cal_bins = expected_calibration_error(y_te, p_te_cal, n_bins=12)
    pd.DataFrame(cal_bins).to_csv(OUTDIR / "reliability_bins_test_2024.csv", index=False)

    log.info("STEP 9: Save outputs")
    test_out = test.reset_index(drop=True).copy()
    test_out["p_ignition_raw"] = p_te_raw
    test_out["p_ignition_cal"] = p_te_cal
    test_out.to_csv(OUTDIR / "panel_predictions_test.csv", index=False)

    X_full   = prep.transform(panel[feats_full])
    p_full_r = model.predict_proba(X_full)[:, 1]
    p_full_c = pozzolo_calibrate(p_full_r, beta)
    panel_out = panel.copy()
    panel_out["p_ignition_raw"] = p_full_r
    panel_out["p_ignition_cal"] = p_full_c
    panel_out.to_csv(OUTDIR / "panel_full.csv", index=False)

    log.info("STEP 9b: Segment-level seasonal risk outputs")
    seasonal, wx_thr = summarize_segment_seasonal_risk(panel_out)
    seasonal.to_csv(OUTDIR / "segment_seasonal_risk.csv", index=False)
    latest_year = int(seasonal["year"].max())
    shift_latest = seasonal[seasonal["year"] == latest_year].copy()
    shift_latest = shift_latest.sort_values("rank_extreme_mean").reset_index(drop=True)
    shift_latest.to_csv(OUTDIR / f"segment_rank_shift_extreme_vs_normal_{latest_year}.csv", index=False)
    map_path = save_top_risk_segment_map(seasonal, latest_year)

    joblib.dump({
        "model": model, "preprocessor": prep, "beta": beta,
        "numeric_features": full_numeric,
        "categorical_features": full_categorical_hftd,
        "feature_names": feat_names,
        "n_clusters": N_CLUSTERS,
        "centroids": centroids.to_dict(),
        "outcome": "P(confirmed_ignition>=0.25ac | location-day, weather)",
        "note": "Positives from 2021+ only due to pre-2021 linear size reporting",
        "hftd_note": HFTD_NOTE,
        "transmission_proxy_note": "Used only in sensitivity ablation; excluded from final model.",
        "actionable_metrics_test_2024": topk_metrics,
        "calibration_ece_test_2024": ece_test,
        "seasonal_risk_files": {
            "segment_seasonal_risk_csv": str(OUTDIR / "segment_seasonal_risk.csv"),
            "segment_rank_shift_latest_csv": str(OUTDIR / f"segment_rank_shift_extreme_vs_normal_{latest_year}.csv"),
            "segment_risk_map_png": str(map_path),
        },
        "ablations": exp_results,
    }, OUTDIR / "panel_model.joblib")

    metrics = {
        "outcome": "P(confirmed_ignition | location-day, weather)",
        "panel_size": len(panel),
        "n_ignitions": int(panel["ignition"].sum()),
        "base_rate_pct": float(panel["ignition"].mean() * 100),
        "beta_pozzolo": float(beta),
        "n_clusters": N_CLUSTERS,
        "data_note": (
            "Pre-2021 CPUC files report fire size in linear meters, not acres. "
            "Confirmed ignition label (>=0.25 ac) only valid from 2021 onward. "
            "Pre-2021 rows contribute non-ignition denominator days only."
        ),
        "hftd_note": HFTD_NOTE,
        "transmission_proxy_note": "Used only in sensitivity ablation; excluded from final model.",
        "transmission_proxy_buffer_km": TX_BUFFER_KM,
        "fire_season_months": list(FIRE_SEASON_MONTHS),
        "extreme_weather_quantile": EXTREME_WEATHER_Q,
        "extreme_weather_threshold_score": float(wx_thr),
        "val_2023":  val_m,
        "test_2024": test_m,
        "actionable_metrics_test_2024": topk_metrics,
        "calibration_test_2024": {
            "ece": float(ece_test),
            "reliability_curve_path": str(rel_path),
            "reliability_bins_path": str(OUTDIR / "reliability_bins_test_2024.csv"),
        },
        "segment_risk_outputs": {
            "segment_seasonal_risk_csv": str(OUTDIR / "segment_seasonal_risk.csv"),
            "segment_rank_shift_latest_csv": str(OUTDIR / f"segment_rank_shift_extreme_vs_normal_{latest_year}.csv"),
            "segment_risk_map_png": str(map_path),
            "latest_year": latest_year,
        },
        "ablations": exp_results,
    }
    with open(OUTDIR / "panel_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print("\n" + "="*65)
    print("RESULTS")
    print("="*65)
    print(f"  Panel size             : {len(panel):,} location-days")
    print(f"  Confirmed ignitions    : {int(panel['ignition'].sum()):,}  (2021-2024 only)")
    print(f"  Base rate              : {panel['ignition'].mean()*100:.4f}%")
    print(f"  β (Pozzolo)            : {beta:.5f}")
    print(f"  Val  AUC-ROC (2023)    : {val_m['auc']:.4f}")
    print(f"  Test AUC-ROC (2024)    : {test_m['auc']:.4f}")
    print(f"  Test PR-AUC  (2024)    : {test_m['pr_auc']:.4f}")
    print(f"  Test Brier   (2024)    : {test_m['brier']:.6f}")
    print(f"  corr(raw, cal)         : {test_m['corr_raw_cal']:.4f}  ({'✓ correct' if test_m['corr_raw_cal'] > 0 else '✗ INVERTED'})")
    for row in topk_metrics["topk"]:
        pct = row["top_fraction"] * 100
        print(f"  Top {pct:>4.1f}% capture      : {row['capture_recall']:.4f}  lift={row['lift_vs_base_rate']:.2f}x")
    print(f"  Calibration ECE (2024) : {ece_test:.6f}")
    print(f"  Seasonal risk CSV       : {(OUTDIR / 'segment_seasonal_risk.csv').resolve()}")
    print(f"  Outputs                : {OUTDIR.resolve()}")
    print("="*65)


if __name__ == "__main__":
    main()
