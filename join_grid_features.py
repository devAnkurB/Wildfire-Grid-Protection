"""
join_grid_features.py
=====================
Create cluster-level grid features for ignition modeling.

Current outputs:
  - cluster_id
  - hftd_tier

This script reproduces the same clustering logic as ignition_probability.py so
cluster IDs align with the ignition panel model.
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ignition_probability import (
    INPUT_CSV,
    N_CLUSTERS,
    add_topography_to_centroids,
    add_transmission_proxy_to_centroids,
    cluster_locations,
    load_incidents,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTDIR / "cluster_grid_features.csv"

HFTD_CANDIDATES = [
    HERE / "Electrical_Grid" / "cpuc_hftd" / "CPUC_FireThreatMapTiers2_3_v3r_08_19_2021.gdb",
    HERE / "environmental_data" / "hftd.gdb",
]


def _find_hftd_gdb() -> Path:
    for p in HFTD_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No HFTD geodatabase found. Expected one of:\n"
        + "\n".join(f"  - {p}" for p in HFTD_CANDIDATES)
    )


def _pick_layer(gdb_path: Path) -> str:
    import fiona

    layers = fiona.listlayers(gdb_path)
    if not layers:
        raise RuntimeError(f"No layers found in {gdb_path}")
    preferred = [lyr for lyr in layers if "FireThreat" in lyr or "HFTD" in lyr]
    return preferred[0] if preferred else layers[0]


def _normalize_hftd(value: object) -> str:
    txt = str(value).strip().lower()
    txt = re.sub(r"[^a-z0-9]+", " ", txt)
    if "tier 3" in txt or txt == "3":
        return "tier3"
    if "tier 2" in txt or txt == "2":
        return "tier2"
    if "tier 1" in txt or "hhz" in txt:
        return "tier1_hhz"
    return "non_hftd"


def add_hftd_by_spatial_join(centroids: pd.DataFrame, gdb_path: Path) -> pd.DataFrame:
    layer = _pick_layer(gdb_path)
    log.info(f"HFTD source: {gdb_path.name} | layer: {layer}")

    warnings.filterwarnings("ignore", message="organizePolygons.*", category=RuntimeWarning)
    hftd = gpd.read_file(gdb_path, layer=layer)
    if hftd.empty:
        raise RuntimeError("Loaded HFTD layer is empty.")

    if "HFTD" not in hftd.columns:
        raise KeyError(f"HFTD field not found in layer. Columns: {list(hftd.columns)}")

    hftd = hftd[["HFTD", "geometry"]].copy()
    hftd = hftd[hftd.geometry.notna() & ~hftd.geometry.is_empty].copy()
    if hftd.crs is None:
        hftd = hftd.set_crs(epsg=3310, allow_override=True)
    else:
        hftd = hftd.to_crs(epsg=3310)
    hftd["hftd_tier"] = hftd["HFTD"].map(_normalize_hftd)

    pts = gpd.GeoDataFrame(
        centroids[["cluster_id", "cluster_lat", "cluster_lon"]].copy(),
        geometry=gpd.points_from_xy(centroids["cluster_lon"], centroids["cluster_lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3310)

    # Point-in-polygon assignment to get cluster-level HFTD tier.
    joined = gpd.sjoin(
        pts,
        hftd[["hftd_tier", "geometry"]],
        how="left",
        predicate="within",
    )
    joined["hftd_tier"] = joined["hftd_tier"].fillna("non_hftd")

    # Safety: keep one row per cluster.
    out = (
        joined[["cluster_id", "hftd_tier"]]
        .drop_duplicates(subset=["cluster_id"], keep="first")
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )
    return out


def main() -> None:
    log.info("STEP 1: Load incidents and build model-consistent centroids")
    incidents = load_incidents(INPUT_CSV)
    _, centroids = cluster_locations(incidents)
    if len(centroids) != N_CLUSTERS:
        raise RuntimeError(f"Unexpected centroid count: {len(centroids)} != {N_CLUSTERS}")

    log.info("STEP 2: Add static features (topography + transmission proxy)")
    centroids = add_topography_to_centroids(centroids)
    centroids = add_transmission_proxy_to_centroids(centroids)

    log.info("STEP 3: Spatial join HFTD tier")
    hftd_gdb = _find_hftd_gdb()
    hftd_df = add_hftd_by_spatial_join(centroids, hftd_gdb)

    out = centroids.merge(hftd_df, on="cluster_id", how="left")
    out["hftd_tier"] = out["hftd_tier"].fillna("non_hftd")

    cols = [
        "cluster_id",
        "cluster_lat",
        "cluster_lon",
        "elevation_m",
        "slope_deg",
        "tx_dist_km",
        "tx_len_km_10km",
        "tx_count_10km",
        "hftd_tier",
    ]
    out = out[cols].sort_values("cluster_id").reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    counts = out["hftd_tier"].value_counts()
    log.info(f"Saved: {OUT_CSV}")
    log.info(f"Rows: {len(out)}")
    for tier, cnt in counts.items():
        log.info(f"  {tier:<10}: {cnt}")


if __name__ == "__main__":
    main()

