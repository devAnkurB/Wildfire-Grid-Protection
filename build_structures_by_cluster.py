"""
build_structures_by_cluster.py
==============================
Create outputs/structures_by_cluster.csv for severity modeling.

What it does
------------
1) Loads cluster centroids from outputs/panel_full.csv
2) Builds Voronoi cluster polygons from centroids
3) Clips Voronoi polygons to California boundary
4) Streams building features from an input vector file (GeoJSON/GeoPackage/Shapefile)
5) Assigns each building to nearest cluster centroid
6) Computes structures_per_sqmi = building_count / clipped_cluster_area_sqmi

This is memory-efficient for large building datasets because it streams features.

Default paths
-------------
- panel: outputs/panel_full.csv
- CA boundary: tl_2025_us_state/tl_2025_us_state.shp
- output: outputs/structures_by_cluster.csv

Example
-------
.venv/bin/python build_structures_by_cluster.py \
  --buildings environmental_data/structures/California.geojson
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import Voronoi, cKDTree
from shapely.geometry import MultiPolygon, Polygon, Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent

PANEL_DEFAULT = HERE / "outputs" / "panel_full.csv"
CA_STATES_DEFAULT = HERE / "tl_2025_us_state" / "tl_2025_us_state.shp"
OUT_DEFAULT = HERE / "outputs" / "structures_by_cluster.csv"

CA_ALBERS_EPSG = 3310
SQM_PER_SQMI = 2_589_988.110336


def voronoi_finite_polygons_2d(vor: Voronoi, radius: float | None = None):
    """Reconstruct infinite Voronoi regions to finite polygons."""
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]

        if region and all(v >= 0 for v in region):
            new_regions.append(region)
            continue

        ridges = all_ridges.get(p1, [])
        new_region = [v for v in (region or []) if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]

        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def load_cluster_centroids(panel_csv: Path) -> gpd.GeoDataFrame:
    panel = pd.read_csv(panel_csv, usecols=["cluster_id", "cluster_lat", "cluster_lon"])
    c = panel.drop_duplicates("cluster_id").copy()
    c["cluster_id"] = c["cluster_id"].astype(int)
    g = gpd.GeoDataFrame(
        c,
        geometry=gpd.points_from_xy(c["cluster_lon"], c["cluster_lat"]),
        crs="EPSG:4326",
    )
    return g


def load_california_polygon(states_shp: Path) -> gpd.GeoDataFrame:
    states = gpd.read_file(states_shp)
    if "STATEFP" in states.columns:
        ca = states[states["STATEFP"].astype(str).str.zfill(2) == "06"].copy()
    elif "STUSPS" in states.columns:
        ca = states[states["STUSPS"] == "CA"].copy()
    elif "NAME" in states.columns:
        ca = states[states["NAME"].str.upper() == "CALIFORNIA"].copy()
    else:
        raise ValueError("Cannot identify California in state shapefile.")

    if ca.empty:
        raise ValueError("California polygon not found in state shapefile.")

    ca = ca.to_crs(epsg=CA_ALBERS_EPSG)
    ca = gpd.GeoDataFrame(geometry=[unary_union(ca.geometry)], crs=f"EPSG:{CA_ALBERS_EPSG}")
    return ca


def build_cluster_polygons(centroids_wgs84: gpd.GeoDataFrame, ca_poly: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cent = centroids_wgs84.to_crs(epsg=CA_ALBERS_EPSG).copy()
    pts = np.column_stack([cent.geometry.x.values, cent.geometry.y.values])

    vor = Voronoi(pts)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    polys = []
    for region in regions:
        poly = Polygon(vertices[region])
        polys.append(poly)

    vdf = gpd.GeoDataFrame(
        cent[["cluster_id"]].copy(),
        geometry=polys,
        crs=f"EPSG:{CA_ALBERS_EPSG}",
    )

    ca_geom = ca_poly.geometry.iloc[0]
    vdf["geometry"] = vdf.geometry.intersection(ca_geom)
    vdf = vdf[~vdf.geometry.is_empty].copy()

    vdf["cluster_area_sqmi"] = vdf.geometry.area / SQM_PER_SQMI
    return vdf


def assign_buildings_to_nearest_cluster(
    buildings_path: Path,
    centroids_wgs84: gpd.GeoDataFrame,
    ca_poly_albers: gpd.GeoDataFrame,
) -> pd.Series:
    cent_albers = centroids_wgs84.to_crs(epsg=CA_ALBERS_EPSG)
    xy = np.column_stack([cent_albers.geometry.x.values, cent_albers.geometry.y.values])
    cluster_ids = cent_albers["cluster_id"].values
    tree = cKDTree(xy)

    ca_geom = ca_poly_albers.geometry.iloc[0]
    ca_prepared = prep(ca_geom)

    counts = pd.Series(0, index=cluster_ids, dtype="int64")

    with fiona.open(buildings_path) as src:
        src_crs = src.crs_wkt or src.crs
        if src_crs:
            transformer = Transformer.from_crs(src_crs, f"EPSG:{CA_ALBERS_EPSG}", always_xy=True)
        else:
            log.warning("Buildings CRS missing; assuming EPSG:4326")
            transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{CA_ALBERS_EPSG}", always_xy=True)

        n = 0
        kept = 0

        for feat in src:
            n += 1
            geom = feat.get("geometry")
            if geom is None:
                continue

            g = shape(geom)
            if g.is_empty:
                continue

            rp = g.representative_point()
            x, y = transformer.transform(rp.x, rp.y)
            p = Point(x, y)

            if not ca_prepared.contains(p):
                continue

            _, idx = tree.query([x, y], k=1)
            cid = int(cluster_ids[int(idx)])
            counts.loc[cid] += 1
            kept += 1

            if n % 200000 == 0:
                log.info(f"Processed {n:,} buildings | kept in CA {kept:,}")

    log.info(f"Finished buildings pass: processed {n:,}, kept in CA {kept:,}")
    return counts


def build_structures_by_cluster(
    buildings_path: Path,
    panel_csv: Path,
    states_shp: Path,
    out_csv: Path,
):
    if not buildings_path.exists():
        raise FileNotFoundError(f"Buildings file not found: {buildings_path}")
    if not panel_csv.exists():
        raise FileNotFoundError(f"Panel file not found: {panel_csv}")
    if not states_shp.exists():
        raise FileNotFoundError(f"State shapefile not found: {states_shp}")

    log.info("Loading cluster centroids...")
    centroids = load_cluster_centroids(panel_csv)

    log.info("Loading California polygon...")
    ca_poly = load_california_polygon(states_shp)

    log.info("Building Voronoi cluster polygons clipped to California...")
    cluster_polys = build_cluster_polygons(centroids, ca_poly)

    log.info("Assigning buildings to nearest cluster centroid (streaming)...")
    counts = assign_buildings_to_nearest_cluster(buildings_path, centroids, ca_poly)

    out = cluster_polys[["cluster_id", "cluster_area_sqmi"]].copy()
    out = out.merge(counts.rename("n_structures").reset_index().rename(columns={"index": "cluster_id"}),
                    on="cluster_id", how="left")
    out["n_structures"] = out["n_structures"].fillna(0).astype(int)
    out["structures_per_sqmi"] = np.where(
        out["cluster_area_sqmi"] > 0,
        out["n_structures"] / out["cluster_area_sqmi"],
        np.nan,
    )

    result = out[["cluster_id", "structures_per_sqmi"]].sort_values("cluster_id")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)

    log.info(f"Saved {len(result):,} clusters -> {out_csv}")
    log.info(
        "Coverage: non-null density %.1f%% | mean density %.2f",
        result["structures_per_sqmi"].notna().mean() * 100.0,
        result["structures_per_sqmi"].mean(skipna=True),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build structures_by_cluster.csv")
    p.add_argument("--buildings", type=Path, required=True,
                   help="Path to buildings vector file (GeoJSON/GPKG/SHP)")
    p.add_argument("--panel", type=Path, default=PANEL_DEFAULT,
                   help=f"Path to panel_full.csv (default: {PANEL_DEFAULT})")
    p.add_argument("--states", type=Path, default=CA_STATES_DEFAULT,
                   help=f"Path to US states shapefile (default: {CA_STATES_DEFAULT})")
    p.add_argument("--out", type=Path, default=OUT_DEFAULT,
                   help=f"Output CSV path (default: {OUT_DEFAULT})")
    return p.parse_args()


def main():
    args = parse_args()
    build_structures_by_cluster(
        buildings_path=args.buildings,
        panel_csv=args.panel,
        states_shp=args.states,
        out_csv=args.out,
    )


if __name__ == "__main__":
    main()
