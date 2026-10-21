"""
compare_cluster_counts.py
=========================
Compare ignition-model performance across different cluster counts using
the same final feature recipe as ignition_probability.py.

Run:
  .venv/bin/python compare_cluster_counts.py
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

import ignition_probability as ip

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTDIR / "cluster_count_comparison.csv"
OUT_JSON = OUTDIR / "cluster_count_comparison.json"

CLUSTER_COUNTS = (500, 750, 1000)

HFTD_CANDIDATES = [
    HERE / "Electrical_Grid" / "cpuc_hftd" / "CPUC_FireThreatMapTiers2_3_v3r_08_19_2021.gdb",
    HERE / "environmental_data" / "hftd.gdb",
]


def _find_hftd_gdb() -> Path | None:
    for p in HFTD_CANDIDATES:
        if p.exists():
            return p
    return None


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


def add_hftd_by_spatial_join(centroids: pd.DataFrame) -> pd.DataFrame:
    gdb = _find_hftd_gdb()
    out = centroids.copy()
    out["hftd_tier"] = "non_hftd"
    if gdb is None:
        log.warning("HFTD geodatabase not found. Using non_hftd for all clusters.")
        return out

    layer = _pick_layer(gdb)
    hftd = gpd.read_file(gdb, layer=layer)
    if hftd.empty or "HFTD" not in hftd.columns:
        log.warning("HFTD data missing/empty. Using non_hftd for all clusters.")
        return out
    hftd = hftd[["HFTD", "geometry"]].copy()
    hftd = hftd[hftd.geometry.notna() & ~hftd.geometry.is_empty].copy()
    if hftd.crs is None:
        hftd = hftd.set_crs(epsg=3310, allow_override=True)
    else:
        hftd = hftd.to_crs(epsg=3310)
    hftd["hftd_tier"] = hftd["HFTD"].map(_normalize_hftd)

    pts = gpd.GeoDataFrame(
        out[["cluster_id", "cluster_lat", "cluster_lon"]].copy(),
        geometry=gpd.points_from_xy(out["cluster_lon"], out["cluster_lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3310)
    joined = gpd.sjoin(pts, hftd[["hftd_tier", "geometry"]], how="left", predicate="within")
    tier = (
        joined[["cluster_id", "hftd_tier"]]
        .drop_duplicates("cluster_id")
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )
    tier["hftd_tier"] = tier["hftd_tier"].fillna("non_hftd")
    out = out.merge(tier, on="cluster_id", how="left", suffixes=("", "_j"))
    if "hftd_tier_j" in out.columns:
        out["hftd_tier"] = out["hftd_tier_j"].fillna(out["hftd_tier"]).fillna("non_hftd")
        out = out.drop(columns=["hftd_tier_j"])
    out["hftd_tier"] = out["hftd_tier"].fillna("non_hftd")
    return out


def run_single(n_clusters: int, incidents: pd.DataFrame) -> dict[str, Any]:
    log.info("=" * 70)
    log.info(f"Running cluster-count comparison for n_clusters={n_clusters}")
    ip.N_CLUSTERS = int(n_clusters)

    df_inc, centroids = ip.cluster_locations(incidents)
    centroids = ip.add_topography_to_centroids(centroids)
    centroids = add_hftd_by_spatial_join(centroids)

    panel = ip.build_panel(df_inc, centroids)
    panel = ip.join_gridmet(panel)
    panel = ip.add_rolling_weather_features(panel)
    panel = ip.add_interaction_features(panel)
    panel = ip.add_ignition_history_features(panel)

    train, val, test = ip.time_split(panel)
    train_us, beta = ip.undersample(train, ratio=200.0, seed=42)

    numeric = ip.BASE_NUMERIC + ip.COORD_COLS + ip.TOPO_COLS
    categorical = ip.BASE_CATEGORICAL + ip.FUEL_COLS + ip.HFTD_COLS
    feats = numeric + categorical

    prep = ip.build_preprocessor(numeric, categorical)
    X_tr = prep.fit_transform(train_us[feats])
    X_val = prep.transform(val[feats])
    X_te = prep.transform(test[feats])
    y_tr = train_us[ip.LABEL_COL].values
    y_val = val[ip.LABEL_COL].values
    y_te = test[ip.LABEL_COL].values

    model = ip.train_rf(X_tr, y_tr)
    val_m, _, _ = ip.evaluate(model, X_val, y_val, beta, f"n={n_clusters} | val 2023")
    test_m, p_te_cal, _ = ip.evaluate(model, X_te, y_te, beta, f"n={n_clusters} | test 2024")
    topk = ip.compute_topk_capture_lift(y_te, p_te_cal)

    out = {
        "n_clusters": int(n_clusters),
        "panel_size": int(len(panel)),
        "n_ignitions": int(panel[ip.LABEL_COL].sum()),
        "base_rate_pct": float(panel[ip.LABEL_COL].mean() * 100),
        "beta": float(beta),
        "val_2023": val_m,
        "test_2024": test_m,
        "topk_test_2024": topk,
    }
    return out


def main() -> None:
    incidents = ip.load_incidents(ip.INPUT_CSV)
    results: list[dict[str, Any]] = []
    for n in CLUSTER_COUNTS:
        res = run_single(int(n), incidents)
        results.append(res)

    rows = []
    for r in results:
        top10 = next((x for x in r["topk_test_2024"]["topk"] if abs(x["top_fraction"] - 0.01) < 1e-12), None)
        rows.append(
            {
                "n_clusters": r["n_clusters"],
                "panel_size": r["panel_size"],
                "n_ignitions": r["n_ignitions"],
                "base_rate_pct": r["base_rate_pct"],
                "val_auc_2023": r["val_2023"]["auc"],
                "test_auc_2024": r["test_2024"]["auc"],
                "test_pr_auc_2024": r["test_2024"]["pr_auc"],
                "test_brier_2024": r["test_2024"]["brier"],
                "top1pct_capture_2024": (top10["capture_recall"] if top10 else None),
                "top1pct_lift_2024": (top10["lift_vs_base_rate"] if top10 else None),
            }
        )

    df = pd.DataFrame(rows).sort_values("n_clusters").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w") as f:
        json.dump({"results": results}, f, indent=2)

    print("\nCluster-count comparison complete")
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_JSON}")
    print("\nSummary (test 2024):")
    print(df[["n_clusters", "test_auc_2024", "test_pr_auc_2024", "test_brier_2024", "top1pct_capture_2024", "top1pct_lift_2024"]].to_string(index=False))


if __name__ == "__main__":
    main()

