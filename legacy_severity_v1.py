# LEGACY: superseded by severity_hurdle_model.py. Retained for reproducibility only.
"""
severity_model.py  v3
=====================
Step 2 of the wildfire-grid risk workflow.

Builds:
  E[damage | ignition, cluster] = E[structures_exposed | ignition, cluster]
                                 * damage_per_structure_usd

Then computes:
  expected_loss_usd = P(>=1 ignition | cluster, year) * E[damage | ignition, cluster]

And assigns each cluster a cost-justified intervention recommendation.

-----------------------------------------------------------------------------
INTERVENTION LOGIC -- grounded in Warner, Callaway & Fowlie (2025) and
CPUC Wildfire Mitigation Plan filings
-----------------------------------------------------------------------------

1. UNDERGROUNDING
   Capital cost:  ~$3-5M per circuit mile (CPUC WMP filings, 2023)
   Assuming ~10 circuit miles per cluster -> ~$40M capital
   Annualized over 30yr at 5% discount rate -> ~$2.6M/yr
   Effect: eliminates overhead ignition risk (delta_P ~= 100%)
   Cost-justified when: EL > $2.6M/yr AND structures_per_sqmi > 500
   The structure-density condition targets the wildland-urban interface
   where consequence is high and fast-trip settings alone are insufficient.
   Warner et al. (2025) find undergrounding costs ~7x more per ignition
   avoided than fast-trip, so it is reserved for highest-consequence clusters.

2. FAST-TRIP SETTINGS + COVERED CONDUCTOR (hardening)
   Fast-trip: near-zero capital; reduces ignitions ~80% (Warner 2025)
   Covered conductor: ~$500K-1M per circuit mile
   Combined annualized cost: ~$100-300K per cluster
   Cost-justified when: EL > $1M/yr
   (10yr PV of $1M/yr at 5% = $7.7M >> $300K cost)

3. VEGETATION MANAGEMENT
   Cost: ~$500-800/acre/yr (CPUC EVM program data)
   ~300-500 acres per cluster -> ~$150-400K/yr
   Cost-justified when: $200K < EL <= $1M

4. TARGETED MONITORING
   Near-zero cost; preserves optionality for future reassessment.
   Applied when: EL <= $200K/yr

-----------------------------------------------------------------------------
KEY FIXES vs v2
-----------------------------------------------------------------------------
  1. Annualized ignition probability uses complement rule:
       P(>=1 ignition in year) = 1 - (1 - mean_daily_p)^365
     v1 used mean_daily_p * 365, exceeding 1.0 for most clusters.

  2. Size model: LogisticRegression -> RandomForestClassifier with
     class_weight='balanced'. With 374/66/12 distribution, logistic
     regression collapsed to predicting the majority class exclusively.

  3. Intervention thresholds: relative quantile splits -> absolute
     cost-justified EL thresholds from Warner et al. (2025) and CPUC
     WMP filings. Quantile thresholds are arbitrary and change with
     sample composition; cost-justified thresholds are stable and
     directly comparable across studies.

Run:
  python severity_model.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

HERE      = Path(__file__).resolve().parent
OUTDIR    = HERE / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

INCIDENTS_CSV  = HERE / "fire_incidents" / "incidents_features.csv"
PANEL_CSV      = OUTDIR / "panel_full.csv"
STRUCTURES_CSV = OUTDIR / "structures_by_cluster.csv"

SIZE_BUCKETS = ["0.25_to_10ac", "10_to_100ac", "over_100ac"]

BUCKET_AREA_AC = {
    "0.25_to_10ac":  5.0,
    "10_to_100ac":  55.0,
    "over_100ac":  200.0,
}

DEFAULT_DAMAGE_PER_STRUCTURE = 500_000.0
# Total economic loss per destroyed CA structure: insured + uninsured + indirect.
# CoreLogic (2020): median insured loss ~$250-350K; total ~1.5-2x insured -> ~$500K.

# Intervention EL thresholds (USD/yr) -- see module docstring for derivation
THRESHOLD_UNDERGROUNDING_EL     = 2_600_000
THRESHOLD_UNDERGROUNDING_STRUCT = 500         # structures per sq mi
THRESHOLD_HARDENING_EL          = 1_000_000
THRESHOLD_VEGETATION_EL         =   200_000

SEASON_MAP = {
    1: "winter", 2: "winter",  3: "spring", 4: "spring",  5: "spring",
    6: "summer", 7: "summer",  8: "summer", 9: "fall",   10: "fall",
    11: "fall",  12: "winter",
}


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

def nearest_cluster_id(lat: float, lon: float, centroids: pd.DataFrame) -> int:
    d2 = ((centroids["cluster_lat"].values - lat) ** 2
          + (centroids["cluster_lon"].values - lon) ** 2)
    return int(centroids.iloc[int(np.argmin(d2))]["cluster_id"])


def assign_incident_clusters(df_inc: pd.DataFrame,
                              centroids: pd.DataFrame) -> pd.DataFrame:
    out = []
    for lat, lon in df_inc[["latitude", "longitude"]].itertuples(index=False):
        if pd.isna(lat) or pd.isna(lon):
            out.append(np.nan)
        else:
            out.append(nearest_cluster_id(float(lat), float(lon), centroids))
    df_inc = df_inc.copy()
    df_inc["cluster_id"] = pd.Series(out, index=df_inc.index, dtype="Int64")
    return df_inc


def load_base_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("Loading panel and incident data...")
    panel     = pd.read_csv(PANEL_CSV,     low_memory=False)
    incidents = pd.read_csv(INCIDENTS_CSV, low_memory=False)

    panel["date"]     = pd.to_datetime(panel["date"],     errors="coerce")
    incidents["date"] = pd.to_datetime(incidents["date"], errors="coerce")

    centroid_cols = ["cluster_id", "cluster_lat", "cluster_lon", "fuel_category"]
    centroids = panel[centroid_cols].drop_duplicates("cluster_id").copy()

    for col in ["elevation_m", "slope_deg"]:
        if col in panel.columns:
            centroids = centroids.merge(
                panel[["cluster_id", col]].drop_duplicates("cluster_id"),
                on="cluster_id", how="left"
            )
        else:
            centroids[col] = np.nan

    incidents = incidents[
        incidents["date"].notna()
        & incidents["latitude"].notna()
        & incidents["longitude"].notna()
    ].copy()
    incidents["season"] = incidents["date"].dt.month.map(SEASON_MAP)
    incidents = assign_incident_clusters(incidents, centroids)
    incidents = incidents[incidents["cluster_id"].notna()].copy()
    incidents["cluster_id"] = incidents["cluster_id"].astype(int)
    incidents = incidents.merge(
        centroids[["cluster_id", "fuel_category", "slope_deg"]],
        on="cluster_id", how="left", suffixes=("", "_cluster")
    )
    if "fuel_category_cluster" in incidents.columns:
        incidents["fuel_category"] = (
            incidents["fuel_category_cluster"]
            .fillna(incidents.get("fuel_category", np.nan))
        )
        incidents.drop(columns=["fuel_category_cluster"], inplace=True)

    log.info(f"  Panel:     {len(panel):,} rows | {panel['cluster_id'].nunique()} clusters")
    log.info(f"  Incidents: {len(incidents):,} rows")
    return panel, centroids, incidents


# -----------------------------------------------------------------------------
# FIRE SIZE DISTRIBUTION MODEL
# -----------------------------------------------------------------------------

def fit_size_model(incidents: pd.DataFrame) -> tuple[Pipeline, dict]:
    """
    Random Forest: P(size_bucket | fuel_category, slope_deg, wind_ms, season).
    Restricted to 2021+ where CPUC reports acreage (not linear distance).
    class_weight='balanced' corrects for 374/66/12 class distribution.

    Note: with only 12 large-fire training examples the model cannot reliably
    predict individual fire sizes. Its value is the calibrated probability
    distribution from which E[burned_area] is computed as a weighted sum.
    """
    df = incidents[
        (incidents["confirmed_ignition"] == 1)
        & (incidents["fire_size_bucket"].isin(SIZE_BUCKETS))
        & (incidents["date"].dt.year >= 2021)
    ].copy()

    if df.empty:
        raise ValueError("No confirmed-ignition rows for size modeling.")

    log.info(f"  Size model: {len(df)} events | {df['fire_size_bucket'].value_counts().to_dict()}")

    features     = ["fuel_category", "slope_deg", "wind_ms", "season"]
    num_features = ["slope_deg", "wind_ms"]
    cat_features = ["fuel_category", "season"]

    X = df[features].copy()
    y = pd.Categorical(df["fire_size_bucket"], categories=SIZE_BUCKETS).codes

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp",   SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_features),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_features),
    ])

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model = Pipeline([("pre", pre), ("clf", clf)])

    train_mask = df["date"].dt.year <= 2023
    X_tr, y_tr = X.loc[train_mask],  y[train_mask.values]
    X_te, y_te = X.loc[~train_mask], y[~train_mask.values]
    model.fit(X_tr, y_tr)

    metrics = {
        "n_confirmed_ignitions_used": int(len(df)),
        "bucket_counts": df["fire_size_bucket"].value_counts().to_dict(),
        "note": (
            "Primary output is E[burned_area_ac] computed as probability-weighted "
            "sum across buckets, not individual class predictions. "
            "Classification accuracy reported for transparency only."
        ),
    }
    if len(X_te) > 0 and len(np.unique(y_te)) > 1:
        pred   = model.predict(X_te)
        report = classification_report(y_te, pred, output_dict=True, zero_division=0)
        metrics["size_model_test_2024_accuracy"] = report["accuracy"]
        metrics["size_model_test_2024_report"]   = report
        log.info(f"  Test accuracy: {report['accuracy']:.3f} (informational only)")

    return model, metrics


# -----------------------------------------------------------------------------
# EXPECTED BURNED AREA BY CLUSTER
# -----------------------------------------------------------------------------

def cluster_season_scenarios(panel: pd.DataFrame,
                              centroids: pd.DataFrame) -> pd.DataFrame:
    base = panel[["cluster_id", "season", "wind_ms", "p_ignition_cal"]].copy()
    wind = base.groupby(["cluster_id","season"], as_index=False)["wind_ms"].mean()
    s_w  = (base.groupby(["cluster_id","season"], as_index=False)["p_ignition_cal"]
            .mean().rename(columns={"p_ignition_cal": "season_weight_raw"}))
    scen = wind.merge(s_w, on=["cluster_id","season"], how="left")
    scen = scen.merge(
        centroids[["cluster_id","fuel_category","slope_deg"]],
        on="cluster_id", how="left"
    )
    scen["season_weight_raw"] = scen["season_weight_raw"].fillna(0.0)
    total = scen.groupby("cluster_id")["season_weight_raw"].transform("sum")
    scen["season_weight"] = np.where(total > 0, scen["season_weight_raw"]/total, 0.25)
    norm = scen.groupby("cluster_id")["season_weight"].transform("sum")
    scen["season_weight"] = np.where(norm > 0, scen["season_weight"]/norm, scen["season_weight"])
    return scen


def expected_burned_area_by_cluster(size_model: Pipeline,
                                     panel: pd.DataFrame,
                                     centroids: pd.DataFrame) -> pd.DataFrame:
    """
    E[burned_area | ignition, cluster] = season-weighted sum of
      sum_b( P(bucket=b | features) * midpoint_area_b )
    Correct actuarial expectation; more stable than individual predictions.
    """
    scen  = cluster_season_scenarios(panel, centroids)
    Xs    = scen[["fuel_category","slope_deg","wind_ms","season"]].copy()
    probs = size_model.predict_proba(Xs)
    p_df  = pd.DataFrame(probs, columns=[f"p_{b}" for b in SIZE_BUCKETS], index=scen.index)
    scen  = pd.concat([scen, p_df], axis=1)
    scen["expected_area_ac"] = sum(scen[f"p_{b}"] * BUCKET_AREA_AC[b] for b in SIZE_BUCKETS)
    e_area = (
        scen.assign(weighted_area=lambda d: d["expected_area_ac"] * d["season_weight"])
        .groupby("cluster_id", as_index=False)["weighted_area"].sum()
        .rename(columns={"weighted_area": "expected_burned_area_ac"})
    )
    return e_area


# -----------------------------------------------------------------------------
# STRUCTURE DENSITY
# -----------------------------------------------------------------------------

def load_structure_density(centroids: pd.DataFrame) -> pd.DataFrame:
    if STRUCTURES_CSV.exists():
        st = pd.read_csv(STRUCTURES_CSV)
        needed = {"cluster_id", "structures_per_sqmi"}
        if not needed.issubset(set(st.columns)):
            raise ValueError(f"{STRUCTURES_CSV} must have: {sorted(needed)}")
        st = st[["cluster_id","structures_per_sqmi"]].copy()
        st["structures_per_sqmi"] = pd.to_numeric(st["structures_per_sqmi"], errors="coerce")
        log.info(f"  Structure density: {st['structures_per_sqmi'].notna().sum()} / {len(st)} clusters")
        return st
    log.warning("structures_by_cluster.csv not found -- density will be NaN.")
    st = centroids[["cluster_id"]].copy()
    st["structures_per_sqmi"] = np.nan
    return st


# -----------------------------------------------------------------------------
# INTERVENTION RECOMMENDATION
# -----------------------------------------------------------------------------

def recommend_mitigation(el: float, structures_per_sqmi: float) -> str:
    """
    Cost-justified intervention based on expected annual loss and structure
    density. Thresholds derived from annualized intervention costs per cluster.
    See module docstring for full derivation and citations.
    """
    if (el > THRESHOLD_UNDERGROUNDING_EL
            and structures_per_sqmi > THRESHOLD_UNDERGROUNDING_STRUCT):
        return "undergrounding"
    if el > THRESHOLD_HARDENING_EL:
        return "fast_trip_and_hardening"
    if el > THRESHOLD_VEGETATION_EL:
        return "vegetation_management"
    return "targeted_monitoring"


# -----------------------------------------------------------------------------
# SEVERITY TABLE
# -----------------------------------------------------------------------------

def build_severity_table(
    panel: pd.DataFrame,
    centroids: pd.DataFrame,
    e_area: pd.DataFrame,
    structures: pd.DataFrame,
    damage_per_structure: float = DEFAULT_DAMAGE_PER_STRUCTURE,
) -> pd.DataFrame:

    # Complement-rule annualized ignition probability
    risk = (
        panel.groupby("cluster_id", as_index=False)["p_ignition_cal"]
        .mean().rename(columns={"p_ignition_cal": "mean_daily_ignition_prob"})
    )
    risk["annualized_ignition_prob"] = 1.0 - (1.0 - risk["mean_daily_ignition_prob"]) ** 365

    out = centroids[["cluster_id","cluster_lat","cluster_lon",
                      "fuel_category","slope_deg","elevation_m"]].copy()
    out = out.merge(risk,       on="cluster_id", how="left")
    out = out.merge(e_area,     on="cluster_id", how="left")
    out = out.merge(structures, on="cluster_id", how="left")

    out["expected_burned_area_sqmi"]   = out["expected_burned_area_ac"] / 640.0
    out["expected_structures_exposed"] = out["expected_burned_area_sqmi"] * out["structures_per_sqmi"]
    out["damage_per_structure_usd"]    = float(damage_per_structure)
    out["severity_usd_given_ignition"] = out["expected_structures_exposed"] * out["damage_per_structure_usd"]
    # EL(s) = P(>=1 ignition | year, s) * E[damage | ignition, s]
    out["expected_loss_usd"] = out["annualized_ignition_prob"] * out["severity_usd_given_ignition"]

    out["recommended_mitigation"] = [
        recommend_mitigation(el, struct)
        for el, struct in zip(
            out["expected_loss_usd"].fillna(0.0),
            out["structures_per_sqmi"].fillna(0.0),
        )
    ]
    return out


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    panel, centroids, incidents = load_base_data()

    log.info("Fitting fire size distribution model...")
    size_model, size_metrics = fit_size_model(incidents)

    log.info("Computing E[burned area | ignition, cluster]...")
    e_area = expected_burned_area_by_cluster(size_model, panel, centroids)

    log.info("Loading structure density...")
    structures = load_structure_density(centroids)

    log.info("Building severity + expected loss table...")
    severity = build_severity_table(
        panel=panel, centroids=centroids, e_area=e_area,
        structures=structures, damage_per_structure=DEFAULT_DAMAGE_PER_STRUCTURE,
    )

    total_el  = severity["expected_loss_usd"].sum()
    df_sorted = severity.sort_values("expected_loss_usd", ascending=False)

    log.info(f"\n{'='*65}")
    log.info(f"  Total portfolio expected annual loss: ${total_el:,.0f}")
    log.info(f"  Mean / Median / Max per cluster: "
             f"${severity['expected_loss_usd'].mean():,.0f} / "
             f"${severity['expected_loss_usd'].median():,.0f} / "
             f"${severity['expected_loss_usd'].max():,.0f}")
    log.info(f"")
    log.info(f"  Pareto concentration:")
    for pct in [0.05, 0.10, 0.20]:
        n   = max(1, int(300*pct))
        top = df_sorted.head(n)["expected_loss_usd"].sum()
        log.info(f"    Top {int(pct*100):2d}% ({n:3d} clusters): "
                 f"${top:,.0f} = {top/total_el*100:.1f}% of total EL")
    log.info(f"")
    log.info(f"  Interventions:")
    mit = severity.groupby("recommended_mitigation")["expected_loss_usd"].agg(["count","sum"])
    mit["pct"] = mit["sum"]/total_el*100
    for row in mit.sort_values("sum", ascending=False).itertuples():
        log.info(f"    {row.Index:<35}: n={row.count:3d} ({row.count/300*100:.0f}%)  "
                 f"EL_share={row.pct:.1f}%")
    log.info(f"{'='*65}")

    out_csv = OUTDIR / "severity_by_cluster.csv"
    severity.to_csv(out_csv, index=False)

    concentration = {
        f"top_{int(p*100)}pct_el_share": float(
            df_sorted.head(max(1,int(300*p)))["expected_loss_usd"].sum() / total_el
        ) for p in [0.05, 0.10, 0.20, 0.50]
    }

    metrics = {
        "methodology": {
            "annualized_prob_formula": "1 - (1 - mean_daily_p)^365",
            "damage_per_structure_usd": DEFAULT_DAMAGE_PER_STRUCTURE,
            "damage_basis": "CoreLogic (2020): insured + uninsured + indirect economic loss per structure",
            "size_bucket_area_ac": BUCKET_AREA_AC,
            "intervention_thresholds_usd_yr": {
                "undergrounding_el":          THRESHOLD_UNDERGROUNDING_EL,
                "undergrounding_struct_sqmi": THRESHOLD_UNDERGROUNDING_STRUCT,
                "hardening_el":               THRESHOLD_HARDENING_EL,
                "vegetation_el":              THRESHOLD_VEGETATION_EL,
            },
            "threshold_basis": "Warner, Callaway & Fowlie (2025); CPUC WMP filings (2023)",
        },
        "portfolio": {
            "n_clusters":                       int(severity["cluster_id"].nunique()),
            "total_expected_loss_usd":          float(total_el),
            "mean_expected_loss_usd":           float(severity["expected_loss_usd"].mean()),
            "median_expected_loss_usd":         float(severity["expected_loss_usd"].median()),
            "structures_density_available_pct": float(severity["structures_per_sqmi"].notna().mean()*100),
            "mean_expected_burned_area_ac":     float(severity["expected_burned_area_ac"].mean()),
            "mean_severity_usd_given_ignition": float(severity["severity_usd_given_ignition"].mean(skipna=True)),
        },
        "concentration": concentration,
        "recommendation_counts":  severity["recommended_mitigation"].value_counts(dropna=False).to_dict(),
        "recommendation_el_share": {
            str(row.Index): float(row.pct / 100.0)
            for row in mit.sort_values("sum", ascending=False).itertuples()
        },
        "size_model": size_metrics,
    }

    out_json = OUTDIR / "severity_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print("\n" + "="*65)
    print("SEVERITY MODEL COMPLETE")
    print("="*65)
    print(f"  Clusters:           {severity['cluster_id'].nunique()}")
    print(f"  Total portfolio EL: ${total_el:,.0f}")
    print(f"  Saved: {out_csv}")
    print(f"  Saved: {out_json}")
    print("="*65)


if __name__ == "__main__":
    main()
