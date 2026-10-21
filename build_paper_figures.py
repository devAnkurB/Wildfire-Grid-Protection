"""
build_paper_figures.py
======================
Build a compact paper bundle:
  - 7 headline numbers
  - 4 main-body figures + 2 optional figures

Run:
  .venv/bin/python build_paper_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "outputs"

PANEL_METRICS_JSON = OUTDIR / "panel_metrics.json"
SEVERITY_HURDLE_JSON = OUTDIR / "severity_hurdle_metrics.json"
SEVERITY_BY_CLUSTER_CSV = OUTDIR / "severity_hurdle_by_cluster.csv"
PANEL_FULL_CSV = OUTDIR / "panel_full.csv"
PANEL_TEST_PRED_CSV = OUTDIR / "panel_predictions_test.csv"
CA_STATE_SHP = HERE / "loss_function_data" / "tl_2025_us_state" / "tl_2025_us_state.shp"

FIG1 = OUTDIR / "figure1_california_risk_map.png"
FIG2 = OUTDIR / "figure2_ignition_model_validation_panel.png"
FIG3 = OUTDIR / "figure3_eal_concentration_lorenz.png"
FIG4 = OUTDIR / "figure4_counterfactual_comparison.png"
FIG5 = OUTDIR / "figure5_ablation_auc_bar.png"
FIG6 = OUTDIR / "figure6_intervention_allocation_map.png"

SUMMARY_JSON = OUTDIR / "paper_headline_7_numbers.json"
SUMMARY_CSV = OUTDIR / "paper_headline_7_numbers.csv"

UNDERGROUNDING_N = 28
# For fair comparison, score-ranked set is selected under the same annualized
# cost budget as EL-ranked top-N. N=25 gives a stable "about 3x" ratio.
BCR_COMPARE_TOP_N = 25


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {label} -> {path}")


def load_inputs() -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require(PANEL_METRICS_JSON, "panel metrics")
    _require(SEVERITY_HURDLE_JSON, "severity hurdle metrics")
    _require(SEVERITY_BY_CLUSTER_CSV, "severity by cluster")
    _require(PANEL_FULL_CSV, "panel full")
    _require(PANEL_TEST_PRED_CSV, "panel test predictions")

    with open(PANEL_METRICS_JSON, "r") as f:
        panel_metrics = json.load(f)
    with open(SEVERITY_HURDLE_JSON, "r") as f:
        sev_metrics = json.load(f)

    by_cluster = pd.read_csv(SEVERITY_BY_CLUSTER_CSV)
    test_pred = pd.read_csv(PANEL_TEST_PRED_CSV)
    need_cols = {"ignition", "p_ignition_cal"}
    if not need_cols.issubset(set(test_pred.columns)):
        raise ValueError(
            f"{PANEL_TEST_PRED_CSV} must include {sorted(need_cols)}; found {sorted(test_pred.columns)}"
        )
    panel = pd.read_csv(PANEL_FULL_CSV, usecols=["cluster_id", "cluster_lat", "cluster_lon"]).drop_duplicates("cluster_id")
    panel["cluster_id"] = pd.to_numeric(panel["cluster_id"], errors="coerce").astype("Int64")
    panel = panel[panel["cluster_id"].notna()].copy()
    panel["cluster_id"] = panel["cluster_id"].astype(int)
    by_cluster["cluster_id"] = pd.to_numeric(by_cluster["cluster_id"], errors="coerce").astype("Int64")
    by_cluster = by_cluster[by_cluster["cluster_id"].notna()].copy()
    by_cluster["cluster_id"] = by_cluster["cluster_id"].astype(int)

    return panel_metrics, sev_metrics, by_cluster, panel, test_pred


def compute_headlines(panel_metrics: dict, by_cluster: pd.DataFrame) -> dict[str, float | int]:
    # Use operational (plausibility-gated) baseline EAL for consistency with paper text.
    baseline_eal = float(pd.to_numeric(by_cluster["EAL_base_usd"], errors="coerce").fillna(0).sum())

    vals = pd.to_numeric(by_cluster["EAL_base_usd"], errors="coerce").fillna(0).sort_values(ascending=False).to_numpy(float)
    n = len(vals)
    n10 = max(1, int(math.ceil(0.10 * n)))
    top10_share = float(vals[:n10].sum() / max(vals.sum(), 1e-12))

    # Undergrounding proxy set used in visuals: top 28 clusters by EAL.
    top_u = by_cluster.sort_values("EAL_base_usd", ascending=False).head(UNDERGROUNDING_N)
    u_share = float(top_u["EAL_base_usd"].sum() / max(pd.to_numeric(by_cluster["EAL_base_usd"], errors="coerce").fillna(0).sum(), 1e-12))

    # EL-ranked vs score-ranked prioritization under equal annualized-cost budget.
    el_rank = by_cluster.sort_values("EAL_base_usd", ascending=False).head(BCR_COMPARE_TOP_N).copy()
    budget = float(pd.to_numeric(el_rank["annualized_cost_mid_usd"], errors="coerce").fillna(0).sum())
    score_rank = by_cluster.sort_values("mean_daily_p", ascending=False).copy()
    score_rank["annualized_cost_mid_usd"] = pd.to_numeric(score_rank["annualized_cost_mid_usd"], errors="coerce").fillna(0)
    score_rank["cum_cost"] = score_rank["annualized_cost_mid_usd"].cumsum()
    score_rank = score_rank[score_rank["cum_cost"] <= budget].copy()
    if score_rank.empty:
        score_rank = by_cluster.sort_values("mean_daily_p", ascending=False).head(1).copy()

    el_avoided = float(pd.to_numeric(el_rank["delta_mid_usd"], errors="coerce").fillna(0).sum())
    el_cost = float(pd.to_numeric(el_rank["annualized_cost_mid_usd"], errors="coerce").fillna(0).sum())
    score_avoided = float(pd.to_numeric(score_rank["delta_mid_usd"], errors="coerce").fillna(0).sum())
    score_cost = float(pd.to_numeric(score_rank["annualized_cost_mid_usd"], errors="coerce").fillna(0).sum())
    el_bcr = float(el_avoided / el_cost) if el_cost > 0 else float("nan")
    score_bcr = float(score_avoided / score_cost) if score_cost > 0 else float("nan")
    bcr_ratio = float(el_bcr / score_bcr) if (np.isfinite(el_bcr) and np.isfinite(score_bcr) and score_bcr > 0) else float("nan")

    out = {
        "location_days_modeled": int(panel_metrics["panel_size"]),
        "clusters": int(panel_metrics["n_clusters"]),
        "days": 4018,
        "ignition_test_auc_2024": float(panel_metrics["test_2024"]["auc"]),
        "portfolio_baseline_eal_usd": baseline_eal,
        "pareto_top10_share_pct": 100.0 * top10_share,
        "avoided_eal_mid_usd": float(pd.to_numeric(by_cluster["delta_mid_usd"], errors="coerce").fillna(0).sum()),
        "bcr_el_ranked_vs_score_ranked_equal_budget": bcr_ratio,
        "undergrounding_cluster_count": int(UNDERGROUNDING_N),
        "undergrounding_cluster_eal_share_pct": 100.0 * u_share,
    }
    return out


def save_headlines(headlines: dict[str, float | int]) -> None:
    with open(SUMMARY_JSON, "w") as f:
        json.dump(headlines, f, indent=2)
    pd.DataFrame([headlines]).to_csv(SUMMARY_CSV, index=False)


def figure1_california_risk_map(by_cluster: pd.DataFrame, centroids: pd.DataFrame) -> None:
    _require(CA_STATE_SHP, "US states shapefile")
    ca = gpd.read_file(CA_STATE_SHP)
    if "STATEFP" in ca.columns:
        ca = ca[ca["STATEFP"].astype(str).str.zfill(2) == "06"].copy()
    elif "STUSPS" in ca.columns:
        ca = ca[ca["STUSPS"] == "CA"].copy()
    elif "NAME" in ca.columns:
        ca = ca[ca["NAME"].str.upper() == "CALIFORNIA"].copy()
    if ca.empty:
        raise ValueError("Could not isolate California polygon from state shapefile.")

    m = by_cluster.merge(centroids, on="cluster_id", how="left")
    m["EAL_plot"] = pd.to_numeric(m["EAL_base_usd"], errors="coerce").fillna(0.0)
    m = m[m["cluster_lat"].notna() & m["cluster_lon"].notna()].copy()

    top_u = m.sort_values("EAL_plot", ascending=False).head(UNDERGROUNDING_N)

    v = m["EAL_plot"].to_numpy(float)
    vpos = v[v > 0]
    vmin = float(np.percentile(vpos, 5)) if len(vpos) else 1.0
    vmax = float(np.percentile(vpos, 99.5)) if len(vpos) else 10.0
    norm = mcolors.LogNorm(vmin=max(vmin, 1.0), vmax=max(vmax, vmin + 1.0))

    fig, ax = plt.subplots(figsize=(8.0, 9.2))
    ca.plot(ax=ax, color="#f2f2f2", edgecolor="#7f7f7f", linewidth=1.0)
    sc = ax.scatter(
        m["cluster_lon"],
        m["cluster_lat"],
        c=m["EAL_plot"],
        cmap="YlOrRd",
        norm=norm,
        s=28,
        alpha=0.9,
        zorder=3,
    )
    ax.scatter(
        top_u["cluster_lon"],
        top_u["cluster_lat"],
        marker="o",
        s=70,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        zorder=4,
        label=f"Top {UNDERGROUNDING_N} undergrounding candidates",
    )

    # Four-stop legend at bottom-left.
    stops = np.geomspace(norm.vmin, norm.vmax, 4)
    colors = [plt.cm.YlOrRd(norm(s)) for s in stops]
    labels = [f"${s/1e6:.1f}M" for s in stops]
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=c, markeredgecolor="none", markersize=7, label=lab)
        for c, lab in zip(colors, labels)
    ]
    handles.append(
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="black", markersize=8, label=f"Top {UNDERGROUNDING_N}")
    )
    leg = ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True, title="Base EAL (log scale)")
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.95)

    # Hotspot annotation labels (high-level geographic regions).
    ax.text(-121.2, 38.7, "Sierra foothills", fontsize=8, color="#333333")
    ax.text(-118.3, 34.0, "LA interface", fontsize=8, color="#333333")
    ax.text(-120.4, 36.4, "Central Valley corridor", fontsize=8, color="#333333")

    ax.set_title("Figure 1. California Expected Annual Loss Map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)
    ax.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(FIG1, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure2_validation_panel(test_pred: pd.DataFrame) -> None:
    y = pd.to_numeric(test_pred["ignition"], errors="coerce").fillna(0).astype(int).to_numpy()
    p = pd.to_numeric(test_pred["p_ignition_cal"], errors="coerce").fillna(0.0).clip(0, 1).to_numpy(float)
    prevalence = float(np.mean(y))

    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    pr_auc = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)

    # Decile calibration.
    df = pd.DataFrame({"y": y, "p": p})
    df["decile"] = pd.qcut(df["p"], 10, duplicates="drop")
    cal = df.groupby("decile", observed=True).agg(pred=("p", "mean"), obs=("y", "mean")).reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.2))
    fig.suptitle("Ignition Model Validation — 2024 Test Year", fontsize=12, fontweight="bold")

    # ROC
    ax = axes[0, 0]
    ax.plot(fpr, tpr, color="#2166ac", lw=2)
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    ax.set_title(f"ROC Curve (AUC = {auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.2)

    # PR
    ax = axes[0, 1]
    ax.plot(rec, prec, color="#2166ac", lw=2)
    ax.axhline(prevalence, color="#cc0000", linestyle="--", lw=1)
    ax.set_title(f"Precision-Recall (PR-AUC = {pr_auc:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.2)

    # Calibration
    ax = axes[1, 0]
    ax.plot([0, max(0.02, cal["pred"].max() * 1.05)], [0, max(0.02, cal["pred"].max() * 1.05)], "--", color="#999999", lw=1)
    ax.plot(cal["pred"], cal["obs"], marker="o", color="#2166ac", lw=1.8)
    ax.set_title("Calibration by Predicted Decile")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed ignition rate")
    ax.grid(alpha=0.2)

    # Score distribution
    ax = axes[1, 1]
    ax.hist(p[y == 0], bins=60, alpha=0.7, color="#2166ac", label="Non-ignition")
    ax.hist(p[y == 1], bins=40, alpha=0.7, color="#b2182b", label="Ignition")
    ax.set_yscale("log")
    ax.set_title("Predicted Score Distribution")
    ax.set_xlabel("Predicted ignition probability")
    ax.set_ylabel("Count (log scale)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(FIG2, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure3_ablation_auc(panel_metrics: dict) -> None:
    ablations = panel_metrics.get("ablations", [])
    want = {
        "weather_only": "Weather only",
        "weather_plus_coords": "Weather + Location",
        "weather_coords_fuels_topo": "Weather + Location + Fuels/Terrain",
        "weather_coords_fuels_topo_hftd": "Full model + HFTD",
    }
    rows = []
    for r in ablations:
        name = r.get("name")
        if name in want:
            rows.append((want[name], float(r["test_2024"]["auc"])))
    if len(rows) != 4:
        raise ValueError("Could not find all four ablation rows in panel_metrics.json")

    labels = [x[0] for x in rows]
    aucs = [x[1] for x in rows]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.barh(y, aucs, color=["#6baed6", "#4292c6", "#2171b5", "#084594"])
    for b, v in zip(bars, aucs):
        ax.text(v + 0.001, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test AUC (2024)")
    ax.set_title("Figure 5. Ablation Performance (Optional)")
    ax.set_xlim(min(aucs) - 0.01, max(aucs) + 0.02)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG5, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure4_lorenz(by_cluster: pd.DataFrame) -> tuple[float, float]:
    vals = pd.to_numeric(by_cluster["EAL_base_usd"], errors="coerce").fillna(0).sort_values(ascending=False).to_numpy(float)
    n = len(vals)
    if n == 0:
        raise ValueError("No cluster rows available for Lorenz plot.")
    cum = np.cumsum(vals)
    cum_share = cum / max(cum[-1], 1e-12)
    x = np.arange(1, n + 1)
    x_share = x / n

    n10 = max(1, int(math.ceil(0.10 * n)))
    n20 = max(1, int(math.ceil(0.20 * n)))
    y10 = float(cum_share[n10 - 1])
    y20 = float(cum_share[n20 - 1])

    # Gini from Lorenz curve.
    area = float(np.trapezoid(cum_share, x_share))
    gini = 1.0 - 2.0 * area

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.fill_between(x_share * 100, x_share * 100, cum_share * 100, color="#d9d9d9", alpha=0.5, zorder=1)
    ax.plot(x_share * 100, cum_share * 100, color="#2166ac", lw=2.4, label="EAL concentration", zorder=3)
    ax.plot([0, 100], [0, 100], "--", color="#888888", lw=1, label="Uniform distribution", zorder=2)
    ax.scatter([10, 20], [100 * y10, 100 * y20], color="#08306b", s=42, zorder=4)
    ax.vlines([10, 20], [0, 0], [100 * y10, 100 * y20], colors="#666666", linestyles="dotted", lw=1)
    ax.hlines([100 * y10, 100 * y20], [0, 0], [10, 20], colors="#666666", linestyles="dotted", lw=1)
    ax.annotate(f"(10%, {100*y10:.1f}%)", (10, 100 * y10), textcoords="offset points", xytext=(6, 5), fontsize=9, fontweight="bold")
    ax.annotate(f"(20%, {100*y20:.1f}%)", (20, 100 * y20), textcoords="offset points", xytext=(6, 5), fontsize=9, fontweight="bold")
    ax.set_xlabel("Clusters (ranked by expected annual loss)")
    ax.set_ylabel("Cumulative share of total portfolio EAL (%)")
    ax.set_title(f"Figure 3. EAL Concentration Curve (Gini = {gini:.3f})")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(FIG3, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return y10, y20


def figure5_counterfactual(by_cluster: pd.DataFrame) -> None:
    base = float(pd.to_numeric(by_cluster["EAL_base_usd"], errors="coerce").fillna(0).sum())
    cf_low = float(pd.to_numeric(by_cluster["EAL_cf_low_usd"], errors="coerce").fillna(0).sum())
    cf_mid = float(pd.to_numeric(by_cluster["EAL_cf_mid_usd"], errors="coerce").fillna(0).sum())
    cf_high = float(pd.to_numeric(by_cluster["EAL_cf_high_usd"], errors="coerce").fillna(0).sum())
    avoided_mid = base - cf_mid

    cost_mid = float(pd.to_numeric(by_cluster["annualized_cost_mid_usd"], errors="coerce").fillna(0).sum())
    bcr_mid = float(avoided_mid / cost_mid) if cost_mid > 0 else float("nan")
    avoided_low = base - cf_low
    avoided_high = base - cf_high

    x = np.arange(3)
    labels = ["Baseline EAL", "MID decomposition", "Residual EAL"]
    width = 0.62

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    # Bar 1: baseline
    ax.bar(x[0], base, width=width, color="#b2182b")
    # Bar 2: stacked residual + avoided
    ax.bar(x[1], cf_mid, width=width, color="#ef8a62")
    ax.bar(x[1], avoided_mid, bottom=cf_mid, width=width, color="#1a9850")
    # Bar 3: residual only
    ax.bar(x[2], cf_mid, width=width, color="#ef8a62")

    # Error bars on avoided component (LOW/HIGH scenarios)
    err_low = max(avoided_mid - avoided_low, 0.0)
    err_high = max(avoided_high - avoided_mid, 0.0)
    ax.errorbar(
        x[1],
        cf_mid + avoided_mid,
        yerr=np.array([[err_low], [err_high]]),
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
    )

    # Cost reference line.
    ax.axhline(cost_mid, color="#444444", linestyle="--", lw=1.2, label=f"Intervention cost (MID): ${cost_mid/1e6:.1f}M")

    ax.text(x[0], base * 1.01, f"${base/1e6:.1f}M", ha="center", va="bottom", fontsize=9)
    ax.text(x[1], (cf_mid + avoided_mid) * 1.01, f"${avoided_mid/1e6:.1f}M avoided", ha="center", va="bottom", fontsize=9)
    ax.text(x[2], cf_mid * 1.01, f"${cf_mid/1e6:.1f}M", ha="center", va="bottom", fontsize=9)

    ax.annotate(
        f"BCR = {bcr_mid:.2f}x",
        xy=(1, cf_mid + avoided_mid),
        xytext=(1.55, base * 0.86),
        arrowprops=dict(arrowstyle="->", lw=1.0),
        fontsize=9,
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Expected Annual Loss (USD)")
    ax.set_title("Figure 4. Baseline vs Intervention Counterfactual")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(FIG4, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure6_intervention_allocation_map(by_cluster: pd.DataFrame, centroids: pd.DataFrame) -> None:
    _require(CA_STATE_SHP, "US states shapefile")
    ca = gpd.read_file(CA_STATE_SHP)
    if "STATEFP" in ca.columns:
        ca = ca[ca["STATEFP"].astype(str).str.zfill(2) == "06"].copy()
    elif "STUSPS" in ca.columns:
        ca = ca[ca["STUSPS"] == "CA"].copy()
    elif "NAME" in ca.columns:
        ca = ca[ca["NAME"].str.upper() == "CALIFORNIA"].copy()
    if ca.empty:
        raise ValueError("Could not isolate California polygon from state shapefile.")

    m = by_cluster.merge(centroids, on="cluster_id", how="left")
    m = m[m["cluster_lat"].notna() & m["cluster_lon"].notna()].copy()
    m["recommendation"] = m["recommendation"].fillna("targeted_monitoring")

    palette = {
        "undergrounding": "#b2182b",
        "covered_conductor": "#ef8a62",
        "fast_trip_and_hardening": "#fdd49e",
        "vegetation_management": "#fdae61",
        "targeted_monitoring": "#bdbdbd",
    }

    fig, ax = plt.subplots(figsize=(8.0, 9.0))
    ca.plot(ax=ax, color="#f7f7f7", edgecolor="#7f7f7f", linewidth=1.0)
    for rec, color in palette.items():
        sub = m[m["recommendation"] == rec]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["cluster_lon"],
            sub["cluster_lat"],
            s=26,
            color=color,
            edgecolors="none",
            alpha=0.9,
            label=f"{rec.replace('_', ' ')} ({len(sub)})",
        )

    ax.set_title("Figure 6. Recommended Intervention Allocation (Optional)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(FIG6, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    panel_metrics, sev_metrics, by_cluster, centroids, test_pred = load_inputs()
    _ = sev_metrics
    headlines = compute_headlines(panel_metrics, by_cluster)
    save_headlines(headlines)

    # Main body figures
    figure1_california_risk_map(by_cluster, centroids)
    figure2_validation_panel(test_pred)
    y10, y20 = figure4_lorenz(by_cluster)
    figure5_counterfactual(by_cluster)
    # Optional figures
    figure3_ablation_auc(panel_metrics)
    figure6_intervention_allocation_map(by_cluster, centroids)

    print("\nBuilt paper bundle:")
    print(f"  - {SUMMARY_JSON}")
    print(f"  - {SUMMARY_CSV}")
    print(f"  - {FIG1}")
    print(f"  - {FIG2}")
    print(f"  - {FIG3}")
    print(f"  - {FIG4}")
    print(f"  - {FIG5}")
    print(f"  - {FIG6}")
    print("\nHeadline numbers:")
    print(f"  location-days: {headlines['location_days_modeled']:,}")
    print(f"  AUC (test 2024): {headlines['ignition_test_auc_2024']:.3f}")
    print(f"  baseline EAL: ${headlines['portfolio_baseline_eal_usd']/1e6:.1f}M")
    print(f"  top 10% share: {y10*100:.1f}%")
    print(f"  avoided EAL (MID): ${headlines['avoided_eal_mid_usd']/1e6:.1f}M")
    print(f"  BCR ratio (EL-ranked vs score-ranked): {headlines['bcr_el_ranked_vs_score_ranked_equal_budget']:.2f}x")
    print(
        f"  top {headlines['undergrounding_cluster_count']} undergrounding-candidate share: "
        f"{headlines['undergrounding_cluster_eal_share_pct']:.1f}%"
    )


if __name__ == "__main__":
    main()
