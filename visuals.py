"""
VISUALIZATIONS — Wildfire Grid Risk Model (Baseline v1)
Outputs:
  - fig_shap_importance.png
  - fig_roc_curve.png
  - fig_temporal_auc.png

Run from the same directory as features_ignitions.csv and model outputs.
Requires: matplotlib, seaborn, xgboost, shap, sklearn, pandas, numpy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
import shap
import warnings
warnings.filterwarnings("ignore")

# ── STYLE ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font="DejaVu Sans")
PALETTE = {
    "asset"   : "#2563EB",
    "spatial" : "#059669",
    "temporal": "#D97706",
    "cause"   : "#7C3AED",
    "blue"    : "#2563EB",
    "red"     : "#DC2626",
    "gray"    : "#94A3B8",
}
TITLE_SIZE  = 14
LABEL_SIZE  = 11
TICK_SIZE   = 10
ANNOT_SIZE  = 9

# ── LOAD DATA & RETRAIN ────────────────────────────────────────────────────
print("Loading data and retraining model...")
df = pd.read_csv("features_ignitions.csv")
df = df[~df["year"].isin([2019, 2020])]

NON_FEATURES = [
    "fire_spread", "fire_size_ordinal", "year", "latitude", "longitude",
    "outage_occurred", "utility_PG&E", "utility_SCE", "utility_SDG&E",
    "previously_burned",
    "ignition_cause_clean_contact_object", "ignition_cause_clean_equipment_failure",
    "ignition_cause_clean_other", "ignition_cause_clean_third_party",
    "ignition_cause_clean_unknown", "ignition_cause_clean_wire_contact",
]
FEATURE_COLS = [c for c in df.columns if c not in NON_FEATURES]
X = df[FEATURE_COLS]
y = df["fire_spread"]

best_params = dict(max_depth=3, n_estimators=200, learning_rate=0.1,
                   subsample=1.0, min_child_weight=1)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# OOF predictions for ROC
model_cv = XGBClassifier(**best_params, eval_metric="logloss",
                         random_state=42, n_jobs=-1)
oof_probs = cross_val_predict(model_cv, X, y, cv=cv,
                              method="predict_proba", n_jobs=-1)[:, 1]

# Per-year AUC
df_eval = df[["year"]].copy()
df_eval["y_true"] = y.values
df_eval["y_prob"] = oof_probs
year_aucs = {}
for yr, grp in df_eval.groupby("year"):
    if grp["y_true"].nunique() == 2:
        year_aucs[yr] = roc_auc_score(grp["y_true"], grp["y_prob"])

# Final model for SHAP
final_model = XGBClassifier(**best_params, eval_metric="logloss",
                            random_state=42, n_jobs=-1)
final_model.fit(X, y, verbose=False)

# SHAP
print("Computing SHAP values...")
explainer   = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)
mean_shap   = pd.DataFrame({
    "feature"       : FEATURE_COLS,
    "mean_abs_shap" : np.abs(shap_values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False).head(15)

overall_auc = roc_auc_score(y, oof_probs)
overall_pr  = average_precision_score(y, oof_probs)
fpr, tpr, _ = roc_curve(y, oof_probs)

print(f"AUC-ROC: {overall_auc:.4f} | PR-AUC: {overall_pr:.4f}")

# ── FEATURE LABELS & CATEGORIES ────────────────────────────────────────────
LABEL_MAP = {
    "voltage_kv"                    : "Voltage (kV)",
    "land_use_clean_rural"          : "Land Use: Rural",
    "month"                         : "Month",
    "land_use_clean_urban"          : "Land Use: Urban",
    "hftd_tier_Tier 2"              : "HFTD Tier 2",
    "land_use_clean_other"          : "Land Use: Other",
    "equipment_category_conductor"  : "Equipment: Conductor",
    "equipment_category_other"      : "Equipment: Other",
    "facility_type_clean_underground": "Facility: Underground",
    "hftd_tier_non_hftd"           : "HFTD: Non-Zone",
    "had_contributing_factor"       : "Contributing Factor",
    "season_spring"                 : "Season: Spring",
    "voltage_tier_dist_21kv"        : "Voltage Tier: 21kV",
    "is_weekend"                    : "Is Weekend",
    "season_summer"                 : "Season: Summer",
}
CATEGORY_MAP = {
    "voltage_kv"                    : "asset",
    "land_use_clean_rural"          : "spatial",
    "month"                         : "temporal",
    "land_use_clean_urban"          : "spatial",
    "hftd_tier_Tier 2"              : "spatial",
    "land_use_clean_other"          : "spatial",
    "equipment_category_conductor"  : "asset",
    "equipment_category_other"      : "asset",
    "facility_type_clean_underground": "asset",
    "hftd_tier_non_hftd"           : "spatial",
    "had_contributing_factor"       : "cause",
    "season_spring"                 : "temporal",
    "voltage_tier_dist_21kv"        : "asset",
    "is_weekend"                    : "temporal",
    "season_summer"                 : "temporal",
}

mean_shap["label"]    = mean_shap["feature"].map(LABEL_MAP).fillna(mean_shap["feature"])
mean_shap["category"] = mean_shap["feature"].map(CATEGORY_MAP).fillna("other")
mean_shap["color"]    = mean_shap["category"].map(PALETTE)
mean_shap_plot = mean_shap.sort_values("mean_abs_shap", ascending=True)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: SHAP FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(10, 7))
fig1.patch.set_facecolor("white")

bars = ax1.barh(
    mean_shap_plot["label"],
    mean_shap_plot["mean_abs_shap"],
    color=mean_shap_plot["color"],
    edgecolor="white",
    linewidth=0.5,
    height=0.65,
)

# Value labels on bars
for bar, val in zip(bars, mean_shap_plot["mean_abs_shap"]):
    ax1.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", ha="left",
             fontsize=ANNOT_SIZE, color="#374151")

# Vertical reference line at top feature
ax1.axvline(mean_shap_plot["mean_abs_shap"].max(), color="#E5E7EB",
            linewidth=1, linestyle="--", zorder=0)

ax1.set_xlabel("Mean Absolute SHAP Value", fontsize=LABEL_SIZE, color="#374151", labelpad=10)
ax1.set_title("SHAP Feature Importance\nWildfire Ignition Spread Model (Baseline v1 · No Weather Features)",
              fontsize=TITLE_SIZE, fontweight="bold", color="#111827", pad=16, loc="left")

ax1.tick_params(axis="y", labelsize=TICK_SIZE, colors="#374151")
ax1.tick_params(axis="x", labelsize=TICK_SIZE, colors="#6B7280")
ax1.set_xlim(0, mean_shap_plot["mean_abs_shap"].max() * 1.18)
ax1.spines[["top", "right", "left"]].set_visible(False)
ax1.spines["bottom"].set_color("#E5E7EB")
ax1.xaxis.grid(True, color="#F3F4F6", linewidth=1)
ax1.set_axisbelow(True)

# Legend
legend_patches = [
    mpatches.Patch(color=PALETTE["asset"],    label="Asset"),
    mpatches.Patch(color=PALETTE["spatial"],  label="Spatial"),
    mpatches.Patch(color=PALETTE["temporal"], label="Temporal"),
    mpatches.Patch(color=PALETTE["cause"],    label="Cause Flag"),
]
ax1.legend(handles=legend_patches, loc="lower right", fontsize=ANNOT_SIZE,
           frameon=True, framealpha=0.9, edgecolor="#E5E7EB")

plt.tight_layout(pad=2)
plt.savefig("fig_shap_importance.png", dpi=180, bbox_inches="tight",
            facecolor="white")
plt.close()
print("Saved: fig_shap_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: ROC CURVE
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(7, 6))
fig2.patch.set_facecolor("white")

ax2.plot(fpr, tpr, color=PALETTE["blue"], lw=2.5,
         label=f"XGBoost Baseline v1  (AUC = {overall_auc:.4f})")
ax2.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["blue"])
ax2.plot([0, 1], [0, 1], color=PALETTE["gray"], lw=1.5,
         linestyle="--", label="Random classifier  (AUC = 0.500)")

# AUC annotation box
ax2.annotate(
    f"AUC-ROC\n{overall_auc:.4f}",
    xy=(0.6, 0.55), fontsize=12, fontweight="bold",
    color=PALETTE["blue"], ha="center",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor=PALETTE["blue"], alpha=0.9),
)

ax2.set_xlabel("False Positive Rate", fontsize=LABEL_SIZE, color="#374151", labelpad=8)
ax2.set_ylabel("True Positive Rate", fontsize=LABEL_SIZE, color="#374151", labelpad=8)
ax2.set_title("ROC Curve\nWildfire Ignition Spread Model (5-Fold Stratified CV · Out-of-Fold)",
              fontsize=TITLE_SIZE, fontweight="bold", color="#111827", pad=14, loc="left")

ax2.set_xlim(-0.01, 1.01)
ax2.set_ylim(-0.01, 1.01)
ax2.tick_params(labelsize=TICK_SIZE, colors="#6B7280")
ax2.spines[["top", "right"]].set_visible(False)
ax2.spines[["left", "bottom"]].set_color("#E5E7EB")
ax2.xaxis.grid(True, color="#F3F4F6", linewidth=1)
ax2.yaxis.grid(True, color="#F3F4F6", linewidth=1)
ax2.set_axisbelow(True)
ax2.legend(fontsize=ANNOT_SIZE, frameon=True, framealpha=0.9,
           edgecolor="#E5E7EB", loc="lower right")

plt.tight_layout(pad=2)
plt.savefig("fig_roc_curve.png", dpi=180, bbox_inches="tight",
            facecolor="white")
plt.close()
print("Saved: fig_roc_curve.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: PER-YEAR AUC STABILITY
# ══════════════════════════════════════════════════════════════════════════════
years  = list(year_aucs.keys())
aucs   = [year_aucs[y] for y in years]
colors = [PALETTE["red"] if a < 0.60 else PALETTE["blue"] for a in aucs]

fig3, ax3 = plt.subplots(figsize=(9, 5))
fig3.patch.set_facecolor("white")

bars3 = ax3.bar(
    [str(y) for y in years], aucs,
    color=colors, edgecolor="white", linewidth=0.5, width=0.6,
)

# Overall AUC reference line
ax3.axhline(overall_auc, color="#374151", linewidth=1.5, linestyle="--",
            label=f"Overall AUC = {overall_auc:.4f}", zorder=3)

# Value labels
for bar, val, yr in zip(bars3, aucs, years):
    ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.004,
             f"{val:.3f}", ha="center", va="bottom",
             fontsize=ANNOT_SIZE, color="#374151", fontweight="bold")

# Annotate flagged years
for bar, val, yr in zip(bars3, aucs, years):
    if val < 0.60:
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_y() + 0.01,
                 "⚠ weather\nfeatures\nabsent",
                 ha="center", va="bottom", fontsize=7.5,
                 color="white", fontweight="bold", linespacing=1.3)

ax3.set_xlabel("Year", fontsize=LABEL_SIZE, color="#374151", labelpad=8)
ax3.set_ylabel("AUC-ROC", fontsize=LABEL_SIZE, color="#374151", labelpad=8)
ax3.set_title("Per-Year AUC-ROC Stability Check\n5-Fold CV Out-of-Fold · 2019-2020 Excluded (Reporting Anomaly)",
              fontsize=TITLE_SIZE, fontweight="bold", color="#111827", pad=14, loc="left")

ax3.set_ylim(0.45, 0.76)
ax3.tick_params(labelsize=TICK_SIZE, colors="#6B7280")
ax3.spines[["top", "right"]].set_visible(False)
ax3.spines[["left", "bottom"]].set_color("#E5E7EB")
ax3.yaxis.grid(True, color="#F3F4F6", linewidth=1)
ax3.set_axisbelow(True)

legend_patches2 = [
    mpatches.Patch(color=PALETTE["blue"], label="Normal performance"),
    mpatches.Patch(color=PALETTE["red"],  label="Below average (< 0.60)"),
    plt.Line2D([0], [0], color="#374151", linewidth=1.5,
               linestyle="--", label=f"Overall AUC = {overall_auc:.4f}"),
]
ax3.legend(handles=legend_patches2, fontsize=ANNOT_SIZE, frameon=True,
           framealpha=0.9, edgecolor="#E5E7EB", loc="upper left")

plt.tight_layout(pad=2)
plt.savefig("fig_temporal_auc.png", dpi=180, bbox_inches="tight",
            facecolor="white")
plt.close()
print("Saved: fig_temporal_auc.png")

print("\nAll figures saved successfully.")