# Mathematical Modeling Methodology

This document lists the mathematical and statistical modeling techniques used in the ignition probability model.

## Modeling Target
- Binary event model: `P(confirmed_ignition | location-day, weather, fuels, topography)`

## Mathematical / Modeling Techniques
- Spatial segmentation with K-means clustering (`n_clusters = 1000`)
- Full denominator panel construction (`cluster x day`, 2014-01-01 to 2024-12-31)
- Event overlay for positive labels on cluster-day pairs
- Binary supervised learning formulation (ignition vs non-ignition)
- Temporal holdout design (train <= 2022, validation = 2023, test = 2024)
- Severe class-imbalance handling via negative undersampling
- Undersampling ratio control (`neg:pos = 200:1`)
- Sampling-prior correction with Pozzolo calibration (`beta = n_pos / n_neg_sampled`)
- Calibration transform: `p_cal = p / (p + (1 - p) / beta)`
- Random Forest classifier (`n_estimators=300`, `max_depth=8`, `min_samples_leaf=20`, `class_weight='balanced'`)
- Numeric preprocessing: median imputation + standardization
- Categorical preprocessing: constant imputation + one-hot encoding
- Mixed-feature model design (weather + time + coordinates + fuels + topography)
- Ablation-study design for incremental feature contribution:
  - `weather_only`
  - `weather_plus_coords`
  - `weather_coords_fuels_topo`
- Performance evaluation with ROC-AUC
- Performance evaluation with PR-AUC (for rare-event sensitivity)
- Performance evaluation with Brier score (probability accuracy)
- Raw-vs-calibrated probability consistency check (`corr(raw, cal)`)
- Permutation feature-importance analysis (validation split)
- SHAP-based feature-importance diagnostics (optional)

## Modeling Outputs Produced
- `/Users/ankurbanerjee/Desktop/MTF/wildfire_grid_protection/outputs/panel_metrics.json`
- `/Users/ankurbanerjee/Desktop/MTF/wildfire_grid_protection/outputs/panel_predictions_test.csv`
- `/Users/ankurbanerjee/Desktop/MTF/wildfire_grid_protection/outputs/panel_full.csv`
- `/Users/ankurbanerjee/Desktop/MTF/wildfire_grid_protection/outputs/panel_model.joblib`
- `/Users/ankurbanerjee/Desktop/MTF/wildfire_grid_protection/outputs/ignition_model_panel.png`
- `/Users/ankurbanerjee/Desktop/MTF/wildfire_grid_protection/outputs/shap_summary_bar_val_2023.png`
- `/Users/ankurbanerjee/Desktop/MTF/wildfire_grid_protection/outputs/shap_summary_swarm_val_2023.png`

## Canonical Severity / Loss Pipeline
- Canonical script for severity + recommendations + counterfactuals:
  - `/Users/ankurbanerjee/Desktop/MTF/wildfire_grid_protection/severity_hurdle_model.py`
- Legacy scripts retained only for reproducibility (not cited for final results):
  - `/Users/ankurbanerjee/Desktop/MTF/wildfire_grid_protection/legacy_severity_v1.py`
  - `/Users/ankurbanerjee/Desktop/MTF/wildfire_grid_protection/legacy_severity_v2.py`
