# Enhanced Vehicle Efficiency Analysis: EV vs ICE Cohorts with Feature Engineering and Targeted Fine‑Tuning

Author: Your Name
Date: 2025‑09‑02
Keywords: vehicle efficiency, EV, ICE, feature engineering, regression, cross‑validation, fine‑tuning, model comparison, reproducibility

## Abstract

We develop a modular machine‑learning pipeline to predict vehicle efficiency (km per unit energy) for Electric Vehicles (EV) and Internal Combustion Engine (ICE) cohorts. The workflow computes the efficiency target, engineers domain‑informed features with strict leakage prevention, performs multi‑criterion feature selection, trains multiple regressors with robust cross‑validation, and optionally fine‑tunes the top two models per cohort. We evaluate models using MAE, RMSE, and R² on a hold‑out set, and produce dashboards and a final report. Recent results show small positive R² for EV and negative R² for ICE with current features, making MAE the more actionable metric. The pipeline is fully reproducible and logs all steps.

## 1. Introduction

Predicting vehicle efficiency enables benchmarking of powertrains and supports operational and policy decisions. EV and ICE vehicles differ materially in propulsion and energy systems, motivating cohort‑specific modeling. We aim for a transparent, repeatable baseline with sound feature engineering, robust validation, and targeted fine‑tuning that balances accuracy and interpretability.

## 2. Data

- Source: `data/vehicle_comparison_dataset_030417.csv`
- Target: `efficiency = mileage_km / energy_consumption` (computed at load time)
- Cohorts: Split by `vehicle_type` into EV and ICE; `vehicle_type` is dropped for modeling
- Cleaning:
  - Clip `co2_emissions_g_per_km` to lower=0
  - Remove outliers by IQR on `efficiency`
- Leakage Prevention: Because `efficiency` derives from `mileage_km` and `energy_consumption`, both are excluded from the feature set used for modeling.

## 3. Methods

### 3.1 Feature Engineering

We construct features spanning performance, environmental intensity, maintenance burden, interactions, and transforms (examples):
- Performance & ratios: `power_efficiency = torque_Nm / acceleration_0_100_kph_sec`, `storage_per_torque`, `cost_efficiency`
- Maintenance & lifespan: `maintenance_per_year`, `maintenance_per_torque`, `lifespan_torque_ratio`
- Environmental: EV — `eco_efficiency = 1/(co2+1)`, `green_performance = torque/(co2+1)`; ICE — `emission_intensity = co2/torque`, `emission_per_storage = co2/storage`
- Categories → codes: `torque_category_num`, `acceleration_category_num`
- Polynomial & interactions: `torque_squared`, `cost_squared`, `torque_x_lifespan`, `storage_x_lifespan`
- Transforms & normalization: `log_maintenance`, `log_torque`, `log_storage`, `normalized_*`

Leakage safeguards: `mileage_km` and `energy_consumption` are removed from the modeling matrix.

### 3.2 Feature Selection

We apply a three‑stage filter per cohort:
- Correlation filter vs target: EV threshold > 0.02; ICE threshold > 0.03 (absolute Pearson)
- Variance filter: remove near‑constant features (threshold 0.01)
- Multicollinearity pruning: greedy inclusion with pairwise |corr| < 0.85 among selected features
- Cap selection at 20 features.

### 3.3 Models and Preprocessing

- Linear: Linear Regression, Ridge, Lasso (with increased `max_iter` for convergence)
- Trees/Boosters: Decision Tree, Random Forest, Gradient Boosting; optional XGBoost, LightGBM, CatBoost
- Preprocessing: PowerTransformer (Yeo‑Johnson) for linear models; raw features for trees/boosting (scale‑invariant)
- Validation: 5‑fold KFold (shuffle=True, random_state=42) with MAE as primary CV score
- Hold‑out: 20% test split with MAE, RMSE, and R²
- Ranking: by `test_r2` (default) or `test_mae` (configurable)

### 3.4 Fine‑Tuning Strategy

We fine‑tune only the top two models per cohort as determined by the selected ranking metric.
- Boosting models (Gradient Boosting, XGBoost, LightGBM, CatBoost): Optuna search (trials configurable). If Optuna is unavailable, we fall back to sklearn search.
- Non‑boosting models: RandomizedSearchCV by default or GridSearchCV when requested.
- Tuning metric: MAE (default) or R².

## 4. Experiments and Results

All metrics and rankings are saved after each run; exact values depend on the current dataset snapshot and random seed.

- EV rankings: `output/enhanced_ev_model_rankings.csv`
- ICE rankings: `output/enhanced_ice_model_rankings.csv`
- Best models and parameters: `output/best_enhanced_*_model.joblib`, `output/*_model_parameters.json`

Recent Observations (illustrative):
- EV Cohort: best model by R² typically yields small positive R² (≈ 0.0–0.01); MAE around ~2,040 (units depend on target scaling). Small R² suggests limited variance explained; MAE is the more informative selection criterion.
- ICE Cohort: best R² often slightly negative (≈ −0.03 to −0.08) with MAE ~5,300–5,900. This indicates the current features are insufficient to model ICE efficiency variance; consider adding physical attributes (mass, displacement, gearing, aero drag coefficients, fuel type nuances).

Tuned vs Base (when tuned wins):
- The final report (`output/final_modeling_report.md`) includes a “Tuned vs Base” section when the best model is a tuned variant, detailing ΔMAE (positive means lower/better MAE) and ΔR² improvements.

## 5. Visualizations

Key figures are emitted to `output/`:
- Main dashboard: `output/enhanced_efficiency_dashboard.png`
- CV stability: `output/cv_stability_ev.png`, `output/cv_stability_ice.png`, `output/cv_stability_comparison.png`
- Correlation dashboards (advanced): `output/correlation_analysis_dashboard.png`, `output/detailed_correlation_matrices.png`
- Individual plots (advanced): `output/individual_plots/01..15_*.png` (distributions, model comparison, feature importance, correlations, selection summaries)

## 6. Discussion

- Metric choice: With small/negative R², MAE is a more stable and actionable comparator.
- Cohort differences: EV appears slightly more predictable with current features; ICE needs richer physical descriptors.
- Transformations: Yeo‑Johnson helps linear models; trees are scale‑invariant.
- Scope for improvement: Expand feature set with domain‑specific attributes (mass, frontal area, Cd, rolling resistance, gearing); consider alternative targets (e.g., cost‑normalized efficiency).

## 7. Threats to Validity and Limitations

- Data representativeness: If synthetic or incomplete, real‑world generalization may be limited.
- Feature coverage: Missing physical factors limit the explained variance, especially for ICE.
- Hyperparameter budgets: Limited trials/iterations trade runtime for accuracy; revisit for top performers.

## 8. Reproducibility and System Specs

- Deterministic CV and splits: KFold(shuffle=True, random_state=42)
- System specs recorded in `output/enhanced_analysis_results.json`
- Logging: console + `output/logs/pipeline.log` via Loguru

## 9. How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run full pipeline (defaults: full visualization + fine‑tuning enabled):

```bash
python scripts/run_enhanced.py --data-path data/vehicle_comparison_dataset_030417.csv --output-dir output
```

Back‑compat wrapper:

```bash
python enhanced_vehicle_efficiency_analysis.py --data-path data/vehicle_comparison_dataset_030417.csv --output-dir output
```

Toggle options (common):
- `--no-full-viz` to skip heavy plots
- `--no-fine-tune-top2` to skip fine‑tuning
- `--rank-by {r2|mae}` (ranking), `--tuning-metric {mae|r2}` (tuning)
- `--search {random|grid}`, `--random-iter N`, `--optuna-trials N`
- `--log-level {DEBUG|INFO|WARNING|ERROR}`

## 10. Conclusions

We deliver a transparent, modular baseline for EV/ICE efficiency modeling. The current feature set supports only modest predictive power by R², particularly for ICE; MAE is therefore preferred operationally. Future improvements should focus on augmenting the feature space with physical vehicle attributes and increasing tuning budgets for the top candidates.

---

### Appendix A: Repository Map

- Pipeline entry (modular): `scripts/run_enhanced.py`
- Back‑compat wrapper: `enhanced_vehicle_efficiency_analysis.py`
- Package:
  - Data: `ve/data.py`
  - Features & selection: `ve/features.py`
  - Models & search spaces: `ve/models.py`
  - Training & ranking: `ve/training.py`
  - Tuning (top‑2): `ve/tuning.py`
  - Visualization: `ve/viz.py`
  - Reporting: `ve/reporting.py`
  - Logging: `ve/logging_utils.py`
- Reports: `output/final_modeling_report.md`, `reports/FINAL_PAPER.md`

