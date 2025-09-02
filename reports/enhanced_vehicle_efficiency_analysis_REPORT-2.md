# Enhanced Vehicle Efficiency Analysis – Code Walkthrough & Report

_Generated: 2025-09-01 14:44:08_

## Purpose & Scope


This script builds a complete, reproducible pipeline to **predict vehicle efficiency** for **EV** and **ICE** cohorts.
It loads a CSV dataset, engineers domain-informed features, selects the most informative subset, trains and tunes multiple regression
models (with cross-validation), evaluates hold‑out performance, and exports models, metrics, and figures to an `output/` folder.

## Expected Input Schema

A CSV file at `data/vehicle_comparison_dataset_030417.csv` by default. Minimum required columns:

- `acceleration_0_100_kph_sec`
- `co2_emissions_g_per_km`
- `cost_per_km`
- `energy_consumption`
- `energy_storage_capacity`
- `lifespan_years`
- `maintenance_cost_annual`
- `mileage_km`
- `torque_Nm`
- `vehicle_type`

> `vehicle_type` is expected to contain two categories: `'EV'` and `'ICE'`.
> The script computes `efficiency = mileage_km / energy_consumption` internally.

## High‑Level Pipeline


1. **Load & Clean**
   - Read CSV and compute `efficiency = mileage_km / energy_consumption`.
   - Remove efficiency outliers via IQR (keep rows within `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]`).
   - Split into EV and ICE subsets and drop `vehicle_type`.

2. **Feature Engineering** (no target leakage):
   - Create 25+ numeric features (ratios, interactions, logs, scaled versions, and coarse categories converted to codes).
   - Explicitly **drop** `mileage_km` and `energy_consumption` from the modeling matrix to avoid leaking the target.

3. **Feature Selection (per cohort)**:
   - Keep features passing **(a)** correlation vs. `efficiency` (EV threshold 0.02; ICE 0.03) and **(b)** variance threshold 0.01.
   - Remove multicollinearity: greedily add features if pairwise |corr| < 0.85 (cap at 20 features).

4. **Modeling**:
   - **Preprocessing**: `PowerTransformer(method='yeo-johnson')` applied via a `ColumnTransformer` to selected features.
   - **Models** (baseline): Linear Regression, Ridge Regression, Lasso Regression, Random Forest, Gradient Boosting, Decision Tree.
   - **Optional models** auto‑included if installed: XGBoost, CatBoost, LightGBM.
   - **Validation**: 5‑fold `KFold(shuffle=True, random_state=42)` with MAE.
   - **Tuning**: `RandomizedSearchCV` with model‑specific grids when available.
   - **Hold‑out test**: 80/20 split; metrics: **MAE** and **R²**.

5. **Reporting & Artifacts**:
   - Ranked model tables (per cohort), multiple dashboards/plots, serialized best models, selected features, and parameter JSONs.
   - A human‑readable text report and a consolidated Markdown report.

## Data Leakage Prevention


- The target is `efficiency = mileage_km / energy_consumption`.
- Both `mileage_km` and `energy_consumption` are **explicitly removed** from the feature matrix after engineering.
- Categorical bins (e.g., torque/acceleration categories) are converted to codes and the original categories are dropped.

## Engineered Features (What & Why)

| Feature | Formula (from code) | Rationale |
|---|---|---|

| `power_efficiency` | `df['torque_Nm'] / df['acceleration_0_100_kph_sec']` | Higher torque with faster acceleration suggests drivetrain performance. |
| `storage_per_torque` | `df['energy_storage_capacity'] / df['torque_Nm']` | Energy capacity normalized by torque; powertrain balance. |
| `cost_efficiency` | `df['cost_per_km'] / df['torque_Nm']` | Operating cost relative to torque; proxy for cost-performance. |
| `maintenance_per_year` | `df['maintenance_cost_annual'] / df['lifespan_years']` | Annualized maintenance burden. |
| `maintenance_per_torque` | `df['maintenance_cost_annual'] / df['torque_Nm']` | Maintenance normalized by torque; durability vs. output. |
| `lifespan_torque_ratio` | `df['lifespan_years'] * df['torque_Nm']` | Aggregate service potential (years × torque). |
| `eco_efficiency` | `1 / (df['co2_emissions_g_per_km'] + 1)  # +1 to avoid division by zero` | Inverse emissions for monotonicity & stability (avoid div-by-zero with +1). |
| `green_performance` | `df['torque_Nm'] / (df['co2_emissions_g_per_km'] + 1)` | Torque per emission; ‘clean performance’. |
| `emission_intensity` | `df['co2_emissions_g_per_km'] / df['torque_Nm']` | Emissions per torque; inverse of above. |
| `emission_per_storage` | `df['co2_emissions_g_per_km'] / df['energy_storage_capacity']` | Emissions per capacity; environmental intensity. |
| `torque_category` | `pd.cut(df['torque_Nm'], bins=5, labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'])` | Coarse torque binning for nonlinearity capture (dropped post-encoding). |
| `acceleration_category` | `pd.cut(df['acceleration_0_100_kph_sec'], bins=5, labels=['Very Fast', 'Fast', 'Medium', 'Slow', 'Very Slow'])` | Coarse acceleration binning (dropped post-encoding). |
| `torque_category_num` | `df['torque_category'].cat.codes` | Encoded category (model-friendly). |
| `acceleration_category_num` | `df['acceleration_category'].cat.codes` | Encoded category (model-friendly). |
| `torque_squared` | `df['torque_Nm'] ** 2` | Nonlinear torque effects. |
| `acceleration_squared` | `df['acceleration_0_100_kph_sec'] ** 2` | Nonlinear acceleration effects. |
| `cost_squared` | `df['cost_per_km'] ** 2` | Nonlinear operating cost effects. |
| `torque_x_lifespan` | `df['torque_Nm'] * df['lifespan_years']` | Interaction: performance over lifetime. |
| `cost_x_maintenance` | `df['cost_per_km'] * df['maintenance_cost_annual']` | Interaction: opex interactions. |
| `storage_x_lifespan` | `df['energy_storage_capacity'] * df['lifespan_years']` | Interaction: capacity over lifetime. |
| `log_maintenance` | `np.log1p(df['maintenance_cost_annual'])` | Stabilize skew and compress outliers. |
| `log_torque` | `np.log1p(df['torque_Nm'])` | Stabilize skew and compress outliers. |
| `log_storage` | `np.log1p(df['energy_storage_capacity'])` | Stabilize skew and compress outliers. |
| `normalized_torque` | `df['torque_Nm'] / df['torque_Nm'].max()` | Scale torque 0–1 for comparability. |
| `normalized_cost` | `df['cost_per_km'] / df['cost_per_km'].max()` | Scale cost 0–1 for comparability. |
| `normalized_maintenance` | `df['maintenance_cost_annual'] / df['maintenance_cost_annual'].max()` | Scale maintenance 0–1 for comparability. |

## Feature Selection (How)


1. Compute Pearson correlations between candidate features and `efficiency`.
   - Threshold is **0.02** (EV) and **0.03** (ICE).
2. Remove low-variance features with `VarianceThreshold(0.01)`.
3. Combine both filters (intersection).
4. Greedy multicollinearity filtering: add features in order of |corr| with target, keeping only those with pairwise |corr| < **0.85**.
5. Cap selected features at **20**; fallback to original features if empty, then print a summary.

## Modeling & Evaluation Details


- **Train/Test split**: 80/20 with `random_state=42`.
- **Cross‑validation**: 5‑fold `KFold` (shuffled), scored with **negative MAE**.
- **Preprocessing**: Yeo‑Johnson power transform on **selected numeric features** only (via `ColumnTransformer`).  
  This generally benefits linear models; trees are scale‑invariant but will still receive transformed inputs.
- **Metrics recorded** per model:
  - `cv_mae_mean`, `cv_mae_std` (from CV)
  - `train_r2` (fit on train fold), `test_r2`, `test_mae` (on hold‑out)
- **Hyperparameter tuning**: `RandomizedSearchCV` with model‑specific grids (e.g., depth/estimators for forests/GBMs; alphas for Ridge/Lasso).

## Visualizations & Dashboards

The script saves the following figures:
- `output/comprehensive_main_dashboard.png`
- `output/correlation_analysis_dashboard.png`
- `output/cv_stability_comparison.png`
- `output/cv_stability_ev.png`
- `output/cv_stability_ice.png`
- `output/detailed_correlation_matrices.png`
- `output/enhanced_efficiency_dashboard.png`
- `output/individual_plots/01_ev_efficiency_distribution.png`
- `output/individual_plots/02_ice_efficiency_distribution.png`
- `output/individual_plots/03_ev_vs_ice_comparison.png`
- `output/individual_plots/04_model_r2_comparison.png`
- `output/individual_plots/05_model_mae_comparison.png`
- `output/individual_plots/06_ev_feature_importance.png`
- `output/individual_plots/07_ice_feature_importance.png`
- `output/individual_plots/08_ev_correlation_analysis.png`
- `output/individual_plots/09_ice_correlation_analysis.png`
- `output/individual_plots/10_ev_correlation_heatmap.png`
- `output/individual_plots/11_ice_correlation_heatmap.png`
- `output/individual_plots/12_correlation_comparison.png`
- `output/individual_plots/13_feature_engineering_summary.png`
- `output/individual_plots/14_cv_stability_comparison.png`
- `output/individual_plots/15_correlation_statistics.png`

## Additional Targets


A small **baseline** experiment is run for:
- `maintenance_cost_annual`
- `mileage_km`

Two models (Linear Regression and Random Forest) are trained with safe features (no leakage), and metrics/plots are saved per target.

## Outputs & Artifacts

When you run the script end‑to‑end, expect files like:
- `output/best_enhanced_ev_model.joblib`
- `output/best_enhanced_ice_model.joblib`
- `output/best_ev_model_coefficients.csv`
- `output/best_ev_model_importances.csv`
- `output/best_ice_model_coefficients.csv`
- `output/best_ice_model_importances.csv`
- `output/comprehensive_main_dashboard.png`
- `output/correlation_analysis_dashboard.png`
- `output/cv_stability_comparison.png`
- `output/cv_stability_ev.png`
- `output/cv_stability_ice.png`
- `output/detailed_correlation_matrices.png`
- `output/enhanced_analysis_report.txt`
- `output/enhanced_analysis_results.json`
- `output/enhanced_efficiency_dashboard.png`
- `output/ev_model_parameters.json`
- `output/ev_selected_features.json`
- `output/ice_model_parameters.json`
- `output/ice_selected_features.json`
- `output/individual_plots/01_ev_efficiency_distribution.png`
- `output/individual_plots/02_ice_efficiency_distribution.png`
- `output/individual_plots/03_ev_vs_ice_comparison.png`
- `output/individual_plots/04_model_r2_comparison.png`
- `output/individual_plots/05_model_mae_comparison.png`
- `output/individual_plots/06_ev_feature_importance.png`
- `output/individual_plots/07_ice_feature_importance.png`
- `output/individual_plots/08_ev_correlation_analysis.png`
- `output/individual_plots/09_ice_correlation_analysis.png`
- `output/individual_plots/10_ev_correlation_heatmap.png`
- `output/individual_plots/11_ice_correlation_heatmap.png`
- `output/individual_plots/12_correlation_comparison.png`
- `output/individual_plots/13_feature_engineering_summary.png`
- `output/individual_plots/14_cv_stability_comparison.png`
- `output/individual_plots/15_correlation_statistics.png`
- `output/{target}_baseline_results.csv`
- `output/{target}_{name.replace(" ", "_").lower()}_test_scatter.png`

## How to Run


```bash
# 1) Install deps (examples)
pip install numpy pandas matplotlib seaborn scikit-learn joblib
# Optional (if you want the extra models too):
pip install xgboost lightgbm catboost

# 2) Ensure your CSV is available (or override the path below)
python enhanced_vehicle_efficiency_analysis.py

# OR from Python:
from enhanced_vehicle_efficiency_analysis import EnhancedVehicleEfficiencyAnalyzer
analyzer = EnhancedVehicleEfficiencyAnalyzer(data_path="path/to/your.csv")
analyzer.run_enhanced_analysis()
```

## Notes, Limitations & Improvement Ideas


- **Uncalled helper**: `create_comprehensive_plots()` is defined (12‑panel dashboard) but **not invoked** in the main pipeline;
  call it if you want `enhanced_efficiency_dashboard.png` generated automatically.
- **Thresholds** for correlation (0.02/0.03), variance (0.01), and multicollinearity (0.85) are heuristic; consider data‑driven tuning.
- **PowerTransformer** is applied to all models. Linear models often benefit; tree ensembles are scale‑invariant, so consider bypassing the
  transform for tree models if you want strict parity comparisons.
- **Sklearn version string** is saved as `"unknown"` in results; switch to `sklearn.__version__` for traceability.
- **Synthetic data caveat**: If your dataset is synthetic (as implied by comments), validate with real‑world data before production use.
- **Reproducibility**: Seeds are set via `random_state=42` in splits and model constructors, but parallelism (`n_jobs=-1`) may introduce slight nondeterminism.
- **Data assumptions**: For EVs, `co2_emissions_g_per_km` may be zero; the code uses `+1` denominators to avoid divide‑by‑zero (e.g., `eco_efficiency`).

## End


## Hyperparameter Tuning — Full Details

This pipeline performs **model‑specific RandomizedSearchCV** on any model with an entry in `self.random_search_spaces`.

**Search configuration**:

- Estimator: the full `Pipeline([('preprocessor', PowerTransformer via ColumnTransformer), ('model', <estimator>)])`

- CV strategy: `KFold(n_splits=5, shuffle=True, random_state=42)`

- Scoring: `neg_mean_absolute_error` (lower MAE is better)

- `n_iter`: 30 combinations per model

- `n_jobs=-1`, `random_state=42`, and `verbose=1`


**Parameter distributions (built‑in models):**

- **Random Forest**
  - `model__n_estimators`: [50
  - `model__max_depth`: [None
  - `model__min_samples_split`: [2
  - `model__min_samples_leaf`: [1

- **Gradient Boosting**
  - `model__n_estimators`: [100
  - `model__learning_rate`: [0.3
  - `model__max_depth`: [2
  - `model__subsample`: [0.6

- **Decision Tree**
  - `model__max_depth`: [2
  - `model__min_samples_split`: [2
  - `model__min_samples_leaf`: [1

- **Ridge Regression**
  - `model__alpha`: np.logspace(-4

- **Lasso Regression**
  - `model__alpha`: np.logspace(-4

**Parameter distributions (optional models, used only if you add these estimators):**

- **XGBoost**
  - `model__n_estimators`: [200
  - `model__learning_rate`: [0.3
  - `model__max_depth`: [3
  - `model__subsample`: [0.6
  - `model__colsample_bytree`: [0.6

- **LightGBM**
  - `model__n_estimators`: [200
  - `model__learning_rate`: [0.3
  - `model__max_depth`: [-1
  - `model__subsample`: [0.6
  - `model__colsample_bytree`: [0.6

- **CatBoost**
  - `model__n_estimators`: [200
  - `model__learning_rate`: [0.3
  - `model__depth`: [3

> Note: `Linear Regression` has no hyperparameters and is trained without tuning (still cross‑validated).
