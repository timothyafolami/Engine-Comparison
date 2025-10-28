# Vehicle Efficiency Analysis Package

A comprehensive machine learning package for analyzing and predicting vehicle efficiency across different vehicle types (Electric Vehicles and Internal Combustion Engine vehicles).

## Overview

This repository contains a professional Python package for vehicle efficiency analysis with:
- Modular architecture following Python packaging best practices
- Command-line interface for easy usage
- Comprehensive machine learning pipeline
- Advanced visualization and reporting capabilities
- Support for multiple ML models and hyperparameter optimization

## Installation

### Option 1: Development Installation
```bash
# Clone the repository
git clone <repository-url>
cd "Engine Comparison"

# Install in development mode
pip install -e .
```

### Option 2: Direct Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Or install with the package
pip install .
```

## Quick Start

### Method 1: Command Line Interface
```bash
# Basic usage
vehicle-efficiency data/vehicle_comparison_dataset_030417.csv

# With custom options
vehicle-efficiency data/vehicle_comparison_dataset_030417.csv \
  --output-dir results \
  --rank-by r2 \
  --tuning-metric mae \
  --search-method optuna
```

### Method 2: Python Module
```bash
# Using the CLI module
python -m vehicle_efficiency.cli data/vehicle_comparison_dataset_030417.csv --output-dir results

# Using the main entry point
python main.py  # (requires vehicle_efficiency_data.csv in current directory)
```

### Method 3: Python API
```python
from vehicle_efficiency import EnhancedPipeline

# Create and run pipeline
pipeline = EnhancedPipeline(
    data_path="data/vehicle_comparison_dataset_030417.csv",
    output_dir="results",
    rank_by="r2",
    enable_tuning=True,
    full_viz=True
)

pipeline.run()
```

## Package Structure

```
src/vehicle_efficiency/
├── __init__.py              # Main package exports
├── cli.py                   # Command-line interface
├── core/                    # Core pipeline components
│   ├── __init__.py
│   ├── pipeline.py          # Main pipeline orchestrator
│   ├── training.py          # Model training and evaluation
│   └── tuning.py           # Hyperparameter optimization
├── data/                    # Data processing
│   ├── __init__.py
│   └── preprocessing.py     # Data loading and preprocessing
├── features/                # Feature engineering
│   ├── __init__.py
│   ├── engineering.py       # Feature creation and transformation
│   └── selection.py         # Feature selection algorithms
├── models/                  # Model definitions
│   ├── __init__.py
│   └── model_registry.py    # Model registry and configurations
└── utils/                   # Utilities
    ├── __init__.py
    ├── logging.py           # Logging configuration
    ├── reporting.py         # Report generation
    └── visualization.py     # Plotting and visualization
```

## Command Line Options

Flags:
- `--rank-by {r2|mae}`: Choose ranking metric (default r2).
- `--no-tuning`: Disable hyperparameter tuning for faster runs.

Outputs are written to `output/`:
- Rankings: `enhanced_ev_model_rankings.csv`, `enhanced_ice_model_rankings.csv`
- Best models: `best_enhanced_ev_model.joblib`, `best_enhanced_ice_model.joblib`
- Parameters: `ev_model_parameters.json`, `ice_model_parameters.json`
- Selected features: `ev_selected_features.json`, `ice_selected_features.json`
- Summary JSON: `enhanced_analysis_results.json`
- Final report: `final_modeling_report.md`
- Plots: `enhanced_efficiency_dashboard.png`, `cv_stability_{ev,ice,comparison}.png`
  - If `--full-viz` is set: `individual_plots/01..15_*.png`, `correlation_analysis_dashboard.png`, `detailed_correlation_matrices.png`

Fine-tune Top Models
You can fine-tune the top two models per cohort using Optuna for boosting models and sklearn search for others:

```bash
python scripts/finetune_top.py \
  --data-path data/vehicle_comparison_dataset_030417.csv \
  --vehicle both \
  --rank-by r2 \
  --tuning-metric mae \
  --search random \
  --random-iter 60 \
  --optuna-trials 80
```

Notes:
- Boosting models (Gradient Boosting, XGBoost, LightGBM, CatBoost) use Optuna if installed; falls back to sklearn search otherwise.
- Non-boosting models use `RandomizedSearchCV` by default, or `GridSearchCV` with `--search grid`.
- Tuning artifacts are saved to `output/` as `tuned_<ev|ice>_<model>.joblib`, `tuned_*_metrics.json`, and summaries as `tuned_<ev|ice>_top2_summary.json`.

Disable defaults if needed:
- `--no-full-viz` to skip heavy plots
- `--no-fine-tune-top2` to skip fine-tuning

## What's New (Package Refactor)

The codebase has been completely restructured into a professional Python package:

### Package Architecture
- **Modular Design**: Split monolithic scripts into focused, reusable modules
- **Standard Structure**: Follows Python packaging best practices with `src/` layout
- **Clean Imports**: Proper package hierarchy with clear module boundaries
- **CLI Interface**: Professional command-line interface with comprehensive options
- **API Access**: Programmatic access to all functionality through clean APIs

### Module Organization
- **`src/vehicle_efficiency/core/`**: Pipeline orchestration, training, and hyperparameter tuning
- **`src/vehicle_efficiency/data/`**: Data loading, preprocessing, and efficiency computation
- **`src/vehicle_efficiency/features/`**: Feature engineering and selection (prevents data leakage)
- **`src/vehicle_efficiency/models/`**: Model registry and hyperparameter search spaces
- **`src/vehicle_efficiency/utils/`**: Logging, reporting, and visualization utilities

### Enhanced Features
- **Multiple Entry Points**: CLI (`vehicle-efficiency`), Python module, and programmatic API
- **Flexible Configuration**: Comprehensive command-line options and Python API
- **Professional Logging**: Structured logging with configurable levels
- **Comprehensive Testing**: Import validation and end-to-end pipeline testing
- **Modern Packaging**: `pyproject.toml`, `setup.py`, and proper dependency management



## Legacy and Migration

### Backward Compatibility
- The original scripts (`enhanced_vehicle_efficiency_analysis.py`, `vehicle_efficiency_analysis.py`) remain available for reference
- The `exps/` folder contains earlier maintenance-cost and visualization experiments
- All functionality has been preserved and enhanced in the new package structure

### Migration Guide
- **Old CLI**: `python enhanced_vehicle_efficiency_analysis.py data.csv`
- **New CLI**: `vehicle-efficiency data.csv` or `python -m vehicle_efficiency.cli data.csv`
- **Old imports**: Direct script execution
- **New imports**: `from vehicle_efficiency import EnhancedPipeline, load_and_prepare_data`

Notes
- Data leakage is prevented by excluding `mileage_km` and `energy_consumption` from the feature matrix when predicting `efficiency` (which is `mileage_km / energy_consumption`).
- If your dataset differs from `data/vehicle_comparison_dataset_030417.csv`, ensure matching column names.

Deprecated
- `scripts/generate_report.py` now forwards to `ve.reporting.write_final_report_markdown`. Prefer `scripts/run_enhanced.py` which generates the report automatically.

---

Maintenance Cost Experiment (Legacy)
The repository also includes a legacy experiment focused on predicting annual maintenance cost (see `exps/experiment.ipynb`). It demonstrates a parallel setup (EDA, preprocessing, multiple models, hyperparameter tuning) and persists a tuned XGBoost model and metrics. While still usable, the modular enhanced pipeline is recommended for current work.

Data Handling: numpy for numerical operations, pandas for data manipulation.
Model Persistence: joblib for saving models.
File Handling: json for metrics storage, pathlib for path management.
Preprocessing: sklearn.compose.ColumnTransformer, sklearn.preprocessing.StandardScaler, and sklearn.preprocessing.OneHotEncoder for data preprocessing.
Model Selection an  Evaluation: sklearn.model_selection modules (GridSearchCV, cross_val_score, KFold, train_test_split) and sklearn.metrics (e.g., mean_absolute_error, root_mean_squared_error, r2_score, mean_squared_error) for model training and evaluation.
Models: Core scikit-learn models (LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor) and optional advanced models (catboost.CatBoostRegressor, lightgbm.LGBMRegressor, xgboost.XGBRegressor) with lazy imports.
Visualization: seaborn and matplotlib.pyplot for plotting.

Optional models are imported with try-except blocks to ensure the script runs even if these libraries are unavailable, storing them in a dictionary OPTIONAL_MODELS with a consistent random_state=42.
2. Data Loading and Exploration
Data Source
The dataset, vehicle_comparison_dataset_030417.csv, is loaded from the ../data/ directory using pd.read_csv. It contains 2000 rows and 10 columns, representing vehicle attributes.
Dataset Structure
The dataset includes:

Categorical Feature:
vehicle_type: EV or ICE.


Numerical Features:
energy_consumption: Energy use (kWh/100km for EVs, liters/100km for ICE).
co2_emissions_g_per_km: CO2 emissions in grams per kilometer.
maintenance_cost_annual: Annual maintenance cost (target variable, in currency units).
cost_per_km: Cost per kilometer.
energy_storage_capacity: Battery/fuel capacity (kWh for EVs, liters for ICE).
mileage_km: Total mileage in kilometers.
acceleration_0_100_kph_sec: Time to accelerate from 0 to 100 km/h (seconds).
torque_Nm: Torque in Newton-meters.
lifespan_years: Vehicle lifespan in years.



A preview of the first 10 rows (df.head(10)) shows a mix of EV and ICE vehicles with varied attribute values. The shape (2000, 10) confirms the dataset size.
Exploratory Data Analysis (EDA)
EDA involves visualizing feature distributions:

Box Plots: Generated to compare feature distributions between EV and ICE vehicles (output as an image in the notebook). Key observations:
EVs have higher median energy_consumption (13-15 vs. 6-10) and torque_Nm (250-350 vs. 150-250), lower co2_emissions_g_per_km (near 0 or negative vs. 150-175), and lower maintenance_cost_annual (40,000-50,000 vs. 90,000-100,000).
ICE vehicles show higher cost_per_km (0.30-0.35 vs. 0.12-0.15) and mileage_km (150,000-200,000 vs. 130,000-140,000).
EVs accelerate faster (6-8 seconds vs. 8-10 seconds for ICE).



These insights highlight vehicle_type as a critical feature and suggest potential predictors for maintenance costs.
3. Data Preprocessing

Data Splitting: The dataset is split into features (X) and target (y, maintenance_cost_annual) and then into training (80%) and test (20%) sets using train_test_split with random_state=42.
Preprocessing Pipeline:
A ColumnTransformer applies:
StandardScaler to numerical columns for standardization.
OneHotEncoder to vehicle_type for categorical encoding.


This is integrated into a Pipeline with each model to ensure consistent preprocessing.



4. Model Development and Evaluation
Baseline Models
Seven regression models are trained and evaluated:

Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
CatBoost Regressor
LightGBM Regressor
XGBoost Regressor

Cross-validation (5-fold) is performed using cross_val_score, with metrics (RMSE, MAE, R²) saved to model_comparison_results.csv.
Hyperparameter Tuning for XGBoost
XGBoost is selected for tuning due to its flexibility and feature importance capabilities:

Pipeline: Combines preprocessing with XGBRegressor (objective='reg:squarederror', random_state=42, n_jobs=-1).
Hyperparameter Grid:
n_estimators: [100, 200, 300]
max_depth: [3, 5, 7]
learning_rate: [0.01, 0.05, 0.1]
min_child_weight: [1, 3, 5]
subsample: [0.6, 0.8, 1.0]
colsample_bytree: [0.6, 0.8, 1.0]


GridSearchCV:
5-fold cross-validation.
Scoring metrics: neg_mean_absolute_error, neg_root_mean_squared_error, r2.
Refit on neg_mae for model selection.
n_jobs=-1 for parallel processing, verbose=2 for progress output.



The grid search fits the training data, identifying the best parameters:

model__colsample_bytree: 0.8
model__learning_rate: 0.01
model__max_depth: 3
model__min_child_weight: 3
model__n_estimators: 300
model__subsample: 0.6

Model Evaluation

Cross-Validation Metrics (from tuned_xgboost_metrics.json):
MAE: 4088.91
RMSE: 6033.43
R²: 0.611


Training Set Metrics:
MAE: 3821.36
RMSE: 5557.52
R²: 0.671


Test Set Metrics:
MAE: 3949.18
RMSE: 5642.49
R²: 0.636



The tuned model is evaluated on both sets, showing improved performance over the untuned XGBoost.
Visualizations

Feature Importance: A horizontal bar plot displays feature importances from the tuned XGBoost model, identifying key predictors (specific features not detailed in output but inferred to include energy_consumption, cost_per_km, etc.).
Predictions vs. Actual: A scatter plot compares test set predictions to actual values, with a diagonal line (y=x) and metrics annotated (MAE, RMSE, R²).

Model Saving

The tuned XGBoost model is saved to output/tuned_xgboost_model.joblib.
Metrics and best parameters are saved to output/tuned_xgboost_metrics.json.

5. Results Summary
Model Comparison
From model_comparison_results.csv:



Model
Test RMSE
Test MAE
Test R²



Linear Regression
5626.91
3903.24
0.638


Random Forest
5657.26
4050.87
0.634


Gradient Boosting
5632.56
3977.80
0.637


CatBoost
5949.83
4184.89
0.595


LightGBM
6149.52
4302.86
0.568


XGBoost (untuned)
6138.98
4368.90
0.569


Decision Tree
8192.77
5596.27
0.233



Linear Regression outperforms others with the lowest test MAE (3903.24) and highest R² (0.638).
Tuned XGBoost (test MAE: 3949.18, R²: 0.636) improves significantly over the untuned version (MAE: 4368.90, R²: 0.569) but falls short of Linear Regression.

Tuned XGBoost Performance

R²: 0.636 indicates 63.6% of variance explained.
MAE: 3949.18 suggests an average error of ~3949 units, reasonable given the target’s scale (tens of thousands).

6. Analysis

Best Model: Linear Regression’s superior performance suggests a largely linear relationship between features and maintenance_cost_annual.
Tuned XGBoost: Offers competitive performance and feature importance insights, making it valuable for interpretability.
Key Insights: EVs’ lower maintenance costs and distinct feature distributions (e.g., lower CO2 emissions, higher torque) likely influence predictions.
Limitations: Negative co2_emissions_g_per_km values for EVs may indicate data anomalies (e.g., offsets) requiring preprocessing.

7. Conclusion
The experiment successfully predicts vehicle maintenance costs, with Linear Regression as the top performer. The tuned XGBoost model provides a robust alternative with interpretability benefits. Future work could explore:

Feature Engineering: Address CO2 anomalies, add interaction terms.
Tuning: Broader hyperparameter grids or other models.
Data: Additional samples or features (e.g., usage patterns).

8. Files Generated

output/tuned_xgboost_model.joblib: Tuned XGBoost model.
output/tuned_xgboost_metrics.json: Metrics and parameters.
model_comparison_results.csv: Model comparison metrics.

This report provides a thorough overview of the experiment, suitable for replication or extension.
