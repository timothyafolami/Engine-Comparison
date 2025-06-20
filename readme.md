Experiment Report: Predicting Vehicle Maintenance Costs Using Machine Learning
Overview
This report documents an experiment conducted to predict vehicle maintenance costs using a dataset comparing electric vehicles (EV) and internal combustion engine (ICE) vehicles. The experiment, implemented in the Jupyter Notebook experiment.ipynb, involves data loading, exploratory data analysis (EDA), preprocessing, model training, hyperparameter tuning, and evaluation. Multiple regression models were evaluated, with a particular focus on optimizing an XGBoost model. The results are visualized, and the tuned model and metrics are saved for future use.
Experiment Details
1. Libraries Import
The experiment begins by importing essential Python libraries for data manipulation, machine learning, and visualization:

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
