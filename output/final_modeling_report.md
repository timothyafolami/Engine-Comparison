# Final Modeling Report

## Best Models

- EV: Gradient Boosting (Test R²: 0.0016)

- ICE: XGBoost (Test R²: -0.0274)

## EV Model Rankings (Top 5)

| model | cv_mae_mean | cv_mae_std | train_mae | train_r2 | test_mae | test_rmse | test_r2 | features_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gradient Boosting | 2014.263785046196 | 80.31556082175685 | 1896.1992944209364 | 0.022126077691669965 | 2037.9097072998047 | 2526.6505751132268 | 0.001596348416512794 | 7.0 |
| XGBoost | 2215.674540778974 | 127.65637066781024 | 1898.7687289641553 | 0.017122633575160084 | 2040.3430884162576 | 2528.1940748923935 | 0.00037615079137176544 | 7.0 |
| CatBoost | 2081.2698432410943 | 84.16960556769601 | 1885.8611967043328 | 0.030467886682237766 | 2040.6313770673794 | 2528.710543112391 | -3.2304151473594445e-05 | 7.0 |
| Lasso Regression | 1933.865582067312 | 110.21203684520931 | 1914.8568902635952 | 0.0 | 2041.9172906978324 | 2529.387691619374 | -0.0005679613869529199 | 7.0 |
| LightGBM | 2170.5347227413367 | 113.23689609184096 | 1898.8383149958029 | 0.01727013105027686 | 2043.3437902317062 | 2532.113244816499 | -0.002725456274182525 | 7.0 |

## ICE Model Rankings (Top 5)

| model | cv_mae_mean | cv_mae_std | train_mae | train_r2 | test_mae | test_rmse | test_r2 | features_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 6885.776490802518 | 334.26978159350824 | 5788.0529214468625 | 0.014299769571672782 | 5330.633211288183 | 6627.677052375655 | -0.027397995790911622 | 5.0 |
| CatBoost | 6374.22924998624 | 313.92393811896704 | 5804.002942542261 | 0.008123971179172296 | 5333.6911623903925 | 6630.480152598714 | -0.028267232037246792 | 5.0 |
| Gradient Boosting | 6153.918094081896 | 355.4638890197771 | 5780.965539710824 | 0.017320417070495253 | 5332.124478173311 | 6631.586303477757 | -0.02861034853957256 | 5.0 |
| LightGBM | 6693.713235427293 | 282.30399276893735 | 5781.048517398035 | 0.016239195737929313 | 5327.286128064433 | 6635.928691308773 | -0.029957865479679713 | 5.0 |
| Ridge Regression | 5834.843299820534 | 184.52721080689085 | 5805.542462715296 | 0.006110589648841591 | 5340.504710209606 | 6656.007166708573 | -0.036200027072738505 | 5.0 |

## Tuned Hyperparameters (Selected)

### EV – Gradient Boosting

```json
{
  "alpha": 0.9,
  "ccp_alpha": 0.0,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.001,
  "loss": "squared_error",
  "max_depth": 4,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "random_state": 42,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```
### ICE – XGBoost

```json
{
  "objective": "reg:squarederror",
  "base_score": null,
  "booster": null,
  "callbacks": null,
  "colsample_bylevel": null,
  "colsample_bynode": null,
  "colsample_bytree": 1.0,
  "device": null,
  "early_stopping_rounds": null,
  "enable_categorical": false,
  "eval_metric": null,
  "feature_types": null,
  "gamma": null,
  "grow_policy": null,
  "importance_type": null,
  "interaction_constraints": null,
  "learning_rate": 0.0005,
  "max_bin": null,
  "max_cat_threshold": null,
  "max_cat_to_onehot": null,
  "max_delta_step": null,
  "max_depth": 4,
  "max_leaves": null,
  "min_child_weight": null,
  "missing": NaN,
  "monotone_constraints": null,
  "multi_strategy": null,
  "n_estimators": 200,
  "n_jobs": null,
  "num_parallel_tree": null,
  "random_state": 42,
  "reg_alpha": null,
  "reg_lambda": null,
  "sampling_method": null,
  "scale_pos_weight": null,
  "subsample": 0.6,
  "tree_method": null,
  "validate_parameters": null,
  "verbosity": null
}
```
## Best EV Model – Feature Importances

| feature | importance |
| --- | --- |
| co2_emissions_g_per_km | 0.15959359318963548 |
| normalized_cost | 0.09382805675140708 |
| acceleration_0_100_kph_sec | 0.12456793030894814 |
| maintenance_per_year | 0.00043950847046567225 |
| torque_squared | 0.23878745134593507 |
| lifespan_years | 0.15281617963715796 |
| torque_x_lifespan | 0.2299672802964507 |

## Best ICE Model – Feature Importances

| feature | importance |
| --- | --- |
| cost_x_maintenance | 0.16680631 |
| log_torque | 0.19413222 |
| maintenance_per_torque | 0.24425149 |
| normalized_cost | 0.18564726 |
| power_efficiency | 0.20916273 |

## Best ICE Model – Coefficients

| feature | coefficient |
| --- | --- |
| cost_x_maintenance | 65.30082970829244 |
| log_torque | -150.63721143349355 |
| maintenance_per_torque | 286.92750596832633 |
| normalized_cost | 430.7934504095362 |
| power_efficiency | 123.86953204595342 |

## System Specs

- python_version: 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 12:55:12) [Clang 14.0.6 ]
- numpy_version: 1.26.4
- pandas_version: 2.1.4
- sklearn_version: unknown
