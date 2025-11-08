# Final Modeling Report

## Best Models

- EV: LightGBM (Test R²: 0.0031)

- ICE: XGBoost (Test R²: -0.0274)

## EV Model Rankings (Top 5)

| Unnamed: 0        |   cv_mae_mean |   cv_mae_std |   train_mae |   train_r2 |   test_mae |   test_rmse |     test_r2 |   features_used |
|:------------------|--------------:|-------------:|------------:|-----------:|-----------:|------------:|------------:|----------------:|
| LightGBM          |       2185.23 |     120.286  |     1900.28 |  0.0161239 |    2038.2  |     2524.7  | 0.00313974  |               7 |
| CatBoost          |       2076.62 |     101.133  |     1885.04 |  0.0311369 |    2039.55 |     2526.84 | 0.00144818  |               7 |
| Random Forest     |       2026    |     101.876  |     1795.89 |  0.122959  |    2023.03 |     2527.31 | 0.00107271  |               7 |
| XGBoost           |       2166.01 |     137.404  |     1899.36 |  0.0167021 |    2040.02 |     2527.57 | 0.000865676 |               7 |
| Gradient Boosting |       2024.03 |      85.3134 |     1899.97 |  0.0188556 |    2043.35 |     2528.45 | 0.000173258 |               7 |

## ICE Model Rankings (Top 5)

| Unnamed: 0        |   cv_mae_mean |   cv_mae_std |   train_mae |   train_r2 |   test_mae |   test_rmse |    test_r2 |   features_used |
|:------------------|--------------:|-------------:|------------:|-----------:|-----------:|------------:|-----------:|----------------:|
| XGBoost           |       6885.78 |      334.27  |     5788.05 | 0.0142998  |    5330.63 |     6627.68 | -0.027398  |               5 |
| CatBoost          |       6376.82 |      311.011 |     5804    | 0.00812397 |    5333.69 |     6630.48 | -0.0282672 |               5 |
| Gradient Boosting |       6157.77 |      353.679 |     5780.97 | 0.0173204  |    5332    |     6631.53 | -0.028594  |               5 |
| LightGBM          |       6675.81 |      241.792 |     5780.96 | 0.0163211  |    5327.97 |     6636.48 | -0.0301293 |               5 |
| Ridge Regression  |       5832.42 |      178.317 |     5805.92 | 0.00598409 |    5340.81 |     6656.17 | -0.0362508 |               5 |

## Tuned Hyperparameters (Selected)

### EV – LightGBM

```json
{
  "boosting_type": "gbdt",
  "class_weight": null,
  "colsample_bytree": 0.6,
  "importance_type": "split",
  "learning_rate": 0.001,
  "max_depth": 3,
  "min_child_samples": 20,
  "min_child_weight": 0.001,
  "min_split_gain": 0.0,
  "n_estimators": 200,
  "n_jobs": null,
  "num_leaves": 31,
  "objective": null,
  "random_state": 42,
  "reg_alpha": 0.0,
  "reg_lambda": 0.0,
  "subsample": 0.6,
  "subsample_for_bin": 200000,
  "subsample_freq": 0,
  "verbose": -1
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
  "feature_weights": null,
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
## System Specs

- python_version: 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 08:22:19) [Clang 14.0.6 ]
- numpy_version: 1.26.4
- pandas_version: 2.3.2
- sklearn_version: 1.4.1.post1
