# Final Modeling Report

## Best Models

- EV: Gradient Boosting (Test R²: 0.0122)

- ICE: Ridge Regression (Test R²: -0.0276)

## EV Model Rankings (Top 5)

| Unnamed: 0        |   cv_mae_mean |   cv_mae_std |   train_mae |   train_r2 |   test_mae |   test_rmse |     test_r2 |   features_used |
|:------------------|--------------:|-------------:|------------:|-----------:|-----------:|------------:|------------:|----------------:|
| Gradient Boosting |       2024.03 |      85.3134 |     1521.44 |  0.377411  |    1999.66 |     2513.16 |  0.0122282  |               7 |
| Linear Regression |       1934.08 |     114.982  |     1904.6  |  0.0119052 |    2034.76 |     2535.75 | -0.00560465 |               7 |
| Lasso Regression  |       1933.74 |     114.027  |     1904.09 |  0.0118033 |    2036.95 |     2537.5  | -0.00700021 |               7 |
| Ridge Regression  |       1933.05 |     114.595  |     1904.15 |  0.0118001 |    2036.97 |     2537.54 | -0.00702465 |               7 |
| Random Forest     |       2026    |     101.876  |      732.38 |  0.855892  |    2074.84 |     2570.11 | -0.0330472  |               7 |

## ICE Model Rankings (Top 5)

| Unnamed: 0        |   cv_mae_mean |   cv_mae_std |   train_mae |   train_r2 |   test_mae |   test_rmse |    test_r2 |   features_used |
|:------------------|--------------:|-------------:|------------:|-----------:|-----------:|------------:|-----------:|----------------:|
| Ridge Regression  |       6123.49 |      156.448 |     6046.08 |  0.0136384 |    5455.99 |     7007.49 | -0.0276022 |               9 |
| Lasso Regression  |       6127.57 |      158.305 |     6046.16 |  0.0136601 |    5458.34 |     7009.86 | -0.0282974 |               9 |
| Linear Regression |       6128.29 |      159.962 |     6046.26 |  0.0136677 |    5460.04 |     7011.96 | -0.0289147 |               9 |
| Random Forest     |       6445.61 |      211.908 |     2349.36 |  0.848063  |    5714.99 |     7306.76 | -0.117248  |               9 |
| Gradient Boosting |       6453.23 |      231.559 |     4540.39 |  0.489627  |    5878.49 |     7327.27 | -0.123529  |               9 |

## Tuned Hyperparameters (Selected)

### EV – Gradient Boosting

```json
{
  "alpha": 0.9,
  "ccp_alpha": 0.0,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.1,
  "loss": "squared_error",
  "max_depth": 3,
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
### ICE – Ridge Regression

```json
{
  "alpha": 1.0,
  "copy_X": true,
  "fit_intercept": true,
  "max_iter": null,
  "positive": false,
  "random_state": null,
  "solver": "auto",
  "tol": 0.0001
}
```
## System Specs

- python_version: 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 08:22:19) [Clang 14.0.6 ]
- numpy_version: 1.26.4
- pandas_version: 2.3.2
- sklearn_version: 1.4.1.post1
