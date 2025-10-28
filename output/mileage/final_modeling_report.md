# Final Modeling Report

## Best Models

- EV: Gradient Boosting (Test R²: 0.0122)

- ICE: Lasso Regression (Test R²: -0.0149)

## EV Model Rankings (Top 5)

| Unnamed: 0        |   cv_mae_mean |   cv_mae_std |   train_mae |   train_r2 |   test_mae |   test_rmse |     test_r2 |   features_used |
|:------------------|--------------:|-------------:|------------:|-----------:|-----------:|------------:|------------:|----------------:|
| Gradient Boosting |       2024.03 |      85.3134 |     1521.44 |  0.377411  |    1999.66 |     2513.16 |  0.0122282  |               7 |
| Linear Regression |       1929.34 |     117.005  |     1901.3  |  0.0139322 |    2032.01 |     2533.47 | -0.00380048 |               7 |
| Lasso Regression  |       1927.75 |     114      |     1901.61 |  0.0119791 |    2040.52 |     2540.78 | -0.00960417 |               7 |
| Ridge Regression  |       1927.23 |     114.427  |     1902.04 |  0.0106547 |    2041.99 |     2541.69 | -0.0103225  |               7 |
| Random Forest     |       2026    |     101.876  |      732.38 |  0.855892  |    2074.84 |     2570.11 | -0.0330472  |               7 |

## ICE Model Rankings (Top 5)

| Unnamed: 0        |   cv_mae_mean |   cv_mae_std |   train_mae |   train_r2 |   test_mae |   test_rmse |    test_r2 |   features_used |
|:------------------|--------------:|-------------:|------------:|-----------:|-----------:|------------:|-----------:|----------------:|
| Lasso Regression  |       6109.23 |      135.944 |     6043.85 |  0.0110814 |    5403.48 |     6964.13 | -0.0149252 |               9 |
| Linear Regression |       6107.8  |      134.842 |     6041.12 |  0.0111975 |    5404.68 |     6964.18 | -0.0149405 |               9 |
| Ridge Regression  |       6105.11 |      140.413 |     6047.26 |  0.010637  |    5406.81 |     6964.93 | -0.0151577 |               9 |
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
### ICE – Lasso Regression

```json
{
  "alpha": 1.0,
  "copy_X": true,
  "fit_intercept": true,
  "max_iter": 10000,
  "positive": false,
  "precompute": false,
  "random_state": null,
  "selection": "cyclic",
  "tol": 0.0001,
  "warm_start": false
}
```
## System Specs

- python_version: 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 08:22:19) [Clang 14.0.6 ]
- numpy_version: 1.26.4
- pandas_version: 2.3.2
- sklearn_version: 1.4.1.post1
