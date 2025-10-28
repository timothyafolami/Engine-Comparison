# Final Modeling Report

## Best Models

- EV: Gradient Boosting (Test R²: 0.0122)

- ICE: Tuned Ridge Regression (Test R²: -0.0387)

## EV Model Rankings (Top 5)

| Unnamed: 0              |   cv_mae_mean |   cv_mae_std |   train_mae |   train_r2 |   test_mae |   test_rmse |      test_r2 |   features_used |
|:------------------------|--------------:|-------------:|------------:|-----------:|-----------:|------------:|-------------:|----------------:|
| Gradient Boosting       |       2024.03 |      85.3134 |     1521.44 | 0.377411   |    1999.66 |     2513.16 |  0.0122282   |               7 |
| Tuned Gradient Boosting |        nan    |     nan      |     1912.21 | 0.00316156 |    2041.6  |     2528.89 | -0.000174132 |             nan |
| Linear Regression       |       1934.08 |     114.982  |     1904.6  | 0.0119052  |    2034.76 |     2535.75 | -0.00560465  |               7 |
| Tuned Linear Regression |        nan    |     nan      |     1904.6  | 0.0119052  |    2034.76 |     2535.75 | -0.00560465  |             nan |
| Lasso Regression        |       1933.74 |     114.027  |     1904.09 | 0.0118033  |    2036.95 |     2537.5  | -0.00700021  |               7 |

## ICE Model Rankings (Top 5)

| Unnamed: 0             |   cv_mae_mean |   cv_mae_std |   train_mae |   train_r2 |   test_mae |   test_rmse |    test_r2 |   features_used |
|:-----------------------|--------------:|-------------:|------------:|-----------:|-----------:|------------:|-----------:|----------------:|
| Tuned Ridge Regression |        nan    |      nan     |     5803.9  | 0.00657757 |    5343.68 |     6663.89 | -0.0386556 |             nan |
| Tuned Lasso Regression |        nan    |      nan     |     5802.5  | 0.00698615 |    5345.98 |     6676.59 | -0.0426172 |             nan |
| Lasso Regression       |       5835.3  |      185.844 |     5800.24 | 0.00740591 |    5366.7  |     6697.76 | -0.0492406 |               5 |
| Ridge Regression       |       5834.84 |      184.527 |     5800.3  | 0.00740589 |    5367.07 |     6698.08 | -0.0493406 |               5 |
| Linear Regression      |       5836.6  |      186.517 |     5800.27 | 0.00740609 |    5367.33 |     6698.19 | -0.0493758 |               5 |

## Tuned vs Base – ICE

- Base: Ridge Regression — MAE 5367.07, R² -0.0493
- Tuned: Tuned Ridge Regression — MAE 5343.68, R² -0.0387
- Improvement: ΔMAE +23.39 (lower is better), ΔR² +0.0107

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
### ICE – Tuned Ridge Regression

```json
{
  "alpha": 661.4740641230146,
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
