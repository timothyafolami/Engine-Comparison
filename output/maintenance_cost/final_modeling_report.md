# Final Modeling Report

## Best Models

- EV: Tuned Gradient Boosting (Test R²: 0.9929)

- ICE: Tuned Ridge Regression (Test R²: 0.9804)

## EV Model Rankings (Top 5)

| Unnamed: 0              |   cv_mae_mean |   cv_mae_std |   train_mae |   train_r2 |   test_mae |   test_rmse |     test_r2 |   features_used |
|:------------------------|--------------:|-------------:|------------:|-----------:|-----------:|------------:|------------:|----------------:|
| Tuned Gradient Boosting |        nan    |     nan      |     202.832 |  0.999317  |    544.106 |     853.146 |  0.992878   |             nan |
| Tuned Linear Regression |        nan    |     nan      |    1148.4   |  0.968633  |   1086.15  |    1809.2   |  0.967971   |             nan |
| Gradient Boosting       |       2024.03 |      85.3134 |    1521.44  |  0.377411  |   1999.66  |    2513.16  |  0.0122282  |               7 |
| Linear Regression       |       1934.08 |     114.982  |    1904.6   |  0.0119052 |   2034.76  |    2535.75  | -0.00560465 |               7 |
| Lasso Regression        |       1933.74 |     114.027  |    1904.09  |  0.0118033 |   2036.95  |    2537.5   | -0.00700021 |               7 |

## ICE Model Rankings (Top 5)

| Unnamed: 0             |   cv_mae_mean |   cv_mae_std |   train_mae |   train_r2 |   test_mae |   test_rmse |    test_r2 |   features_used |
|:-----------------------|--------------:|-------------:|------------:|-----------:|-----------:|------------:|-----------:|----------------:|
| Tuned Ridge Regression |        nan    |      nan     |     1318.79 |  0.982733  |    1275.63 |     2058.34 |  0.980441  |             nan |
| Tuned Lasso Regression |        nan    |      nan     |     1318.38 |  0.982734  |    1279.2  |     2060.96 |  0.980391  |             nan |
| Ridge Regression       |       6085.66 |      235.892 |     6018.84 |  0.0143329 |    6420.14 |     8241.99 | -0.0184936 |               8 |
| Lasso Regression       |       6088.07 |      233.268 |     6018.94 |  0.0143805 |    6423.76 |     8244.34 | -0.0190747 |               8 |
| Linear Regression      |       6088.83 |      232.253 |     6019.41 |  0.0143909 |    6426.3  |     8245.91 | -0.019463  |               8 |

## Tuned vs Base – EV

- Base: Gradient Boosting — MAE 1999.66, R² 0.0122
- Tuned: Tuned Gradient Boosting — MAE 544.11, R² 0.9929
- Improvement: ΔMAE +1455.55 (lower is better), ΔR² +0.9806

## Tuned vs Base – ICE

- Base: Ridge Regression — MAE 6420.14, R² -0.0185
- Tuned: Tuned Ridge Regression — MAE 1275.63, R² 0.9804
- Improvement: ΔMAE +5144.51 (lower is better), ΔR² +0.9989

## Tuned Hyperparameters (Selected)

### EV – Tuned Gradient Boosting

```json
{
  "alpha": 0.9,
  "ccp_alpha": 0.0,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.022929257847542305,
  "loss": "squared_error",
  "max_depth": 3,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 1200,
  "n_iter_no_change": null,
  "random_state": 42,
  "subsample": 0.6288253656298914,
  "tol": 0.0001,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```
### ICE – Tuned Ridge Regression

```json
{
  "alpha": 0.17012542798525893,
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
