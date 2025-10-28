# Final Modeling Report

## Best Models

- EV: Gradient Boosting (Test R²: 0.0122)

- ICE: Tuned Ridge Regression (Test R²: -0.0385)

## EV Model Rankings (Top 5)

| Unnamed: 0              |   cv_mae_mean |   cv_mae_std |   train_mae |   train_r2 |   test_mae |   test_rmse |      test_r2 |   features_used |
|:------------------------|--------------:|-------------:|------------:|-----------:|-----------:|------------:|-------------:|----------------:|
| Gradient Boosting       |       2024.03 |      85.3134 |     1521.44 | 0.377411   |    1999.66 |     2513.16 |  0.0122282   |               7 |
| Tuned Gradient Boosting |        nan    |     nan      |     1912.38 | 0.00290024 |    2041.71 |     2528.94 | -0.000212273 |             nan |
| Linear Regression       |       1929.34 |     117.005  |     1901.3  | 0.0139322  |    2032.01 |     2533.47 | -0.00380048  |               7 |
| Tuned Linear Regression |        nan    |     nan      |     1901.3  | 0.0139322  |    2032.01 |     2533.47 | -0.00380048  |             nan |
| Lasso Regression        |       1927.75 |     114      |     1901.61 | 0.0119791  |    2040.52 |     2540.78 | -0.00960417  |               7 |

## ICE Model Rankings (Top 5)

| Unnamed: 0             |   cv_mae_mean |   cv_mae_std |   train_mae |   train_r2 |   test_mae |   test_rmse |    test_r2 |   features_used |
|:-----------------------|--------------:|-------------:|------------:|-----------:|-----------:|------------:|-----------:|----------------:|
| Tuned Ridge Regression |        nan    |      nan     |     5804.24 | 0.00649822 |    5344.12 |     6663.47 | -0.0385241 |             nan |
| Tuned Lasso Regression |        nan    |      nan     |     5804.39 | 0.00680463 |    5344.44 |     6670.6  | -0.0407473 |             nan |
| Ridge Regression       |       5832.42 |      178.317 |     5800.14 | 0.00807258 |    5377.61 |     6711.42 | -0.0535248 |               5 |
| Lasso Regression       |       5845.41 |      178.298 |     5800.5  | 0.00819108 |    5380.95 |     6714.56 | -0.054512  |               5 |
| Linear Regression      |       5853.39 |      179.95  |     5801.08 | 0.00827657 |    5382.16 |     6716.56 | -0.0551404 |               5 |

## Tuned vs Base – ICE

- Base: Ridge Regression — MAE 5377.61, R² -0.0535
- Tuned: Tuned Ridge Regression — MAE 5344.12, R² -0.0385
- Improvement: ΔMAE +33.49 (lower is better), ΔR² +0.0150

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
  "alpha": 24.244620170823307,
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
