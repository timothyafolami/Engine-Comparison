# Final Modeling Report

## Best Models

- EV: Tuned Gradient Boosting (Test R²: 0.9825)

- ICE: Tuned Ridge Regression (Test R²: 0.9773)

## EV Model Rankings (Top 5)

| Unnamed: 0              |   cv_mae_mean |   cv_mae_std |   train_mae |   train_r2 |   test_mae |   test_rmse |     test_r2 |   features_used |
|:------------------------|--------------:|-------------:|------------:|-----------:|-----------:|------------:|------------:|----------------:|
| Tuned Gradient Boosting |        nan    |     nan      |     78.6148 |  0.999903  |    618.029 |     1337.28 |  0.982501   |             nan |
| Tuned Linear Regression |        nan    |     nan      |   2104.27   |  0.856754  |   1961.15  |     2926.08 |  0.916219   |             nan |
| Gradient Boosting       |       2024.03 |      85.3134 |   1521.44   |  0.377411  |   1999.66  |     2513.16 |  0.0122282  |               7 |
| Linear Regression       |       1929.34 |     117.005  |   1901.3    |  0.0139322 |   2032.01  |     2533.47 | -0.00380048 |               7 |
| Lasso Regression        |       1927.75 |     114      |   1901.61   |  0.0119791 |   2040.52  |     2540.78 | -0.00960417 |               7 |

## ICE Model Rankings (Top 5)

| Unnamed: 0             |   cv_mae_mean |   cv_mae_std |   train_mae |   train_r2 |   test_mae |   test_rmse |    test_r2 |   features_used |
|:-----------------------|--------------:|-------------:|------------:|-----------:|-----------:|------------:|-----------:|----------------:|
| Tuned Ridge Regression |        nan    |      nan     |     1698.04 |  0.973289  |    1541.46 |     2217.67 |  0.977296  |             nan |
| Tuned Lasso Regression |        nan    |      nan     |     1696.44 |  0.972574  |    1544.8  |     2261.43 |  0.976391  |             nan |
| Ridge Regression       |       6073.04 |      252.438 |     6015.63 |  0.0111507 |    6392.75 |     8227.57 | -0.0149343 |               8 |
| Lasso Regression       |       6090.04 |      242.843 |     6012.91 |  0.0113378 |    6397.8  |     8230.17 | -0.0155748 |               8 |
| Linear Regression      |       6100.5  |      239.417 |     6010.76 |  0.0113815 |    6401    |     8232.95 | -0.0162607 |               8 |

## Tuned vs Base – EV

- Base: Gradient Boosting — MAE 1999.66, R² 0.0122
- Tuned: Tuned Gradient Boosting — MAE 618.03, R² 0.9825
- Improvement: ΔMAE +1381.63 (lower is better), ΔR² +0.9703

## Tuned vs Base – ICE

- Base: Ridge Regression — MAE 6392.75, R² -0.0149
- Tuned: Tuned Ridge Regression — MAE 1541.46, R² 0.9773
- Improvement: ΔMAE +4851.29 (lower is better), ΔR² +0.9922

## Tuned Hyperparameters (Selected)

### EV – Tuned Gradient Boosting

```json
{
  "alpha": 0.9,
  "ccp_alpha": 0.0,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.02863095344799799,
  "loss": "squared_error",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 600,
  "n_iter_no_change": null,
  "random_state": 42,
  "subsample": 0.6628509119810311,
  "tol": 0.0001,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```
### ICE – Tuned Ridge Regression

```json
{
  "alpha": 0.0001,
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
