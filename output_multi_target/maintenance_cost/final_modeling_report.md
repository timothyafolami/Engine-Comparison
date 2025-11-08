# Final Modeling Report

## Best Models

- EV: Linear Regression (Test R²: 1.0000)

- ICE: Linear Regression (Test R²: 1.0000)

## EV Model Rankings (Top 5)

| Unnamed: 0             |   cv_mae_mean |    cv_mae_std |   train_mae |   train_r2 |    test_mae |   test_rmse |   test_r2 |   features_used |
|:-----------------------|--------------:|--------------:|------------:|-----------:|------------:|------------:|----------:|----------------:|
| Linear Regression      |   1.85764e-11 |   1.74821e-11 | 1.40881e-11 |          1 | 1.3406e-11  | 1.73425e-11 |         1 |              10 |
| Lasso Regression       |   4.8862      |   0.182896    | 0.000492315 |          1 | 0.000494379 | 0.000634491 |         1 |              10 |
| Tuned Lasso Regression | nan           | nan           | 0.00074427  |          1 | 0.00074739  | 0.000959207 |         1 |             nan |
| Ridge Regression       | 491.557       |  63.272       | 0.16431     |          1 | 0.170995    | 0.268202    |         1 |              10 |
| Tuned Ridge Regression | nan           | nan           | 0.16431     |          1 | 0.170995    | 0.268202    |         1 |             nan |

## ICE Model Rankings (Top 5)

| Unnamed: 0              |   cv_mae_mean |    cv_mae_std |   train_mae |   train_r2 |     test_mae |    test_rmse |   test_r2 |   features_used |
|:------------------------|--------------:|--------------:|------------:|-----------:|-------------:|-------------:|----------:|----------------:|
| Linear Regression       |   1.51975e-11 |   3.07344e-12 | 1.54044e-11 |   1        |  1.68438e-11 |  2.24083e-11 |  1        |               9 |
| Lasso Regression        |   4.72571     |   0.269669    | 0.000473992 |   1        |  0.00047052  |  0.000586112 |  1        |               9 |
| Ridge Regression        | 948.347       |  82.0871      | 0.13435     |   1        |  0.116133    |  0.146179    |  1        |               9 |
| Tuned Gradient Boosting | nan           | nan           | 9.93023     |   0.999999 | 42.5385      | 94.842       |  0.999958 |             nan |
| Gradient Boosting       | 152.429       |  12.449       | 2.2579      |   1        | 43.2751      | 99.2211      |  0.999955 |               9 |

## Tuned Hyperparameters (Selected)

### EV – Linear Regression

```json
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "positive": false
}
```
### ICE – Linear Regression

```json
{
  "copy_X": true,
  "fit_intercept": true,
  "n_jobs": null,
  "positive": false
}
```
## System Specs

- python_version: 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 08:22:19) [Clang 14.0.6 ]
- numpy_version: 1.26.4
- pandas_version: 2.3.2
- sklearn_version: 1.4.1.post1
