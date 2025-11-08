# Final Modeling Report

## Best Models

- EV: CatBoost (Test R²: 0.7205)

- ICE: CatBoost (Test R²: 0.4856)

## EV Model Rankings (Top 5)

| Unnamed: 0             |   cv_mae_mean |   cv_mae_std |   train_mae |   train_r2 |   test_mae |   test_rmse |   test_r2 |   features_used |
|:-----------------------|--------------:|-------------:|------------:|-----------:|-----------:|------------:|----------:|----------------:|
| CatBoost               |       14018.5 |      573.998 |     11609.7 |   0.75976  |    12940.2 |     16185.9 |  0.720478 |               7 |
| Gradient Boosting      |       13845.5 |      586.74  |     11835.8 |   0.749946 |    13083.4 |     16211.6 |  0.719588 |               7 |
| Ridge Regression       |       13451.3 |      406.89  |     13170.2 |   0.68375  |    13089   |     16245.3 |  0.718421 |               7 |
| Tuned Lasso Regression |         nan   |      nan     |     13170.2 |   0.68375  |    13089   |     16245.3 |  0.718421 |             nan |
| Lasso Regression       |       13332.6 |      302.628 |     13170.2 |   0.68375  |    13089   |     16245.3 |  0.718421 |               7 |

## ICE Model Rankings (Top 5)

| Unnamed: 0        |   cv_mae_mean |   cv_mae_std |   train_mae |   train_r2 |   test_mae |   test_rmse |   test_r2 |   features_used |
|:------------------|--------------:|-------------:|------------:|-----------:|-----------:|------------:|----------:|----------------:|
| CatBoost          |       21138.5 |     1085.92  |     17910.7 |   0.643188 |    21018.1 |     26693.5 |  0.485566 |               6 |
| Gradient Boosting |       20774.3 |     1008.96  |     18334.5 |   0.628463 |    21047.7 |     26733.2 |  0.484035 |               6 |
| Random Forest     |       20885.6 |      866.685 |     18648.2 |   0.618291 |    21266.4 |     26806.6 |  0.481199 |               6 |
| XGBoost           |       22650.5 |     1222.67  |     17533.2 |   0.654609 |    21373.7 |     27128.4 |  0.468669 |               6 |
| Decision Tree     |       25815.6 |      757.95  |     19769.9 |   0.570158 |    21820.5 |     27300.9 |  0.461888 |               6 |

## Tuned Hyperparameters (Selected)

### EV – CatBoost

```json
{
  "loss_function": "RMSE",
  "silent": true,
  "random_state": 42,
  "n_estimators": 1200,
  "learning_rate": 0.01,
  "depth": 3
}
```
### ICE – CatBoost

```json
{
  "loss_function": "RMSE",
  "silent": true,
  "random_state": 42,
  "n_estimators": 1200,
  "learning_rate": 0.01,
  "depth": 3
}
```
## System Specs

- python_version: 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 08:22:19) [Clang 14.0.6 ]
- numpy_version: 1.26.4
- pandas_version: 2.3.2
- sklearn_version: 1.4.1.post1
