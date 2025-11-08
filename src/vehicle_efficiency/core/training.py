from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from loguru import logger


@dataclass
class TrainResults:
    results: Dict[str, dict]
    models: Dict[str, Pipeline]
    X_test: pd.DataFrame
    y_test: pd.Series


def _preprocessor_for_model(selected_features: List[str], model_name: str, scaler_type: str = "minmax") -> ColumnTransformer | str:
    """Use MinMaxScaler for linear models; passthrough for tree/boosting models."""
    linear_like = {"Linear Regression", "Ridge Regression", "Lasso Regression"}
    if model_name in linear_like:
        if scaler_type == "minmax":
            return ColumnTransformer([
                ("scaler", MinMaxScaler(), selected_features)
            ])
        else:  # fallback to PowerTransformer if specified
            return ColumnTransformer([
                ("scaler", PowerTransformer(method="yeo-johnson"), selected_features)
            ])
    # Passthrough keeps columns unchanged
    return "passthrough"


def train_models_for_vehicle_type(
    data: pd.DataFrame,
    selected_features: List[str],
    models: Dict[str, object],
    target_variable: str = "efficiency",
    search_spaces: Dict[str, dict] | None = None,
    random_search_iter: int = 30,
    scaler_type: str = "minmax",
) -> TrainResults:
    """Train/tune models for a specific dataset with selected features."""
    X = data[selected_features].fillna(data[selected_features].median())
    y = data[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results: Dict[str, dict] = {}
    trained: Dict[str, Pipeline] = {}

    for name, model in models.items():
        logger.info("Training model '{}'", name)
        preprocessor = _preprocessor_for_model(selected_features, name, scaler_type)
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        try:
            cv_scores = cross_val_score(
                pipe, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
            )

            best_pipe = pipe
            if search_spaces and name in search_spaces:
                rnd = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=search_spaces[name],
                    n_iter=random_search_iter,
                    scoring="neg_mean_absolute_error",
                    cv=cv,
                    n_jobs=-1,
                    random_state=42,
                    verbose=0,
                )
                rnd.fit(X_train, y_train)
                best_pipe = rnd.best_estimator_

            best_pipe.fit(X_train, y_train)
            y_pred_tr = best_pipe.predict(X_train)
            y_pred_te = best_pipe.predict(X_test)

            res = {
                "cv_mae_mean": -cv_scores.mean(),
                "cv_mae_std": cv_scores.std(),
                "train_mae": mean_absolute_error(y_train, y_pred_tr),
                "train_r2": r2_score(y_train, y_pred_tr),
                "test_mae": mean_absolute_error(y_test, y_pred_te),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_te))),
                "test_r2": r2_score(y_test, y_pred_te),
                "features_used": len(selected_features),
            }
            results[name] = res
            trained[name] = best_pipe
            logger.info("Model '{}' | CV MAE {:.2f}±{:.2f} | Test MAE {:.2f} | Test R² {:.4f}", name, res["cv_mae_mean"], res["cv_mae_std"], res["test_mae"], res["test_r2"])
        except Exception as e:
            # Skip models that fail to train for any reason
            logger.warning("Skipping '{}' due to error: {}", name, e)
            continue

    return TrainResults(results=results, models=trained, X_test=X_test, y_test=y_test)


def rankings_dataframe(results: Dict[str, dict], by: str = "r2") -> pd.DataFrame:
    df = pd.DataFrame(results).T.copy()
    if by.lower() in ("r2", "test_r2"):
        return df.sort_values("test_r2", ascending=False)
    if by.lower() in ("mae", "test_mae"):
        return df.sort_values("test_mae", ascending=True)
    # fallback
    return df.sort_values("test_r2", ascending=False)