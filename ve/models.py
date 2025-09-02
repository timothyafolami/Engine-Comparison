from __future__ import annotations

from typing import Dict

import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


def get_models() -> Dict[str, object]:
    models = {
        "Linear Regression": LinearRegression(),
        # Drop random_state for Ridge/Lasso for compatibility across sklearn versions
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0, max_iter=10000),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
    }

    # Optional models
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(objective="reg:squarederror", random_state=42)
    except Exception:
        pass
    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(silent=True, random_state=42)
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(random_state=42, verbose=-1)
    except Exception:
        pass

    return models


def get_random_search_spaces(models: Dict[str, object]) -> Dict[str, dict]:
    spaces: Dict[str, dict] = {
        "Random Forest": {
            "model__n_estimators": [50, 100, 200, 400, 800, 1200],
            "model__max_depth": [None, 4, 6, 8, 12, 16, 24, 32],
            "model__min_samples_split": [2, 5, 10, 20, 50],
            "model__min_samples_leaf": [1, 2, 4, 8, 16],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 300, 600, 1000, 1500],
            "model__learning_rate": [0.3, 0.1, 0.05, 0.01, 0.005, 0.001],
            "model__max_depth": [2, 3, 4, 5, 6],
            "model__subsample": [0.6, 0.8, 1.0],
        },
        "Decision Tree": {
            "model__max_depth": [2, 3, 4, 6, 8, 12, 20, None],
            "model__min_samples_split": [2, 5, 10, 20, 50],
            "model__min_samples_leaf": [1, 2, 4, 8, 16],
        },
        "Ridge Regression": {
            "model__alpha": np.logspace(-4, 3, 20),
        },
        "Lasso Regression": {
            "model__alpha": np.logspace(-4, 3, 20),
        },
    }

    if "XGBoost" in models:
        spaces["XGBoost"] = {
            "model__n_estimators": [200, 500, 800, 1200, 1600],
            "model__learning_rate": [0.3, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005],
            "model__max_depth": [3, 4, 5, 6, 8, 10, 12],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
        }
    if "LightGBM" in models:
        spaces["LightGBM"] = {
            "model__n_estimators": [200, 500, 800, 1200, 1600],
            "model__learning_rate": [0.3, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005],
            "model__max_depth": [-1, 3, 5, 7, 10, 12],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
        }
    if "CatBoost" in models:
        spaces["CatBoost"] = {
            "model__n_estimators": [200, 500, 800, 1200, 1600],
            "model__learning_rate": [0.3, 0.1, 0.05, 0.01, 0.005, 0.001],
            "model__depth": [3, 4, 5, 6, 8],
        }

    return spaces
