from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger

from ..models.model_registry import get_models
from .training import rankings_dataframe


LINEAR_MODELS = {"Linear Regression", "Ridge Regression", "Lasso Regression"}
BOOSTING_MODELS = {"Gradient Boosting", "XGBoost", "LightGBM", "CatBoost"}


def _preprocessor_for_model(selected_features: List[str], model_name: str, scaler_type: str = "minmax") -> ColumnTransformer | str:
    if model_name in LINEAR_MODELS:
        if scaler_type == "minmax":
            return ColumnTransformer([
                ("scaler", MinMaxScaler(), selected_features)
            ])
        else:  # fallback to PowerTransformer if specified
            return ColumnTransformer([
                ("scaler", PowerTransformer(method="yeo-johnson"), selected_features)
            ])
    return "passthrough"


@dataclass
class TunedModel:
    name: str
    estimator: Pipeline
    best_params: dict
    metrics: dict


def _cv_scores(pipe: Pipeline, X, y, cv, scoring: str) -> Tuple[float, float]:
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    # For neg MAE, flip sign to be positive magnitude
    if scoring.startswith("neg_"):
        scores = -scores
    return float(scores.mean()), float(scores.std())


def _sklearn_search(
    name: str,
    estimator,
    X_train,
    y_train,
    selected_features: List[str],
    method: str,
    scoring: str,
    cv,
    random_iter: int = 50,
) -> Pipeline:
    pipe = Pipeline([
        ("preprocessor", _preprocessor_for_model(selected_features, name)),
        ("model", estimator),
    ])

    # Parameter grids/distributions
    if name == "Ridge Regression":
        grid = {"model__alpha": np.logspace(-4, 3, 40)}
    elif name == "Lasso Regression":
        grid = {
            "model__alpha": np.logspace(-4, 3, 40),
            "model__max_iter": [2000, 5000, 10000],
            "model__tol": [1e-3, 1e-4],
        }
    elif name == "Random Forest":
        grid = {
            "model__n_estimators": [100, 200, 400, 800, 1200],
            "model__max_depth": [None, 4, 6, 8, 12, 16, 24, 32],
            "model__min_samples_split": [2, 5, 10, 20, 50],
            "model__min_samples_leaf": [1, 2, 4, 8, 16],
        }
    elif name == "Decision Tree":
        grid = {
            "model__max_depth": [2, 3, 4, 6, 8, 12, 20, None],
            "model__min_samples_split": [2, 5, 10, 20, 50],
            "model__min_samples_leaf": [1, 2, 4, 8, 16],
        }
    elif name == "Gradient Boosting":
        # If not using Optuna for some reason, allow sklearn search
        grid = {
            "model__n_estimators": [100, 300, 600, 1000, 1500],
            "model__learning_rate": [0.3, 0.1, 0.05, 0.01, 0.005, 0.001],
            "model__max_depth": [2, 3, 4, 5, 6],
            "model__subsample": [0.6, 0.8, 1.0],
        }
    else:
        # Default: no tuning
        pipe.fit(X_train, y_train)
        return pipe

    if method == "grid":
        search = GridSearchCV(pipe, param_grid=grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=0)
    else:
        search = RandomizedSearchCV(pipe, param_distributions=grid, n_iter=random_iter, scoring=scoring, cv=cv, n_jobs=-1, random_state=42, verbose=0)

    search.fit(X_train, y_train)
    return search.best_estimator_


def _optuna_search(
    name: str,
    estimator,
    X_train,
    y_train,
    selected_features: List[str],
    scoring: str,
    cv,
    n_trials: int = 50,
):
    try:
        import optuna
    except Exception as e:
        return None

    def objective(trial: "optuna.trial.Trial") -> float:
        params = {}
        if name == "XGBoost":
            # Lazy import to avoid hard dependency
            from xgboost import XGBRegressor
            params.update(
                n_estimators=trial.suggest_int("n_estimators", 200, 1600, step=100),
                max_depth=trial.suggest_int("max_depth", 3, 12),
                learning_rate=trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                objective="reg:squarederror",
                random_state=42,
            )
            model = XGBRegressor(**params)
        elif name == "LightGBM":
            from lightgbm import LGBMRegressor
            params.update(
                n_estimators=trial.suggest_int("n_estimators", 200, 1600, step=100),
                max_depth=trial.suggest_int("max_depth", 3, 12),
                learning_rate=trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                random_state=42,
            )
            model = LGBMRegressor(**params)
        elif name == "CatBoost":
            from catboost import CatBoostRegressor
            params.update(
                n_estimators=trial.suggest_int("n_estimators", 200, 1600, step=100),
                depth=trial.suggest_int("depth", 3, 8),
                learning_rate=trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                silent=True,
                random_state=42,
            )
            model = CatBoostRegressor(**params)
        elif name == "Gradient Boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            params.update(
                n_estimators=trial.suggest_int("n_estimators", 100, 1500, step=100),
                max_depth=trial.suggest_int("max_depth", 2, 6),
                learning_rate=trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                random_state=42,
            )
            model = GradientBoostingRegressor(**params)
        else:
            return 1e9  # shouldn't happen

        pipe = Pipeline([
            ("preprocessor", _preprocessor_for_model(selected_features, name)),
            ("model", model),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        score = scores.mean()
        # For neg MAE, Optuna minimizes objective; convert to positive MAE
        if scoring.startswith("neg_"):
            score = -score
            return float(score)
        # For R2, maximize; convert to negative to minimize
        return float(-score)

    direction = "minimize"
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    # Rebuild model with best params
    if name == "XGBoost":
        from xgboost import XGBRegressor
        model = XGBRegressor(objective="reg:squarederror", random_state=42, **best_params)
    elif name == "LightGBM":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(random_state=42, **best_params)
    elif name == "CatBoost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(silent=True, random_state=42, **best_params)
    elif name == "Gradient Boosting":
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(random_state=42, **best_params)
    else:
        return None

    pipe = Pipeline([
        ("preprocessor", _preprocessor_for_model(selected_features, name)),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def finetune_top_two(
    data: pd.DataFrame,
    selected_features: List[str],
    rank_by: str = "r2",
    tuning_metric: str = "mae",
    search_method: str = "random",
    random_iter: int = 50,
    optuna_trials: int = 50,
    target_variable: str = "efficiency",
) -> List[TunedModel]:
    """Train base models, pick top 2 by rank, then fine-tune them.

    - For boosting models, try Optuna (if available); else fall back to sklearn search.
    - For non-boosting, use GridSearchCV (search_method="grid") or RandomizedSearchCV (default).
    - tuning_metric: "mae" uses neg_mean_absolute_error; "r2" uses r2.
    """
    models = get_models()

    X = data[selected_features].fillna(data[selected_features].median())
    y = data[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Base training to rank
    from .training import train_models_for_vehicle_type
    logger.info("Ranking base models to select top-2 for fine-tuning | rank_by='{}'", rank_by)
    base = train_models_for_vehicle_type(data, selected_features, models, search_spaces=None)
    results = base.results
    rank_df = rankings_dataframe(results, by=("r2" if rank_by.lower() == "r2" else "mae"))
    top2 = rank_df.index.tolist()[:2]
    logger.info("Selected top-2 for tuning: {}", top2)

    scoring = "r2" if tuning_metric.lower() == "r2" else "neg_mean_absolute_error"

    tuned: List[TunedModel] = []
    for name in top2:
        est = models[name]
        best_pipe: Optional[Pipeline] = None

        if name in BOOSTING_MODELS:
            logger.info("Tuning '{}' with Optuna (if available) | metric='{}' | trials={}", name, tuning_metric, optuna_trials)
            best_pipe = _optuna_search(name, est, X_train, y_train, selected_features, scoring, cv, n_trials=optuna_trials)
            if best_pipe is None:
                # Fallback to sklearn search
                logger.info("Optuna not available or failed; falling back to sklearn '{}' search", search_method)
                best_pipe = _sklearn_search(name, est, X_train, y_train, selected_features, method=search_method, scoring=scoring, cv=cv, random_iter=random_iter)
        else:
            logger.info("Tuning '{}' with sklearn '{}' search | metric='{}'", name, search_method, tuning_metric)
            best_pipe = _sklearn_search(name, est, X_train, y_train, selected_features, method=search_method, scoring=scoring, cv=cv, random_iter=random_iter)

        # Evaluate on holdout
        y_pred_tr = best_pipe.predict(X_train)
        y_pred_te = best_pipe.predict(X_test)
        metrics = {
            "train_mae": mean_absolute_error(y_train, y_pred_tr),
            "train_r2": r2_score(y_train, y_pred_tr),
            "test_mae": mean_absolute_error(y_test, y_pred_te),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_te))),
            "test_r2": r2_score(y_test, y_pred_te),
        }
        tuned.append(TunedModel(name=name, estimator=best_pipe, best_params=getattr(best_pipe.named_steps["model"], "get_params", lambda: {})(), metrics=metrics))
        logger.info("Tuned '{}' | Test MAE {:.2f} | Test R² {:.4f}", name, metrics["test_mae"], metrics["test_r2"])

    return tuned