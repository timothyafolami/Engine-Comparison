"""
Vehicle Efficiency Model Comparison Pipeline (v1.1)
===================================================

* Compares several regression models (Linear Regression, Decision Tree, Random
  Forest, Gradient Boosting, CatBoost, LightGBM, XGBoost) on a **5-fold K-Fold**
  cross-validated workflow without hyper-parameter tuning.
* Generates a CSV of Mean Absolute Error scores and persists the best pipeline.
* Fixes previous `ValueError` by replacing `StratifiedKFold` (which requires a
  categorical target) with ordinary **`KFold`**.

Usage
-----
```bash
pip install pandas numpy scikit-learn joblib  # core stack
pip install xgboost catboost lightgbm         # optional extras

python model_comparison_pipeline.py \
       --csv_path /path/to/vehicle_comparison_dataset_030417.csv \
       --output_dir ./artifacts
```

Change log
~~~~~~~~~~
* 2025-05-26 — **v1.1**: swap `StratifiedKFold` → `KFold` to support continuous
  efficiency-index target; minor doc tidy-ups.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Optional learners — import lazily so script still runs if libs absent
OPTIONAL_MODELS: dict[str, object] = {}
try:
    from catboost import CatBoostRegressor  # type: ignore

    OPTIONAL_MODELS["CatBoost"] = CatBoostRegressor(silent=True, random_state=42)
except ImportError:
    pass

try:
    from lightgbm import LGBMRegressor  # type: ignore

    OPTIONAL_MODELS["LightGBM"] = LGBMRegressor(random_state=42)
except ImportError:
    pass

try:
    from xgboost import XGBRegressor  # type: ignore

    OPTIONAL_MODELS["XGBoost"] = XGBRegressor(
        objective="reg:squarederror", random_state=42
    )
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_efficiency_index(df: pd.DataFrame) -> pd.Series:
    """Compute a weighted efficiency index (editable weights)."""

    weights = {
        "cost_per_km": 0.35,
        "co2_emissions_g_per_km": 0.30,
        "maintenance_cost_annual": 0.15,
        "energy_consumption": 0.15,
        "mileage_km": -0.05,  # higher mileage → better efficiency
    }
    missing = [c for c in weights if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for efficiency index: {missing}")

    idx = (
        df["cost_per_km"] * weights["cost_per_km"]
        + df["co2_emissions_g_per_km"] * weights["co2_emissions_g_per_km"]
        + df["maintenance_cost_annual"] * weights["maintenance_cost_annual"]
        + df["energy_consumption"] * weights["energy_consumption"]
        - df["mileage_km"] * abs(weights["mileage_km"])
    )
    return idx


def make_preprocessor(num_cols: list[str], cat_cols: list[str]):
    numeric_tf = Pipeline([("scaler", StandardScaler())])
    categoric_tf = Pipeline([("ohe", OneHotEncoder(drop="first"))])
    return ColumnTransformer([
        ("num", numeric_tf, num_cols),
        ("cat", categoric_tf, cat_cols),
    ])


def evaluate_model(name: str, model, X, y, cv):
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    scores = cross_val_score(model, X, y, scoring=scorer, cv=cv, n_jobs=-1)
    return {
        "model": name,
        "mae_mean": -scores.mean(),  # flip sign because scorer returns negative
        "mae_std": scores.std(),
    }


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def main(args):
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    # Basic cleaning — clamp negative CO₂ for EV rows
    df["co2_emissions_g_per_km"] = df["co2_emissions_g_per_km"].clip(lower=0)

    # Build target
    df["efficiency_index"] = build_efficiency_index(df)

    # Feature lists
    num_cols = [
        "energy_consumption",
        "co2_emissions_g_per_km",
        "maintenance_cost_annual",
        "cost_per_km",
        "energy_storage_capacity",
        "mileage_km",
        "acceleration_0_100_kph_sec",
        "torque_Nm",
        "lifespan_years",
    ]
    cat_cols = ["vehicle_type"]

    X = df[num_cols + cat_cols]
    y = df["efficiency_index"]

    cv = KFold(n_splits=args.cv, shuffle=True, random_state=42)

    preprocessor = make_preprocessor(num_cols, cat_cols)

    MODELS = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    MODELS.update(OPTIONAL_MODELS)

    results = []
    best_name: str | None = None
    best_score: float = np.inf
    best_estimator = None

    for name, reg in MODELS.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", reg),
        ])
        res = evaluate_model(name, pipe, X, y, cv)
        results.append(res)
        print(f"{name:<20} | MAE = {res['mae_mean']:.2f} ± {res['mae_std']:.2f}")

        if res["mae_mean"] < best_score:
            best_score = res["mae_mean"]
            best_name = name
            best_estimator = pipe

    results_df = pd.DataFrame(results).sort_values("mae_mean")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "model_comparison_results.csv"
    results_df.to_csv(results_path, index=False)
    print("\nSorted results saved to", results_path)

    # Persist best model
    if best_estimator is not None and best_name is not None:
        best_path = output_dir / f"best_model_{best_name.replace(' ', '_').lower()}.joblib"
        best_estimator.fit(X, y)
        dump(best_estimator, best_path)
        print(f"Best model ({best_name}) saved to {best_path}")

        # Feature importances when available
        if hasattr(best_estimator[-1], "feature_importances_"):
            importances = best_estimator[-1].feature_importances_
            feature_names = (
                best_estimator["prep"].named_transformers_["num"].get_feature_names_out(num_cols).tolist()
                + best_estimator["prep"].named_transformers_["cat"]
                .named_steps["ohe"]
                .get_feature_names_out(cat_cols)
                .tolist()
            )
            fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            fi_path = output_dir / "feature_importances.json"
            fi.to_json(fi_path, orient="records")
            print("Feature importances saved to", fi_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vehicle efficiency model comparison (5-fold CV, no tuning)"
    )
    parser.add_argument("--csv_path", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--output_dir", default="artifacts", help="Where to save results and model"
    )
    parser.add_argument(
        "--cv", type=int, default=5, help="Number of cross-validation folds"
    )
    main(parser.parse_args())
