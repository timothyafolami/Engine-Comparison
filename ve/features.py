from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger


ORIGINAL_FEATURES = [
    "co2_emissions_g_per_km",
    "cost_per_km",
    "energy_storage_capacity",
    "acceleration_0_100_kph_sec",
    "torque_Nm",
    "lifespan_years",
    "maintenance_cost_annual",
]

TARGET = "efficiency"


def engineer_features(df: pd.DataFrame, vehicle_type_label: str) -> Tuple[pd.DataFrame, List[str]]:
    """Create engineered features; ensure no direct leakage from components of target.

    Returns a tuple of (engineered_df, engineered_feature_names).
    """
    data = df.copy()

    logger.debug("Engineering features for '{}' cohort", vehicle_type_label)
    # Ratios and performance metrics
    data["power_efficiency"] = data["torque_Nm"] / data["acceleration_0_100_kph_sec"]
    data["storage_per_torque"] = data["energy_storage_capacity"] / data["torque_Nm"]
    data["cost_efficiency"] = data["cost_per_km"] / data["torque_Nm"]

    # Maintenance and lifespan ratios
    data["maintenance_per_year"] = data["maintenance_cost_annual"] / data["lifespan_years"]
    data["maintenance_per_torque"] = data["maintenance_cost_annual"] / data["torque_Nm"]
    data["lifespan_torque_ratio"] = data["lifespan_years"] * data["torque_Nm"]

    # Environmental efficiency
    if vehicle_type_label.upper().startswith("ELECTRIC") or vehicle_type_label == "EV":
        data["eco_efficiency"] = 1.0 / (data["co2_emissions_g_per_km"] + 1.0)
        data["green_performance"] = data["torque_Nm"] / (data["co2_emissions_g_per_km"] + 1.0)
    else:
        data["emission_intensity"] = data["co2_emissions_g_per_km"] / data["torque_Nm"]
        data["emission_per_storage"] = (
            data["co2_emissions_g_per_km"] / data["energy_storage_capacity"]
        )

    # Categoricals → codes
    data["torque_category"] = pd.cut(
        data["torque_Nm"], bins=5, labels=["Low", "Medium-Low", "Medium", "Medium-High", "High"]
    )
    data["acceleration_category"] = pd.cut(
        data["acceleration_0_100_kph_sec"],
        bins=5,
        labels=["Very Fast", "Fast", "Medium", "Slow", "Very Slow"],
    )
    data["torque_category_num"] = data["torque_category"].cat.codes
    data["acceleration_category_num"] = data["acceleration_category"].cat.codes
    data.drop(columns=["torque_category", "acceleration_category"], inplace=True)

    # Polynomial & interactions
    data["torque_squared"] = data["torque_Nm"] ** 2
    data["acceleration_squared"] = data["acceleration_0_100_kph_sec"] ** 2
    data["cost_squared"] = data["cost_per_km"] ** 2
    data["torque_x_lifespan"] = data["torque_Nm"] * data["lifespan_years"]
    data["cost_x_maintenance"] = data["cost_per_km"] * data["maintenance_cost_annual"]
    data["storage_x_lifespan"] = data["energy_storage_capacity"] * data["lifespan_years"]

    # Logs and normalized
    data["log_maintenance"] = np.log1p(data["maintenance_cost_annual"])
    data["log_torque"] = np.log1p(data["torque_Nm"])
    data["log_storage"] = np.log1p(data["energy_storage_capacity"])
    data["normalized_torque"] = data["torque_Nm"] / data["torque_Nm"].max()
    data["normalized_cost"] = data["cost_per_km"] / data["cost_per_km"].max()
    data["normalized_maintenance"] = (
        data["maintenance_cost_annual"] / data["maintenance_cost_annual"].max()
    )

    # Ensure no leakage: remove target components if present
    data = data.drop(columns=["mileage_km", "energy_consumption"], errors="ignore")

    excluded = set(ORIGINAL_FEATURES + [TARGET])
    engineered = [c for c in data.columns if c not in excluded]
    logger.info("Engineered {} features (excluding originals & target)", len(engineered))
    return data, engineered


def select_features(
    data: pd.DataFrame,
    vehicle_type_label: str,
    corr_threshold_ev: float = 0.02,
    corr_threshold_ice: float = 0.03,
    variance_threshold: float = 0.01,
    max_features: int = 20,
) -> list[str]:
    """Select features via correlation, variance, and multicollinearity filtering."""
    excluded = ["mileage_km", "energy_consumption", TARGET]
    avail = [c for c in data.columns if c not in excluded]

    # Correlation to target
    logger.debug("Selecting features for '{}' cohort | available={} ", vehicle_type_label, len(avail))
    corrs = data[avail + [TARGET]].corr()[TARGET].abs().sort_values(ascending=False)
    feature_corrs = corrs.drop(TARGET)
    threshold = corr_threshold_ev if vehicle_type_label.upper().startswith("ELECTRIC") else corr_threshold_ice
    sig = feature_corrs[feature_corrs > threshold].index.tolist()

    # Variance filter
    from sklearn.feature_selection import VarianceThreshold

    vt = VarianceThreshold(threshold=variance_threshold)
    Xtemp = data[avail].fillna(data[avail].median())
    mask = vt.fit(Xtemp).get_support()
    high_var = [f for f, m in zip(avail, mask) if m]

    # Candidates must pass both
    candidates = list(set(sig) & set(high_var))

    # Greedy multicollinearity pruning
    selected: list[str] = []
    if candidates:
        corr_mat = data[candidates].corr().abs()
        sorted_cands = feature_corrs[candidates].sort_values(ascending=False).index.tolist()
        for feat in sorted_cands:
            if not selected:
                selected.append(feat)
            else:
                max_corr = corr_mat.loc[feat, selected].max()
                if max_corr < 0.85:
                    selected.append(feat)
            if len(selected) >= max_features:
                break

    if not selected:
        selected = [f for f in ORIGINAL_FEATURES if f not in excluded]
    logger.info("Selected {} features for '{}' cohort", len(selected), vehicle_type_label)
    return selected
