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

# All target variables for multi-target analysis
TARGETS = ["maintenance_cost_annual", "efficiency", "mileage_km"]


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