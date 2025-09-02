from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class DatasetSplit:
    ev: pd.DataFrame
    ice: pd.DataFrame


def compute_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["efficiency"] = df["mileage_km"] / df["energy_consumption"]
    return df


def remove_efficiency_outliers(df: pd.DataFrame) -> pd.DataFrame:
    eff = df["efficiency"]
    q1, q3 = eff.quantile(0.25), eff.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return df[(eff >= lower) & (eff <= upper)].copy()


def load_and_prepare_data(csv_path: str | Path) -> DatasetSplit:
    """Load CSV, compute efficiency, remove outliers, split EV/ICE, drop vehicle_type."""
    csv_path = Path(csv_path)
    logger.info("Loading data from '{}'", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded dataframe with shape {}", df.shape)

    # Compute efficiency
    df = compute_efficiency(df)

    # Remove obvious negative CO2 for EVs (safety clamp)
    if "co2_emissions_g_per_km" in df.columns:
        df["co2_emissions_g_per_km"] = df["co2_emissions_g_per_km"].clip(lower=0)

    # Remove outliers by efficiency
    before = len(df)
    df = remove_efficiency_outliers(df)
    after = len(df)
    logger.info("Removed outliers by efficiency | kept {} of {} rows (-{} removed)", after, before, before - after)

    # Split datasets
    ev_df = df[df["vehicle_type"] == "EV"].copy()
    ice_df = df[df["vehicle_type"] == "ICE"].copy()

    # Drop vehicle_type, not needed for per-cohort models
    for part in (ev_df, ice_df):
        if "vehicle_type" in part.columns:
            part.drop(columns=["vehicle_type"], inplace=True)

    logger.info("Split datasets | EV shape {} | ICE shape {}", ev_df.shape, ice_df.shape)
    return DatasetSplit(ev=ev_df, ice=ice_df)
