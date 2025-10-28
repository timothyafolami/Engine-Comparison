from __future__ import annotations

import pandas as pd
from loguru import logger

from .engineering import ORIGINAL_FEATURES, TARGET


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