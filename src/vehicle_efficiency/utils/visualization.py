from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


def cv_stability_plots(ev_results: Dict[str, dict], ice_results: Dict[str, dict], out_dir: str = "output") -> None:
    logger.info("Saving CV stability plots to {}", out_dir)
    common = sorted(list(set(ev_results) & set(ice_results)))
    if not common:
        print("No common models to plot for CV stability.")
        return

    x = np.arange(len(common))
    ev_means = [ev_results[m]["cv_mae_mean"] for m in common]
    ev_stds = [ev_results[m]["cv_mae_std"] for m in common]
    ice_means = [ice_results[m]["cv_mae_mean"] for m in common]
    ice_stds = [ice_results[m]["cv_mae_std"] for m in common]

    # EV
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(common)), ev_means, yerr=ev_stds, fmt="o-", color="green", capsize=5)
    plt.xticks(range(len(common)), common, rotation=45, ha="right")
    plt.ylabel("CV MAE ± Std")
    plt.title("EV Cross-Validation Stability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cv_stability_ev.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ICE
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(common)), ice_means, yerr=ice_stds, fmt="o-", color="blue", capsize=5)
    plt.xticks(range(len(common)), common, rotation=45, ha="right")
    plt.ylabel("CV MAE ± Std")
    plt.title("ICE Cross-Validation Stability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cv_stability_ice.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Combined
    plt.figure(figsize=(10, 6))
    plt.errorbar(x - 0.02, ev_means, yerr=ev_stds, fmt="o-", color="green", capsize=5, label="EV")
    plt.errorbar(x + 0.02, ice_means, yerr=ice_stds, fmt="o-", color="blue", capsize=5, label="ICE")
    plt.xticks(x, common, rotation=45, ha="right")
    plt.ylabel("CV MAE ± Std")
    plt.title("Cross-Validation Stability Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cv_stability_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def simple_dashboard(ev_data: pd.DataFrame, ice_data: pd.DataFrame, ev_results: Dict[str, dict], ice_results: Dict[str, dict], out_path: str) -> None:
    logger.info("Saving simple dashboard to {}", out_path)
    """Lightweight dashboard: distributions and R2/MAE comparisons."""
    common = sorted(list(set(ev_results) & set(ice_results)))
    ev_r2 = [ev_results[m]["test_r2"] for m in common] if common else []
    ice_r2 = [ice_results[m]["test_r2"] for m in common] if common else []
    ev_mae = [ev_results[m]["test_mae"] for m in common] if common else []
    ice_mae = [ice_results[m]["test_mae"] for m in common] if common else []

    fig = plt.figure(figsize=(16, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(ev_data["efficiency"], bins=30, alpha=0.7, color="green")
    ax1.set_title("EV Efficiency Distribution")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(ice_data["efficiency"], bins=30, alpha=0.7, color="blue")
    ax2.set_title("ICE Efficiency Distribution")
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 3)
    if common:
        x = np.arange(len(common))
        w = 0.35
        ax3.bar(x - w / 2, ev_r2, w, label="EV", color="green", alpha=0.7)
        ax3.bar(x + w / 2, ice_r2, w, label="ICE", color="blue", alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels(common, rotation=45, ha="right")
        ax3.set_title("Model R² Comparison")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "No common models", ha="center", va="center")

    ax4 = plt.subplot(2, 2, 4)
    if common:
        x = np.arange(len(common))
        w = 0.35
        ax4.bar(x - w / 2, ev_mae, w, label="EV", color="green", alpha=0.7)
        ax4.bar(x + w / 2, ice_mae, w, label="ICE", color="blue", alpha=0.7)
        ax4.set_xticks(x)
        ax4.set_xticklabels(common, rotation=45, ha="right")
        ax4.set_title("Model MAE Comparison")
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "No common models", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------- Advanced Visualization ---------------------------
def _common_models(ev_results: Dict[str, dict], ice_results: Dict[str, dict]) -> List[str]:
    common = list(set(ev_results.keys()) & set(ice_results.keys()))
    common.sort()
    return common


def _best_models(ev_results: Dict[str, dict], ice_results: Dict[str, dict]) -> tuple[str | None, str | None]:
    best_ev = max(ev_results.keys(), key=lambda k: ev_results[k]["test_r2"]) if ev_results else None
    best_ice = max(ice_results.keys(), key=lambda k: ice_results[k]["test_r2"]) if ice_results else None
    return best_ev, best_ice


def _feature_importance_plot(pipe, feature_names: List[str], title: str, out_path: str, color: str = "green") -> None:
    plt.figure(figsize=(12, 10))
    model = pipe.named_steps["model"]
    names = feature_names
    
    try:
        if hasattr(model, "feature_importances_"):
            vals = np.asarray(model.feature_importances_)
            # Ensure we don't exceed the bounds of feature names
            max_features = min(len(vals), len(names))
            if max_features == 0:
                logger.warning(f"No features available for importance plot: {title}")
                plt.close()
                return
            
            # Limit indices to valid range
            valid_idx = np.arange(max_features)
            vals = vals[:max_features]
            idx = np.argsort(vals)[-min(15, max_features):]
            
            # Safe indexing with bounds checking
            labels = [names[i] if i < len(names) else f"Feature_{i}" for i in idx]
            bars = plt.barh(range(len(idx)), vals[idx], color=color, alpha=0.8, edgecolor="black")
            plt.yticks(range(len(idx)), labels, fontsize=11)
            for bar in bars:
                w = bar.get_width()
                plt.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center")
            plt.xlabel("Feature Importance")
        elif hasattr(model, "coef_"):
            coefs = model.coef_
            if np.ndim(coefs) > 1:
                coefs = coefs.ravel()
            abs_coefs = np.abs(coefs)
            
            # Ensure we don't exceed the bounds of feature names
            max_features = min(len(abs_coefs), len(names))
            if max_features == 0:
                logger.warning(f"No features available for coefficient plot: {title}")
                plt.close()
                return
            
            # Limit indices to valid range
            abs_coefs = abs_coefs[:max_features]
            coefs = coefs[:max_features]
            idx = np.argsort(abs_coefs)[-min(15, max_features):]
            
            # Safe indexing with bounds checking
            labels = [names[i] if i < len(names) else f"Feature_{i}" for i in idx]
            colors = [color if coefs[i] >= 0 else "crimson" for i in idx]
            bars = plt.barh(range(len(idx)), abs_coefs[idx], color=colors, alpha=0.8, edgecolor="black")
            plt.yticks(range(len(idx)), labels, fontsize=11)
            for bar in bars:
                w = bar.get_width()
                plt.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center")
            plt.xlabel("|Coefficient| (sign encoded by color)")
        else:
            plt.text(0.5, 0.5, "No feature importances/coefs available", ha="center", va="center")
        
        plt.title(title, fontsize=16, fontweight="bold", pad=20)
        plt.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating feature importance plot '{title}': {e}")
        logger.error(f"Feature names length: {len(names)}, Model type: {type(model)}")
        if hasattr(model, "feature_importances_"):
            logger.error(f"Feature importances length: {len(model.feature_importances_)}")
        elif hasattr(model, "coef_"):
            coefs = model.coef_
            if np.ndim(coefs) > 1:
                coefs = coefs.ravel()
            logger.error(f"Coefficients length: {len(coefs)}")
        
        # Create a simple error plot
        plt.figure(figsize=(12, 10))
        plt.text(0.5, 0.5, f"Error creating plot:\n{str(e)}", ha="center", va="center", fontsize=12)
        plt.title(title, fontsize=16, fontweight="bold", pad=20)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()


def individual_plots(
    ev_data: pd.DataFrame,
    ice_data: pd.DataFrame,
    ev_results: Dict[str, dict],
    ice_results: Dict[str, dict],
    ev_models: Dict[str, object],
    ice_models: Dict[str, object],
    ev_engineered_data: pd.DataFrame,
    ice_engineered_data: pd.DataFrame,
    ev_features: List[str],
    ice_features: List[str],
    out_dir: str = "output/individual_plots",
) -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Saving individual plots to {}", out_dir)
    common = _common_models(ev_results, ice_results)
    best_ev, best_ice = _best_models(ev_results, ice_results)

    # 1 EV efficiency distribution
    plt.figure(figsize=(12, 8))
    plt.hist(ev_data["efficiency"], bins=40, alpha=0.7, color="green", density=True, edgecolor="black")
    plt.axvline(ev_data["efficiency"].mean(), color="darkgreen", linestyle="--", linewidth=3, label=f"Mean: {ev_data['efficiency'].mean():.0f}")
    plt.axvline(ev_data["efficiency"].median(), color="green", linestyle=":", linewidth=3, label=f"Median: {ev_data['efficiency'].median():.0f}")
    plt.title("🔋 Electric Vehicle Efficiency Distribution", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Efficiency (km per unit energy)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/01_ev_efficiency_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2 ICE efficiency distribution
    plt.figure(figsize=(12, 8))
    plt.hist(ice_data["efficiency"], bins=40, alpha=0.7, color="blue", density=True, edgecolor="black")
    plt.axvline(ice_data["efficiency"].mean(), color="darkblue", linestyle="--", linewidth=3, label=f"Mean: {ice_data['efficiency'].mean():.0f}")
    plt.axvline(ice_data["efficiency"].median(), color="blue", linestyle=":", linewidth=3, label=f"Median: {ice_data['efficiency'].median():.0f}")
    plt.title("⛽ ICE Efficiency Distribution", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Efficiency (km per unit energy)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/02_ice_efficiency_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3 EV vs ICE comparison
    plt.figure(figsize=(14, 8))
    plt.hist(ev_data["efficiency"], bins=35, alpha=0.6, color="green", density=True, label=f"EV (Mean: {ev_data['efficiency'].mean():.0f})", edgecolor="darkgreen")
    plt.hist(ice_data["efficiency"], bins=35, alpha=0.6, color="blue", density=True, label=f"ICE (Mean: {ice_data['efficiency'].mean():.0f})", edgecolor="darkblue")
    plt.title("🔋 vs ⛽ Vehicle Efficiency Comparison", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Efficiency (km per unit energy)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/03_ev_vs_ice_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4 Model R2 comparison
    if common:
        x = np.arange(len(common))
        ev_r2 = [ev_results[m]["test_r2"] for m in common]
        ice_r2 = [ice_results[m]["test_r2"] for m in common]
        plt.figure(figsize=(14, 8))
        w = 0.35
        plt.bar(x - w / 2, ev_r2, w, label="EV", color="green", alpha=0.7)
        plt.bar(x + w / 2, ice_r2, w, label="ICE", color="blue", alpha=0.7)
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        plt.xticks(x, common, rotation=45, ha="right")
        plt.title("Model R² Comparison", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/04_model_r2_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 5 MAE comparison
        ev_mae = [ev_results[m]["test_mae"] for m in common]
        ice_mae = [ice_results[m]["test_mae"] for m in common]
        plt.figure(figsize=(14, 8))
        plt.bar(x - w / 2, ev_mae, w, label="EV", color="green", alpha=0.7)
        plt.bar(x + w / 2, ice_mae, w, label="ICE", color="blue", alpha=0.7)
        plt.xticks(x, common, rotation=45, ha="right")
        plt.title("Model MAE Comparison", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/05_model_mae_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 14 CV stability comparison
        plt.figure(figsize=(14, 8))
        ev_means = [ev_results[m]["cv_mae_mean"] for m in common]
        ev_stds = [ev_results[m]["cv_mae_std"] for m in common]
        ice_means = [ice_results[m]["cv_mae_mean"] for m in common]
        ice_stds = [ice_results[m]["cv_mae_std"] for m in common]
        xpos = np.arange(len(common))
        plt.errorbar(xpos - 0.1, ev_means, yerr=ev_stds, fmt="o-", color="green", alpha=0.8, capsize=8, linewidth=3, label="EV")
        plt.errorbar(xpos + 0.1, ice_means, yerr=ice_stds, fmt="s-", color="blue", alpha=0.8, capsize=8, linewidth=3, label="ICE")
        plt.xticks(xpos, common, rotation=45, ha="right")
        plt.title("📊 Cross-Validation Stability Comparison", fontsize=16, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/14_cv_stability_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 6 EV feature importance
    if best_ev and best_ev in ev_models:
        _feature_importance_plot(ev_models[best_ev], ev_features, f"🔋 EV {best_ev} - Top 15 Features", f"{out_dir}/06_ev_feature_importance.png", color="green")
    # 7 ICE feature importance
    if best_ice and best_ice in ice_models:
        _feature_importance_plot(ice_models[best_ice], ice_features, f"⛽ ICE {best_ice} - Top 15 Features", f"{out_dir}/07_ice_feature_importance.png", color="blue")

    # 8 EV correlation with target
    if ev_features:
        ev_corr = ev_engineered_data[ev_features + ["efficiency"]].corr()["efficiency"].drop("efficiency")
        plt.figure(figsize=(14, 10))
        vals = ev_corr.sort_values()
        bars = plt.barh(range(len(vals)), vals.values, color=["green" if v >= 0 else "crimson" for v in vals.values], alpha=0.8)
        plt.yticks(range(len(vals)), vals.index)
        plt.title("🔋 EV Feature Correlation with Efficiency", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/08_ev_correlation_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 9 ICE correlation with target
    if ice_features:
        ice_corr = ice_engineered_data[ice_features + ["efficiency"]].corr()["efficiency"].drop("efficiency")
        plt.figure(figsize=(14, 10))
        vals = ice_corr.sort_values()
        bars = plt.barh(range(len(vals)), vals.values, color=["blue" if v >= 0 else "crimson" for v in vals.values], alpha=0.8)
        plt.yticks(range(len(vals)), vals.index)
        plt.title("⛽ ICE Feature Correlation with Efficiency", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/09_ice_correlation_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 10 EV correlation heatmap
    if ev_features:
        plt.figure(figsize=(12, 10))
        corr = ev_engineered_data[ev_features + ["efficiency"]].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=False, cmap="RdYlBu_r", center=0, square=True)
        plt.title("🔋 EV Feature Correlations", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/10_ev_correlation_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 11 ICE correlation heatmap
    if ice_features:
        plt.figure(figsize=(12, 10))
        corr = ice_engineered_data[ice_features + ["efficiency"]].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=False, cmap="RdYlBu_r", center=0, square=True)
        plt.title("⛽ ICE Feature Correlations", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/11_ice_correlation_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 12 EV vs ICE correlation comparison (common features)
    common_feats = list(set(ev_features) & set(ice_features))
    if common_feats:
        ev_c = ev_engineered_data[common_feats + ["efficiency"]].corr()["efficiency"].drop("efficiency")
        ice_c = ice_engineered_data[common_feats + ["efficiency"]].corr()["efficiency"].drop("efficiency")
        plt.figure(figsize=(10, 8))
        plt.scatter(ev_c, ice_c, alpha=0.7, s=100)
        plt.plot([-1, 1], [-1, 1], "r--", alpha=0.5)
        for f in common_feats:
            plt.annotate(f, (ev_c[f], ice_c[f]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        plt.xlabel("EV Correlation with Efficiency")
        plt.ylabel("ICE Correlation with Efficiency")
        plt.title("🔄 EV vs ICE Correlation Comparison", fontsize=16, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/12_correlation_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 13 Feature engineering summary
    plt.figure(figsize=(12, 8))
    cats = ["Selected EV", "Selected ICE"]
    counts = [len(ev_features), len(ice_features)]
    bars = plt.bar(cats, counts, color=["green", "blue"], alpha=0.7)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count), ha="center", va="bottom", fontweight="bold")
    plt.title("🔧 Feature Engineering Summary", fontsize=16, fontweight="bold")
    plt.ylabel("Number of Features")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/13_feature_engineering_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 15 Correlation statistics comparison
    if ev_features and ice_features:
        ev_corr_stats = ev_engineered_data[ev_features + ["efficiency"]].corr()["efficiency"].drop("efficiency")
        ice_corr_stats = ice_engineered_data[ice_features + ["efficiency"]].corr()["efficiency"].drop("efficiency")
        stats_categories = ["Mean |Corr|", "Max |Corr|", "Min |Corr|", "Std"]
        ev_stats = [ev_corr_stats.abs().mean(), ev_corr_stats.abs().max(), ev_corr_stats.abs().min(), ev_corr_stats.abs().std()]
        ice_stats = [ice_corr_stats.abs().mean(), ice_corr_stats.abs().max(), ice_corr_stats.abs().min(), ice_corr_stats.abs().std()]
        x_pos = np.arange(len(stats_categories))
        w = 0.35
        plt.figure(figsize=(12, 8))
        plt.bar(x_pos - w / 2, ev_stats, w, label="EV", color="green", alpha=0.8, edgecolor="darkgreen")
        plt.bar(x_pos + w / 2, ice_stats, w, label="ICE", color="blue", alpha=0.8, edgecolor="darkblue")
        plt.xticks(x_pos, stats_categories, rotation=45, ha="right")
        plt.title("📈 Correlation Statistics Comparison", fontsize=16, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/15_correlation_statistics.png", dpi=300, bbox_inches="tight")
        plt.close()


def correlation_analysis_dashboard(
    ev_engineered_data: pd.DataFrame,
    ice_engineered_data: pd.DataFrame,
    ev_features: List[str],
    ice_features: List[str],
    out_path: str = "output/correlation_analysis_dashboard.png",
) -> None:
    logger.info("Saving correlation analysis dashboard to {}", out_path)
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle("📊 Correlation Analysis Dashboard", fontsize=18, fontweight="bold")

    # EV heatmap
    ax1 = plt.subplot(3, 4, 1)
    if ev_features:
        corr = ev_engineered_data[ev_features + ["efficiency"]].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=False, cmap="RdYlBu_r", center=0, square=True, ax=ax1)
        ax1.set_title("🔋 EV Feature Correlations")
    else:
        ax1.text(0.5, 0.5, "No EV features", ha="center", va="center")

    # ICE heatmap
    ax2 = plt.subplot(3, 4, 2)
    if ice_features:
        corr = ice_engineered_data[ice_features + ["efficiency"]].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=False, cmap="RdYlBu_r", center=0, square=True, ax=ax2)
        ax2.set_title("⛽ ICE Feature Correlations")
    else:
        ax2.text(0.5, 0.5, "No ICE features", ha="center", va="center")

    # EV correlation bars (top 10)
    ax3 = plt.subplot(3, 4, 3)
    if ev_features:
        ev_corr = ev_engineered_data[ev_features + ["efficiency"]].corr()["efficiency"].drop("efficiency").sort_values()
        top = ev_corr.tail(min(10, len(ev_corr)))
        ax3.barh(range(len(top)), top.values, color=["green" if v >= 0 else "crimson" for v in top.values])
        ax3.set_yticks(range(len(top)))
        ax3.set_yticklabels(top.index)
        ax3.set_title("🔋 Top EV Correlations")
    else:
        ax3.text(0.5, 0.5, "No EV features", ha="center", va="center")

    # ICE correlation bars (top 10)
    ax4 = plt.subplot(3, 4, 4)
    if ice_features:
        ice_corr = ice_engineered_data[ice_features + ["efficiency"]].corr()["efficiency"].drop("efficiency").sort_values()
        top = ice_corr.tail(min(10, len(ice_corr)))
        ax4.barh(range(len(top)), top.values, color=["blue" if v >= 0 else "crimson" for v in top.values])
        ax4.set_yticks(range(len(top)))
        ax4.set_yticklabels(top.index)
        ax4.set_title("⛽ Top ICE Correlations")
    else:
        ax4.text(0.5, 0.5, "No ICE features", ha="center", va="center")

    # EV vs ICE correlation comparison
    ax5 = plt.subplot(3, 4, 5)
    common_feats = list(set(ev_features) & set(ice_features))
    if common_feats:
        ev_c = ev_engineered_data[common_feats + ["efficiency"]].corr()["efficiency"].drop("efficiency")
        ice_c = ice_engineered_data[common_feats + ["efficiency"]].corr()["efficiency"].drop("efficiency")
        ax5.scatter(ev_c, ice_c, alpha=0.7, s=100)
        ax5.plot([-1, 1], [-1, 1], "r--", alpha=0.5)
        ax5.set_xlabel("EV Correlation")
        ax5.set_ylabel("ICE Correlation")
        ax5.set_title("🔄 EV vs ICE Correlation Comparison")
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, "No common features", ha="center", va="center")

    # Strength distribution & selection impact & stats text
    ax6 = plt.subplot(3, 4, 6)
    if ev_features:
        ev_target_corr = ev_engineered_data[ev_features + ["efficiency"]].corr()["efficiency"].drop("efficiency")
        ev_strong = (ev_target_corr.abs() > 0.05).sum()
        ev_moderate = ((ev_target_corr.abs() > 0.02) & (ev_target_corr.abs() <= 0.05)).sum()
        ev_weak = (ev_target_corr.abs() <= 0.02).sum()
        cats = ["Strong (>0.05)", "Moderate (0.02-0.05)", "Weak (<0.02)"]
        ax6.bar([0, 1, 2], [ev_strong, ev_moderate, ev_weak], color="green", alpha=0.7)
        ax6.set_xticks([0, 1, 2])
        ax6.set_xticklabels(cats, rotation=30, ha="right")
        ax6.set_title("🔋 EV Correlation Strength")
    else:
        ax6.text(0.5, 0.5, "No EV features", ha="center", va="center")

    ax7 = plt.subplot(3, 4, 7)
    if ice_features:
        ice_target_corr = ice_engineered_data[ice_features + ["efficiency"]].corr()["efficiency"].drop("efficiency")
        ice_strong = (ice_target_corr.abs() > 0.05).sum()
        ice_moderate = ((ice_target_corr.abs() > 0.03) & (ice_target_corr.abs() <= 0.05)).sum()
        ice_weak = (ice_target_corr.abs() <= 0.03).sum()
        cats = ["Strong (>0.05)", "Moderate (0.03-0.05)", "Weak (<0.03)"]
        ax7.bar([0, 1, 2], [ice_strong, ice_moderate, ice_weak], color="blue", alpha=0.7)
        ax7.set_xticks([0, 1, 2])
        ax7.set_xticklabels(cats, rotation=30, ha="right")
        ax7.set_title("⛽ ICE Correlation Strength")
    else:
        ax7.text(0.5, 0.5, "No ICE features", ha="center", va="center")

    ax8 = plt.subplot(3, 4, 8)
    ax8.text(0.05, 0.95, "📋 Feature Selection thresholds: EV>0.02, ICE>0.03", transform=ax8.transAxes, va="top", fontfamily="monospace")
    ax8.set_xticks([]); ax8.set_yticks([])
    ax8.set_title("⚙️ Selection Methodology")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def detailed_correlation_matrices(
    ev_engineered_data: pd.DataFrame,
    ice_engineered_data: pd.DataFrame,
    original_features: List[str],
    ev_features: List[str],
    ice_features: List[str],
    out_path: str = "output/detailed_correlation_matrices.png",
) -> None:
    logger.info("Saving detailed correlation matrices to {}", out_path)
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("🔍 Detailed Correlation Matrix Analysis", fontsize=16, fontweight="bold")

    # EV original feature correlations
    ax1 = plt.subplot(2, 3, 1)
    orig_ev = [f for f in original_features if f in ev_engineered_data.columns]
    if orig_ev:
        ev_corr = ev_engineered_data[orig_ev + ["efficiency"]].corr()
        sns.heatmap(ev_corr, annot=False, cmap="RdYlBu_r", center=0, square=True, ax=ax1)
        ax1.set_title("🔋 EV Original Features Correlation")
    else:
        ax1.text(0.5, 0.5, "No original EV features", ha="center", va="center")

    # ICE original feature correlations
    ax2 = plt.subplot(2, 3, 2)
    orig_ice = [f for f in original_features if f in ice_engineered_data.columns]
    if orig_ice:
        ice_corr = ice_engineered_data[orig_ice + ["efficiency"]].corr()
        sns.heatmap(ice_corr, annot=False, cmap="RdYlBu_r", center=0, square=True, ax=ax2)
        ax2.set_title("⛽ ICE Original Features Correlation")
    else:
        ax2.text(0.5, 0.5, "No original ICE features", ha="center", va="center")

    # EV vs ICE correlation comparison (common original)
    ax3 = plt.subplot(2, 3, 3)
    common = list(set(orig_ev) & set(orig_ice))
    if common:
        ev_c = ev_engineered_data[common + ["efficiency"]].corr()["efficiency"].drop("efficiency")
        ice_c = ice_engineered_data[common + ["efficiency"]].corr()["efficiency"].drop("efficiency")
        ax3.scatter(ev_c, ice_c, alpha=0.7, s=100)
        ax3.plot([-1, 1], [-1, 1], "r--", alpha=0.5)
        ax3.set_xlabel("EV Corr")
        ax3.set_ylabel("ICE Corr")
        ax3.set_title("🔄 EV vs ICE Correlation (Original)")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No common original features", ha="center", va="center")

    # Summary text
    ax4 = plt.subplot(2, 3, 4)
    txt = "📊 Notes: Matrices show correlations among selected original features and target (efficiency)."
    ax4.text(0.05, 0.95, txt, transform=ax4.transAxes, va="top", fontfamily="monospace")
    ax4.set_xticks([]); ax4.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def advanced_viz(
    ev_data: pd.DataFrame,
    ice_data: pd.DataFrame,
    ev_results: Dict[str, dict],
    ice_results: Dict[str, dict],
    ev_models: Dict[str, object],
    ice_models: Dict[str, object],
    ev_engineered_data: pd.DataFrame,
    ice_engineered_data: pd.DataFrame,
    original_features: List[str],
    ev_features: List[str],
    ice_features: List[str],
    output_dir: str = "output",
) -> None:
    logger.info("Generating advanced visualizations into {}", output_dir)
    # Main dashboard (light): already created by simple_dashboard
    individual_plots(
        ev_data,
        ice_data,
        ev_results,
        ice_results,
        ev_models,
        ice_models,
        ev_engineered_data,
        ice_engineered_data,
        ev_features,
        ice_features,
        out_dir=f"{output_dir}/individual_plots",
    )
    correlation_analysis_dashboard(
        ev_engineered_data,
        ice_engineered_data,
        ev_features,
        ice_features,
        out_path=f"{output_dir}/correlation_analysis_dashboard.png",
    )
    detailed_correlation_matrices(
        ev_engineered_data,
        ice_engineered_data,
        original_features,
        ev_features,
        ice_features,
        out_path=f"{output_dir}/detailed_correlation_matrices.png",
    )