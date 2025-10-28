#!/usr/bin/env python3
"""
Scaler Comparison Analysis: MinMaxScaler vs PowerTransformer
This script compares the performance of different scalers on the vehicle efficiency dataset.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vehicle_efficiency.data.preprocessing import load_and_prepare_data_for_target
from vehicle_efficiency.features.engineering import engineer_features
from vehicle_efficiency.features.selection import select_features


class ScalerComparison:
    """Compare different scalers on vehicle efficiency prediction."""
    
    def __init__(self, data_path: str, target_variable: str = "efficiency"):
        self.data_path = data_path
        self.target_variable = target_variable
        self.scalers = {
            "PowerTransformer": PowerTransformer(method="yeo-johnson"),
            "MinMaxScaler": MinMaxScaler(),
            "StandardScaler": StandardScaler(),
            "NoScaling": "passthrough"
        }
        self.models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=1.0, max_iter=10000),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
        }
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data for analysis."""
        logger.info("Loading and preparing data for target: {}", self.target_variable)
        
        # Load data
        dataset_split = load_and_prepare_data_for_target(self.data_path, self.target_variable)
        
        # Engineer features for both EV and ICE
        ev_engineered = engineer_features(dataset_split.ev)
        ice_engineered = engineer_features(dataset_split.ice)
        
        # Select features
        ev_features = select_features(ev_engineered, self.target_variable)
        ice_features = select_features(ice_engineered, self.target_variable)
        
        self.data = {
            "EV": {
                "data": ev_engineered,
                "features": ev_features,
                "X": ev_engineered[ev_features],
                "y": ev_engineered[self.target_variable]
            },
            "ICE": {
                "data": ice_engineered,
                "features": ice_features,
                "X": ice_engineered[ice_features],
                "y": ice_engineered[self.target_variable]
            }
        }
        
        logger.info("Data prepared - EV: {} samples, ICE: {} samples", 
                   len(self.data["EV"]["data"]), len(self.data["ICE"]["data"]))
    
    def create_preprocessor(self, features, scaler_name, model_name):
        """Create preprocessor based on scaler and model type."""
        # Only apply scaling to linear models
        linear_models = {"Linear Regression", "Ridge Regression", "Lasso Regression"}
        
        if model_name in linear_models and scaler_name != "NoScaling":
            scaler = self.scalers[scaler_name]
            return ColumnTransformer([("scaler", scaler, features)])
        else:
            return "passthrough"
    
    def evaluate_combination(self, vehicle_type, scaler_name, model_name):
        """Evaluate a specific scaler-model combination."""
        data_info = self.data[vehicle_type]
        X, y = data_info["X"], data_info["y"]
        features = data_info["features"]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create pipeline
        preprocessor = self.create_preprocessor(features, scaler_name, model_name)
        model = self.models[model_name]
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        # Cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, 
            cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        
        # Train on full training set and evaluate on test set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)
        cv_mae_mean = -cv_scores.mean()
        cv_mae_std = cv_scores.std()
        
        return {
            "vehicle_type": vehicle_type,
            "scaler": scaler_name,
            "model": model_name,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "cv_mae_mean": cv_mae_mean,
            "cv_mae_std": cv_mae_std,
            "n_features": len(features),
            "n_samples": len(X)
        }
    
    def run_comparison(self):
        """Run complete scaler comparison analysis."""
        logger.info("Starting scaler comparison analysis")
        
        results_list = []
        
        for vehicle_type in ["EV", "ICE"]:
            logger.info("Analyzing {} vehicles", vehicle_type)
            
            for scaler_name in self.scalers.keys():
                for model_name in self.models.keys():
                    logger.info("  Testing {} + {}", scaler_name, model_name)
                    
                    try:
                        result = self.evaluate_combination(vehicle_type, scaler_name, model_name)
                        results_list.append(result)
                    except Exception as e:
                        logger.warning("Failed {}-{}-{}: {}", vehicle_type, scaler_name, model_name, e)
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(results_list)
        logger.info("Comparison completed with {} results", len(self.results_df))
        
        return self.results_df
    
    def analyze_results(self):
        """Analyze and summarize the comparison results."""
        if self.results_df is None or self.results_df.empty:
            logger.error("No results to analyze")
            return
        
        logger.info("Analyzing scaler comparison results")
        
        # Focus on linear models where scaling matters
        linear_results = self.results_df[
            self.results_df["model"].isin(["Linear Regression", "Ridge Regression", "Lasso Regression"])
        ].copy()
        
        # Summary statistics by scaler
        scaler_summary = linear_results.groupby(["vehicle_type", "scaler"]).agg({
            "test_r2": ["mean", "std", "min", "max"],
            "test_mae": ["mean", "std", "min", "max"],
            "cv_mae_mean": ["mean", "std"]
        }).round(4)
        
        logger.info("Scaler Performance Summary (Linear Models Only):")
        print("\n" + "="*80)
        print("SCALER PERFORMANCE SUMMARY - LINEAR MODELS")
        print("="*80)
        print(scaler_summary)
        
        # Best scaler by vehicle type and metric
        best_by_r2 = linear_results.loc[linear_results.groupby("vehicle_type")["test_r2"].idxmax()]
        best_by_mae = linear_results.loc[linear_results.groupby("vehicle_type")["test_mae"].idxmin()]
        
        print("\n" + "="*60)
        print("BEST SCALERS BY METRIC")
        print("="*60)
        print("\nBest R² Performance:")
        for _, row in best_by_r2.iterrows():
            print(f"  {row['vehicle_type']}: {row['scaler']} + {row['model']} (R²={row['test_r2']:.4f})")
        
        print("\nBest MAE Performance:")
        for _, row in best_by_mae.iterrows():
            print(f"  {row['vehicle_type']}: {row['scaler']} + {row['model']} (MAE={row['test_mae']:.4f})")
        
        return {
            "scaler_summary": scaler_summary,
            "best_by_r2": best_by_r2,
            "best_by_mae": best_by_mae,
            "linear_results": linear_results
        }
    
    def create_visualizations(self, output_dir="output"):
        """Create comparison visualizations."""
        if self.results_df is None or self.results_df.empty:
            logger.error("No results to visualize")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Focus on linear models
        linear_results = self.results_df[
            self.results_df["model"].isin(["Linear Regression", "Ridge Regression", "Lasso Regression"])
        ].copy()
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Scaler Comparison Analysis - {self.target_variable.title()}", fontsize=16)
        
        # R² comparison
        sns.boxplot(data=linear_results, x="scaler", y="test_r2", hue="vehicle_type", ax=axes[0,0])
        axes[0,0].set_title("R² Score by Scaler")
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        sns.boxplot(data=linear_results, x="scaler", y="test_mae", hue="vehicle_type", ax=axes[0,1])
        axes[0,1].set_title("MAE by Scaler")
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Scaler performance heatmap - R²
        pivot_r2 = linear_results.pivot_table(
            values="test_r2", index="scaler", columns="vehicle_type", aggfunc="mean"
        )
        sns.heatmap(pivot_r2, annot=True, fmt=".4f", cmap="RdYlGn", ax=axes[1,0])
        axes[1,0].set_title("Average R² by Scaler and Vehicle Type")
        
        # Scaler performance heatmap - MAE
        pivot_mae = linear_results.pivot_table(
            values="test_mae", index="scaler", columns="vehicle_type", aggfunc="mean"
        )
        sns.heatmap(pivot_mae, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=axes[1,1])
        axes[1,1].set_title("Average MAE by Scaler and Vehicle Type")
        
        plt.tight_layout()
        plot_path = output_path / f"scaler_comparison_{self.target_variable}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info("Visualization saved to: {}", plot_path)
        
        # Save detailed results
        results_path = output_path / f"scaler_comparison_results_{self.target_variable}.csv"
        self.results_df.to_csv(results_path, index=False)
        
        detailed_comparison = linear_results.pivot_table(
            values=["test_r2", "test_mae", "cv_mae_mean"], 
            index=["vehicle_type", "model"], 
            columns="scaler", 
            aggfunc="mean"
        ).round(4)
        
        detailed_path = output_path / f"scaler_comparison_detailed_{self.target_variable}.csv"
        detailed_comparison.to_csv(detailed_path)
        
        logger.info("Results saved to: {}", results_path)
        logger.info("Detailed comparison saved to: {}", detailed_path)
        
        return {
            "plot_path": plot_path,
            "results_path": results_path,
            "detailed_path": detailed_path
        }
    
    def save_summary_report(self, analysis_results, output_dir="output"):
        """Save a summary report of the scaler comparison."""
        output_path = Path(output_dir)
        report_path = output_path / f"SCALER_COMPARISON_REPORT_{self.target_variable.upper()}.md"
        
        with open(report_path, "w") as f:
            f.write(f"# Scaler Comparison Analysis Report\n\n")
            f.write(f"**Target Variable:** {self.target_variable}\n")
            f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This analysis compares the performance of different data scalers on vehicle efficiency prediction models:\n\n")
            f.write("- **PowerTransformer (Yeo-Johnson)**: Current default scaler\n")
            f.write("- **MinMaxScaler**: Scales features to [0,1] range\n")
            f.write("- **StandardScaler**: Z-score normalization\n")
            f.write("- **No Scaling**: Raw features (baseline)\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Best performers
            best_r2 = analysis_results["best_by_r2"]
            best_mae = analysis_results["best_by_mae"]
            
            f.write("### Best Performing Scalers\n\n")
            f.write("**By R² Score:**\n")
            for _, row in best_r2.iterrows():
                f.write(f"- **{row['vehicle_type']}**: {row['scaler']} + {row['model']} (R²={row['test_r2']:.4f})\n")
            
            f.write("\n**By MAE Score:**\n")
            for _, row in best_mae.iterrows():
                f.write(f"- **{row['vehicle_type']}**: {row['scaler']} + {row['model']} (MAE={row['test_mae']:.4f})\n")
            
            f.write("\n### Performance Summary\n\n")
            
            # Calculate scaler rankings
            linear_results = analysis_results["linear_results"]
            scaler_avg = linear_results.groupby(["vehicle_type", "scaler"]).agg({
                "test_r2": "mean",
                "test_mae": "mean"
            }).round(4)
            
            f.write("**Average Performance by Scaler (Linear Models Only):**\n\n")
            for vehicle_type in ["EV", "ICE"]:
                f.write(f"**{vehicle_type} Vehicles:**\n")
                vehicle_data = scaler_avg.loc[vehicle_type]
                
                # Sort by R²
                r2_ranking = vehicle_data.sort_values("test_r2", ascending=False)
                f.write("- R² Ranking:\n")
                for i, (scaler, metrics) in enumerate(r2_ranking.iterrows(), 1):
                    f.write(f"  {i}. {scaler}: R²={metrics['test_r2']:.4f}, MAE={metrics['test_mae']:.4f}\n")
                f.write("\n")
            
            f.write("## Recommendations\n\n")
            
            # Determine best overall scaler
            overall_best = linear_results.groupby("scaler").agg({
                "test_r2": "mean",
                "test_mae": "mean"
            }).round(4)
            
            best_scaler_r2 = overall_best["test_r2"].idxmax()
            best_scaler_mae = overall_best["test_mae"].idxmin()
            
            f.write(f"Based on this analysis:\n\n")
            f.write(f"1. **For R² optimization**: {best_scaler_r2} shows the best average performance\n")
            f.write(f"2. **For MAE optimization**: {best_scaler_mae} shows the best average performance\n")
            
            if best_scaler_r2 == "PowerTransformer":
                f.write(f"3. **Current choice validated**: PowerTransformer remains the optimal choice\n")
            else:
                f.write(f"3. **Consider switching**: {best_scaler_r2} outperforms the current PowerTransformer\n")
            
            f.write("\n## Technical Details\n\n")
            f.write(f"- **Dataset**: {len(self.data['EV']['data'])} EV samples, {len(self.data['ICE']['data'])} ICE samples\n")
            f.write(f"- **Features**: {len(self.data['EV']['features'])} EV features, {len(self.data['ICE']['features'])} ICE features\n")
            f.write(f"- **Models tested**: Linear Regression, Ridge Regression, Lasso Regression\n")
            f.write(f"- **Validation**: 5-fold cross-validation + 80/20 train-test split\n")
            f.write(f"- **Metrics**: R², MAE, RMSE\n\n")
            
            f.write("## Files Generated\n\n")
            f.write(f"- `scaler_comparison_{self.target_variable}.png`: Visualization plots\n")
            f.write(f"- `scaler_comparison_results_{self.target_variable}.csv`: Complete results\n")
            f.write(f"- `scaler_comparison_detailed_{self.target_variable}.csv`: Detailed comparison table\n")
        
        logger.info("Summary report saved to: {}", report_path)
        return report_path


def main():
    """Run scaler comparison analysis."""
    logger.info("Starting Scaler Comparison Analysis")
    
    # Configuration
    data_path = "data/vehicle_comparison_dataset_030417.csv"
    targets = ["efficiency", "maintenance_cost_annual", "mileage_km"]
    output_dir = "output/scaler_comparison"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for target in targets:
        logger.info("="*60)
        logger.info("ANALYZING TARGET: {}", target.upper())
        logger.info("="*60)
        
        try:
            # Run comparison for this target
            comparison = ScalerComparison(data_path, target)
            comparison.load_and_prepare_data()
            results_df = comparison.run_comparison()
            analysis_results = comparison.analyze_results()
            
            # Create visualizations and save results
            file_paths = comparison.create_visualizations(output_dir)
            report_path = comparison.save_summary_report(analysis_results, output_dir)
            
            all_results[target] = {
                "results_df": results_df,
                "analysis": analysis_results,
                "files": file_paths,
                "report": report_path
            }
            
            logger.success("Completed analysis for target: {}", target)
            
        except Exception as e:
            logger.error("Failed analysis for target {}: {}", target, e)
            logger.exception("Full traceback:")
    
    # Create overall summary
    logger.info("Creating overall summary")
    create_overall_summary(all_results, output_dir)
    
    logger.success("Scaler comparison analysis completed!")
    logger.info("Results saved to: {}", output_dir)


def create_overall_summary(all_results, output_dir):
    """Create an overall summary across all targets."""
    summary_path = Path(output_dir) / "OVERALL_SCALER_COMPARISON_SUMMARY.md"
    
    with open(summary_path, "w") as f:
        f.write("# Overall Scaler Comparison Summary\n\n")
        f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary Across All Targets\n\n")
        
        for target, results in all_results.items():
            if "analysis" in results:
                f.write(f"### {target.title()}\n\n")
                
                best_r2 = results["analysis"]["best_by_r2"]
                best_mae = results["analysis"]["best_by_mae"]
                
                f.write("**Best Performers:**\n")
                for _, row in best_r2.iterrows():
                    f.write(f"- {row['vehicle_type']} (R²): {row['scaler']} (R²={row['test_r2']:.4f})\n")
                for _, row in best_mae.iterrows():
                    f.write(f"- {row['vehicle_type']} (MAE): {row['scaler']} (MAE={row['test_mae']:.4f})\n")
                f.write("\n")
        
        f.write("## Files Generated\n\n")
        for target, results in all_results.items():
            if "files" in results:
                f.write(f"### {target.title()}\n")
                for file_type, file_path in results["files"].items():
                    f.write(f"- {file_type}: `{file_path}`\n")
                f.write(f"- Report: `{results['report']}`\n\n")
    
    logger.info("Overall summary saved to: {}", summary_path)


if __name__ == "__main__":
    main()