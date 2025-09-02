"""
Vehicle Efficiency Analysis - Separate EV and ICE Model Training
================================================================

This script implements separate model training for Electric Vehicles (EV) and 
Internal Combustion Engine (ICE) vehicles to predict efficiency based on 
vehicle characteristics.

Efficiency = mileage_km / energy_consumption

Features used for prediction:
- CO2 emissions
- Cost per km
- Energy storage capacity  
- Acceleration (0-100 kph)
- Torque
- Lifespan
- Maintenance cost annual

Author: AI Assistant
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Optional models with error handling
OPTIONAL_MODELS = {}
try:
    from xgboost import XGBRegressor

    OPTIONAL_MODELS["XGBoost"] = XGBRegressor(
        objective="reg:squarederror", random_state=42
    )
except ImportError:
    print("XGBoost not available")

try:
    from catboost import CatBoostRegressor

    OPTIONAL_MODELS["CatBoost"] = CatBoostRegressor(silent=True, random_state=42)
except ImportError:
    print("CatBoost not available")

try:
    from lightgbm import LGBMRegressor

    OPTIONAL_MODELS["LightGBM"] = LGBMRegressor(random_state=42, verbose=-1)
except ImportError:
    print("LightGBM not available")


class VehicleEfficiencyAnalyzer:
    def __init__(self, data_path="data/vehicle_comparison_dataset_030417.csv"):
        self.data_path = data_path
        self.features = [
            "co2_emissions_g_per_km",
            "cost_per_km",
            "energy_storage_capacity",
            "acceleration_0_100_kph_sec",
            "torque_Nm",
            "lifespan_years",
            "maintenance_cost_annual",
        ]
        self.target = "efficiency"

        # Model selection - top 4 from previous analysis
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
        }

        # Add optional models if available
        self.models.update(OPTIONAL_MODELS)

        # Results storage
        self.ev_data = None
        self.ice_data = None
        self.ev_results = {}
        self.ice_results = {}
        self.ev_models = {}
        self.ice_models = {}

        # Create output directory
        Path("output").mkdir(exist_ok=True)

    def load_and_prepare_data(self):
        """Load data and separate into EV and ICE datasets with efficiency calculation"""
        print("Loading and preparing data...")

        # Load original dataset
        df = pd.read_csv(self.data_path)
        print(f"Original dataset shape: {df.shape}")

        # Calculate efficiency for all vehicles
        df["efficiency"] = df["mileage_km"] / df["energy_consumption"]

        # Separate into EV and ICE datasets
        self.ev_data = df[df["vehicle_type"] == "EV"].copy()
        self.ice_data = df[df["vehicle_type"] == "ICE"].copy()

        # Remove vehicle_type column (no longer needed)
        self.ev_data = self.ev_data.drop("vehicle_type", axis=1)
        self.ice_data = self.ice_data.drop("vehicle_type", axis=1)

        print(f"EV dataset shape: {self.ev_data.shape}")
        print(f"ICE dataset shape: {self.ice_data.shape}")

        # Save separated datasets
        self.ev_data.to_csv("output/ev_dataset.csv", index=False)
        self.ice_data.to_csv("output/ice_dataset.csv", index=False)

        # Display basic statistics
        self.display_efficiency_stats()

    def display_efficiency_stats(self):
        """Display basic efficiency statistics for both vehicle types"""
        print("\n" + "=" * 50)
        print("EFFICIENCY STATISTICS")
        print("=" * 50)

        print(f"\nEV Efficiency Stats:")
        print(f"  Mean: {self.ev_data['efficiency'].mean():.2f}")
        print(f"  Median: {self.ev_data['efficiency'].median():.2f}")
        print(f"  Std: {self.ev_data['efficiency'].std():.2f}")
        print(f"  Min: {self.ev_data['efficiency'].min():.2f}")
        print(f"  Max: {self.ev_data['efficiency'].max():.2f}")

        print(f"\nICE Efficiency Stats:")
        print(f"  Mean: {self.ice_data['efficiency'].mean():.2f}")
        print(f"  Median: {self.ice_data['efficiency'].median():.2f}")
        print(f"  Std: {self.ice_data['efficiency'].std():.2f}")
        print(f"  Min: {self.ice_data['efficiency'].min():.2f}")
        print(f"  Max: {self.ice_data['efficiency'].max():.2f}")

    def train_models_for_vehicle_type(self, data, vehicle_type_name):
        """Train all models for a specific vehicle type"""
        print(f"\nTraining models for {vehicle_type_name}...")
        print("-" * 40)

        # Prepare features and target
        X = data[self.features]
        y = data[self.target]

        # Check for any missing values
        if X.isnull().sum().sum() > 0:
            print("Warning: Missing values detected in features")
            X = X.fillna(X.mean())

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Preprocessing pipeline
        preprocessor = ColumnTransformer([("scaler", StandardScaler(), self.features)])

        results = {}
        trained_models = {}

        # Cross-validation setup
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        for model_name, model in self.models.items():
            print(f"  Training {model_name}...")

            # Create pipeline
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

            try:
                # Cross-validation
                cv_scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                )

                # Train on full training set
                pipeline.fit(X_train, y_train)

                # Test set evaluation
                y_pred = pipeline.predict(X_test)

                # Calculate metrics
                test_mae = mean_absolute_error(y_test, y_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                test_r2 = r2_score(y_test, y_pred)

                # Store results
                results[model_name] = {
                    "cv_mae_mean": -cv_scores.mean(),
                    "cv_mae_std": cv_scores.std(),
                    "test_mae": test_mae,
                    "test_rmse": test_rmse,
                    "test_r2": test_r2,
                    "cv_scores": cv_scores.tolist(),
                }

                trained_models[model_name] = pipeline

                print(f"    CV MAE: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                print(f"    Test MAE: {test_mae:.4f}")
                print(f"    Test R²: {test_r2:.4f}")

            except Exception as e:
                print(f"    Error training {model_name}: {str(e)}")
                continue

        return results, trained_models, X_test, y_test

    def train_all_models(self):
        """Train models for both EV and ICE datasets"""
        print("\n" + "=" * 60)
        print("TRAINING MODELS FOR VEHICLE EFFICIENCY PREDICTION")
        print("=" * 60)

        # Train EV models
        self.ev_results, self.ev_models, self.ev_X_test, self.ev_y_test = (
            self.train_models_for_vehicle_type(self.ev_data, "ELECTRIC VEHICLES")
        )

        # Train ICE models
        self.ice_results, self.ice_models, self.ice_X_test, self.ice_y_test = (
            self.train_models_for_vehicle_type(self.ice_data, "ICE VEHICLES")
        )

    def rank_and_display_results(self):
        """Rank models and display top performers for each vehicle type"""
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE RANKINGS")
        print("=" * 60)

        # Convert results to DataFrames for easy sorting
        ev_df = pd.DataFrame(self.ev_results).T.sort_values("test_mae")
        ice_df = pd.DataFrame(self.ice_results).T.sort_values("test_mae")

        # Display top 3 for EVs
        print(f"\n🔋 TOP 3 MODELS FOR ELECTRIC VEHICLES:")
        print("-" * 45)
        for i, (model_name, row) in enumerate(ev_df.head(3).iterrows()):
            print(f"{i+1}. {model_name}")
            print(f"   Test MAE: {row['test_mae']:.4f}")
            print(f"   Test RMSE: {row['test_rmse']:.4f}")
            print(f"   Test R²: {row['test_r2']:.4f}")
            print()

        # Display top 3 for ICEs
        print(f"⛽ TOP 3 MODELS FOR ICE VEHICLES:")
        print("-" * 40)
        for i, (model_name, row) in enumerate(ice_df.head(3).iterrows()):
            print(f"{i+1}. {model_name}")
            print(f"   Test MAE: {row['test_mae']:.4f}")
            print(f"   Test RMSE: {row['test_rmse']:.4f}")
            print(f"   Test R²: {row['test_r2']:.4f}")
            print()

        # Save rankings
        ev_df.to_csv("output/ev_model_rankings.csv")
        ice_df.to_csv("output/ice_model_rankings.csv")

        return ev_df, ice_df

    def analyze_feature_importance(self):
        """Analyze feature importance for models that support it"""
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)

        def get_feature_importance(models, vehicle_type):
            print(f"\n🔍 FEATURE IMPORTANCE FOR {vehicle_type}:")
            print("-" * 50)

            for model_name, pipeline in models.items():
                model = pipeline.named_steps["model"]

                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    feature_importance = pd.Series(importances, index=self.features)
                    feature_importance = feature_importance.sort_values(ascending=False)

                    print(f"\n{model_name} - Top 5 Features:")
                    for i, (feature, importance) in enumerate(
                        feature_importance.head(5).items()
                    ):
                        print(f"  {i+1}. {feature}: {importance:.4f}")

                elif hasattr(model, "coef_") and model_name == "Linear Regression":
                    # For linear regression, use absolute coefficients
                    coefficients = np.abs(model.coef_)
                    feature_importance = pd.Series(coefficients, index=self.features)
                    feature_importance = feature_importance.sort_values(ascending=False)

                    print(f"\n{model_name} - Top 5 Features (|coefficients|):")
                    for i, (feature, importance) in enumerate(
                        feature_importance.head(5).items()
                    ):
                        print(f"  {i+1}. {feature}: {importance:.4f}")

        # Analyze for both vehicle types
        get_feature_importance(self.ev_models, "ELECTRIC VEHICLES")
        get_feature_importance(self.ice_models, "ICE VEHICLES")

    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # 1. Efficiency Distribution Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Vehicle Efficiency Analysis Dashboard", fontsize=16, fontweight="bold"
        )

        # Efficiency distributions
        axes[0, 0].hist(
            self.ev_data["efficiency"], bins=30, alpha=0.7, color="green", label="EV"
        )
        axes[0, 0].axvline(
            self.ev_data["efficiency"].mean(),
            color="darkgreen",
            linestyle="--",
            label=f'Mean: {self.ev_data["efficiency"].mean():.2f}',
        )
        axes[0, 0].set_title("EV Efficiency Distribution")
        axes[0, 0].set_xlabel("Efficiency (km per unit energy)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(
            self.ice_data["efficiency"], bins=30, alpha=0.7, color="blue", label="ICE"
        )
        axes[0, 1].axvline(
            self.ice_data["efficiency"].mean(),
            color="darkblue",
            linestyle="--",
            label=f'Mean: {self.ice_data["efficiency"].mean():.2f}',
        )
        axes[0, 1].set_title("ICE Efficiency Distribution")
        axes[0, 1].set_xlabel("Efficiency (km per unit energy)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 2. Model Performance Comparison
        ev_mae = [
            self.ev_results[model]["test_mae"] for model in self.ev_results.keys()
        ]
        ice_mae = [
            self.ice_results[model]["test_mae"] for model in self.ice_results.keys()
        ]
        model_names = list(self.ev_results.keys())

        x = np.arange(len(model_names))
        width = 0.35

        axes[1, 0].bar(
            x - width / 2, ev_mae, width, label="EV", color="green", alpha=0.7
        )
        axes[1, 0].bar(
            x + width / 2, ice_mae, width, label="ICE", color="blue", alpha=0.7
        )
        axes[1, 0].set_title("Model Performance Comparison (Test MAE)")
        axes[1, 0].set_xlabel("Models")
        axes[1, 0].set_ylabel("Mean Absolute Error")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names, rotation=45, ha="right")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 3. R² Score Comparison
        ev_r2 = [self.ev_results[model]["test_r2"] for model in self.ev_results.keys()]
        ice_r2 = [
            self.ice_results[model]["test_r2"] for model in self.ice_results.keys()
        ]

        axes[1, 1].bar(
            x - width / 2, ev_r2, width, label="EV", color="green", alpha=0.7
        )
        axes[1, 1].bar(
            x + width / 2, ice_r2, width, label="ICE", color="blue", alpha=0.7
        )
        axes[1, 1].set_title("Model Performance Comparison (R² Score)")
        axes[1, 1].set_xlabel("Models")
        axes[1, 1].set_ylabel("R² Score")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names, rotation=45, ha="right")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "output/efficiency_analysis_dashboard.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        # 4. Feature Correlation Heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # EV correlations
        ev_corr = self.ev_data[self.features + [self.target]].corr()
        sns.heatmap(ev_corr, annot=True, cmap="RdYlBu_r", center=0, ax=ax1)
        ax1.set_title("EV Feature Correlations")

        # ICE correlations
        ice_corr = self.ice_data[self.features + [self.target]].corr()
        sns.heatmap(ice_corr, annot=True, cmap="RdYlBu_r", center=0, ax=ax2)
        ax2.set_title("ICE Feature Correlations")

        plt.tight_layout()
        plt.savefig("output/feature_correlations.png", dpi=300, bbox_inches="tight")
        plt.show()

    def save_best_models(self, ev_rankings, ice_rankings):
        """Save the best performing models"""
        print("\nSaving best models...")

        # Get best model names
        best_ev_model_name = ev_rankings.index[0]
        best_ice_model_name = ice_rankings.index[0]

        # Save best models
        joblib.dump(
            self.ev_models[best_ev_model_name], "output/best_ev_efficiency_model.joblib"
        )
        joblib.dump(
            self.ice_models[best_ice_model_name],
            "output/best_ice_efficiency_model.joblib",
        )

        print(
            f"Best EV model ({best_ev_model_name}) saved to: output/best_ev_efficiency_model.joblib"
        )
        print(
            f"Best ICE model ({best_ice_model_name}) saved to: output/best_ice_efficiency_model.joblib"
        )

        # Save all results as JSON
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "features_used": self.features,
            "target_variable": self.target,
            "ev_results": self.ev_results,
            "ice_results": self.ice_results,
            "best_ev_model": best_ev_model_name,
            "best_ice_model": best_ice_model_name,
            "dataset_info": {
                "ev_samples": len(self.ev_data),
                "ice_samples": len(self.ice_data),
                "total_features": len(self.features),
            },
        }

        with open("output/efficiency_analysis_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    def generate_analysis_report(self, ev_rankings, ice_rankings):
        """Generate comprehensive analysis report"""
        print("\nGenerating analysis report...")

        best_ev_model = ev_rankings.index[0]
        best_ice_model = ice_rankings.index[0]

        report = f"""
VEHICLE EFFICIENCY PREDICTION ANALYSIS REPORT
==============================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
-----------------
This analysis compares machine learning models for predicting vehicle efficiency 
(mileage/energy_consumption) separately for Electric Vehicles (EV) and Internal 
Combustion Engine (ICE) vehicles.

DATASET OVERVIEW
----------------
• Total vehicles analyzed: {len(self.ev_data) + len(self.ice_data)}
• Electric Vehicles (EV): {len(self.ev_data)} samples
• ICE Vehicles: {len(self.ice_data)} samples
• Features used for prediction: {len(self.features)}

EFFICIENCY STATISTICS
---------------------
EV Efficiency:
  • Mean: {self.ev_data['efficiency'].mean():.2f} km/unit
  • Median: {self.ev_data['efficiency'].median():.2f} km/unit
  • Standard Deviation: {self.ev_data['efficiency'].std():.2f}
  • Range: {self.ev_data['efficiency'].min():.2f} - {self.ev_data['efficiency'].max():.2f}

ICE Efficiency:
  • Mean: {self.ice_data['efficiency'].mean():.2f} km/unit
  • Median: {self.ice_data['efficiency'].median():.2f} km/unit
  • Standard Deviation: {self.ice_data['efficiency'].std():.2f}
  • Range: {self.ice_data['efficiency'].min():.2f} - {self.ice_data['efficiency'].max():.2f}

FEATURES USED FOR PREDICTION
-----------------------------
{chr(10).join([f"• {feature}" for feature in self.features])}

MODEL PERFORMANCE RESULTS
--------------------------

🔋 ELECTRIC VEHICLES - TOP 3 MODELS:
"""

        for i, (model_name, row) in enumerate(ev_rankings.head(3).iterrows()):
            report += f"""
{i+1}. {model_name}
   • Test MAE: {row['test_mae']:.4f}
   • Test RMSE: {row['test_rmse']:.4f}
   • Test R²: {row['test_r2']:.4f}
   • CV MAE: {row['cv_mae_mean']:.4f} ± {row['cv_mae_std']:.4f}"""

        report += f"""

⛽ ICE VEHICLES - TOP 3 MODELS:
"""

        for i, (model_name, row) in enumerate(ice_rankings.head(3).iterrows()):
            report += f"""
{i+1}. {model_name}
   • Test MAE: {row['test_mae']:.4f}
   • Test RMSE: {row['test_rmse']:.4f}
   • Test R²: {row['test_r2']:.4f}
   • CV MAE: {row['cv_mae_mean']:.4f} ± {row['cv_mae_std']:.4f}"""

        # Compare which vehicle type is more predictable
        best_ev_r2 = ev_rankings.iloc[0]["test_r2"]
        best_ice_r2 = ice_rankings.iloc[0]["test_r2"]

        more_predictable = (
            "Electric Vehicles" if best_ev_r2 > best_ice_r2 else "ICE Vehicles"
        )

        report += f"""

KEY INSIGHTS
------------
• Best EV Model: {best_ev_model} (R² = {best_ev_r2:.4f})
• Best ICE Model: {best_ice_model} (R² = {best_ice_r2:.4f})
• More Predictable Vehicle Type: {more_predictable}
• Average EV Efficiency: {self.ev_data['efficiency'].mean():.2f} km/unit
• Average ICE Efficiency: {self.ice_data['efficiency'].mean():.2f} km/unit

EFFICIENCY COMPARISON
---------------------
"""

        if self.ev_data["efficiency"].mean() > self.ice_data["efficiency"].mean():
            report += f"• Electric Vehicles are {((self.ev_data['efficiency'].mean() / self.ice_data['efficiency'].mean() - 1) * 100):.1f}% more efficient on average\n"
        else:
            report += f"• ICE Vehicles are {((self.ice_data['efficiency'].mean() / self.ev_data['efficiency'].mean() - 1) * 100):.1f}% more efficient on average\n"

        report += f"""
RECOMMENDATIONS
---------------
• Use {best_ev_model} for predicting EV efficiency
• Use {best_ice_model} for predicting ICE efficiency
• Focus on the top-performing features identified in feature importance analysis
• Consider ensemble methods combining top 3 models for each vehicle type

FILES GENERATED
---------------
• ev_dataset.csv - Electric vehicle data with efficiency calculations
• ice_dataset.csv - ICE vehicle data with efficiency calculations
• ev_model_rankings.csv - EV model performance rankings
• ice_model_rankings.csv - ICE model performance rankings
• best_ev_efficiency_model.joblib - Best trained EV model
• best_ice_efficiency_model.joblib - Best trained ICE model
• efficiency_analysis_results.json - Complete results in JSON format
• efficiency_analysis_dashboard.png - Performance visualization
• feature_correlations.png - Feature correlation heatmaps

METHODOLOGY
-----------
• Train-test split: 80/20
• Cross-validation: 5-fold KFold
• Preprocessing: StandardScaler for numerical features
• Evaluation metrics: MAE, RMSE, R²
• Random state: 42 (for reproducibility)

END OF REPORT
=============
"""

        with open("output/efficiency_analysis_report.txt", "w") as f:
            f.write(report)

        print("Analysis report saved to: output/efficiency_analysis_report.txt")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚗 VEHICLE EFFICIENCY ANALYSIS PIPELINE")
        print("=" * 60)

        # Step 1: Load and prepare data
        self.load_and_prepare_data()

        # Step 2: Train all models
        self.train_all_models()

        # Step 3: Rank and display results
        ev_rankings, ice_rankings = self.rank_and_display_results()

        # Step 4: Feature importance analysis
        self.analyze_feature_importance()

        # Step 5: Create visualizations
        self.create_visualizations()

        # Step 6: Save best models
        self.save_best_models(ev_rankings, ice_rankings)

        # Step 7: Generate comprehensive report
        self.generate_analysis_report(ev_rankings, ice_rankings)

        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Check the 'output' directory for all generated files:")
        print("• Model rankings and performance metrics")
        print("• Best trained models (joblib files)")
        print("• Visualizations (PNG files)")
        print("• Comprehensive analysis report")
        print("• Separated datasets (CSV files)")


def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = VehicleEfficiencyAnalyzer()

    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
