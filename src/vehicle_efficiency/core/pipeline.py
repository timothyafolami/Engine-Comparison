from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sys
import sklearn

from ..data.preprocessing import load_and_prepare_data, load_and_prepare_data_for_target
from ..features.engineering import ORIGINAL_FEATURES, TARGET, TARGETS, engineer_features
from ..features.selection import select_features
from ..models.model_registry import get_models, get_random_search_spaces
from .training import TrainResults, rankings_dataframe, train_models_for_vehicle_type
from .tuning import finetune_top_two
from ..utils.visualization import cv_stability_plots, simple_dashboard
from ..utils.reporting import RunContext, save_core_artifacts, write_final_report_markdown
from loguru import logger


@dataclass
class RunArtifacts:
    ev_data: pd.DataFrame
    ice_data: pd.DataFrame
    ev_engineered_data: pd.DataFrame
    ice_engineered_data: pd.DataFrame
    ev_features: List[str]
    ice_features: List[str]
    ev_results: Dict[str, dict]
    ice_results: Dict[str, dict]
    ev_models: Dict[str, object]
    ice_models: Dict[str, object]


class EnhancedPipeline:
    def __init__(
        self,
        data_path: str,
        output_dir: str = "output",
        target_variable: str = "efficiency",
        rank_by: str = "r2",
        enable_tuning: bool = True,
        enable_viz: bool = True,
        enable_fine_tuning: bool = True,
        tuning_metric: str = "mae",
        search_method: str = "random",
        random_iter: int = 50,
        optuna_trials: int = 50,
    ) -> None:
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.target_variable = target_variable
        self.rank_by = rank_by
        self.enable_tuning = enable_tuning
        self.enable_viz = enable_viz
        self.enable_fine_tuning = enable_fine_tuning
        self.tuning_metric = tuning_metric
        self.search_method = search_method
        self.random_iter = random_iter
        self.optuna_trials = optuna_trials
        self.output_dir.mkdir(exist_ok=True)

    def run(self) -> None:
        logger.info("Starting enhanced pipeline run | data_path='{}' | rank_by='{}' | enable_viz={} | enable_fine_tuning={} | enable_tuning={}", self.data_path, self.rank_by, self.enable_viz, self.enable_fine_tuning, self.enable_tuning)
        # Load and split data
        logger.info("Loading and preparing data from '{}' for target '{}'", self.data_path, self.target_variable)
        ds = load_and_prepare_data_for_target(self.data_path, self.target_variable)

        # Feature engineering per cohort
        logger.info("Engineering features for EV and ICE cohorts")
        ev_eng, engineered_features_ev = engineer_features(ds.ev, "ELECTRIC VEHICLES")
        ice_eng, engineered_features_ice = engineer_features(ds.ice, "ICE VEHICLES")
        # union of engineered names for reporting
        engineered_features = sorted(list(set(engineered_features_ev + engineered_features_ice)))

        # Feature selection
        logger.info("Selecting features per cohort")
        ev_features = select_features(ev_eng, "ELECTRIC VEHICLES")
        ice_features = select_features(ice_eng, "ICE VEHICLES")
        logger.info("Selected features | EV={} | ICE={}", len(ev_features), len(ice_features))

        # Models and spaces
        models = get_models()
        search_spaces = get_random_search_spaces(models) if self.enable_tuning else {}

        # Train EV and ICE
        logger.info("Training models for EV cohort")
        ev_train = train_models_for_vehicle_type(ev_eng, ev_features, models, TARGET, search_spaces)
        logger.info("Training models for ICE cohort")
        ice_train = train_models_for_vehicle_type(ice_eng, ice_features, models, TARGET, search_spaces)

        # Optional fine-tuning of top two and merging results
        if self.enable_fine_tuning:
            try:
                from .tuning import finetune_top_two

                def merge_tuned(train: TrainResults, data: pd.DataFrame, feats: list[str]):
                    tuned_list = finetune_top_two(
                        data,
                        feats,
                        target_variable=self.target_variable,
                        rank_by=self.rank_by,
                        tuning_metric=self.tuning_metric,
                        search_method=self.search_method,
                        random_iter=self.random_iter,
                        optuna_trials=self.optuna_trials,
                    )
                    for tm in tuned_list:
                        name = f"Tuned {tm.name}"
                        train.models[name] = tm.estimator
                        train.results[name] = tm.metrics
                logger.info("Fine-tuning top-2 for EV cohort | metric='{}'", self.tuning_metric)
                merge_tuned(ev_train, ev_eng, ev_features)
                logger.info("Fine-tuning top-2 for ICE cohort | metric='{}'", self.tuning_metric)
                merge_tuned(ice_train, ice_eng, ice_features)
            except Exception as e:
                logger.warning("Fine-tune step skipped due to error: {}", e)

        # Rankings (by metric) including tuned models if present
        ev_rank = rankings_dataframe(ev_train.results, by=self.rank_by)
        ice_rank = rankings_dataframe(ice_train.results, by=self.rank_by)
        logger.info("Best models | EV='{}' ({}={:.4f}) | ICE='{}' ({}={:.4f})",
                    ev_rank.index[0] if not ev_rank.empty else None,
                    'test_r2' if self.rank_by=='r2' else 'test_mae',
                    float(ev_rank.iloc[0]['test_r2'] if self.rank_by=='r2' else ev_rank.iloc[0]['test_mae']) if not ev_rank.empty else float('nan'),
                    ice_rank.index[0] if not ice_rank.empty else None,
                    'test_r2' if self.rank_by=='r2' else 'test_mae',
                    float(ice_rank.iloc[0]['test_r2'] if self.rank_by=='r2' else ice_rank.iloc[0]['test_mae']) if not ice_rank.empty else float('nan'))

        # Save params for best models
        ev_best = ev_rank.index[0] if not ev_rank.empty else None
        ice_best = ice_rank.index[0] if not ice_rank.empty else None
        self._save_best_models_and_params(ev_best, ice_best, ev_train.models, ice_train.models)

        # Save core results JSON, rankings, and final MD report
        ctx = RunContext(
            original_features=ORIGINAL_FEATURES,
            engineered_features=engineered_features,
            ev_selected_features=ev_features,
            ice_selected_features=ice_features,
            ev_results=ev_train.results,
            ice_results=ice_train.results,
            best_ev_model=ev_best or "",
            best_ice_model=ice_best or "",
            system_specs={
                "python_version": sys.version,
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "sklearn_version": sklearn.__version__,
            },
        )
        save_core_artifacts(ctx, ev_rank, ice_rank, self.output_dir)
        write_final_report_markdown(self.output_dir)
        logger.info("Final report written to {}/final_modeling_report.md", self.output_dir)

        # Lightweight figures
        simple_dashboard(ds.ev, ds.ice, ev_train.results, ice_train.results, str(self.output_dir / "enhanced_efficiency_dashboard.png"))
        cv_stability_plots(ev_train.results, ice_train.results, out_dir=str(self.output_dir))
        
        # Advanced visuals on demand
        if self.enable_viz:
            from ..utils.visualization import advanced_viz
            logger.info("Generating advanced visualizations (individual plots and dashboards)")
            advanced_viz(
                ds.ev,
                ds.ice,
                ev_train.results,
                ice_train.results,
                ev_train.models,
                ice_train.models,
                ev_eng,
                ice_eng,
                ORIGINAL_FEATURES,
                ev_features,
                ice_features,
                output_dir=str(self.output_dir),
            )

    def _save_best_models_and_params(
        self,
        best_ev: str | None,
        best_ice: str | None,
        ev_models: Dict[str, object],
        ice_models: Dict[str, object],
    ) -> None:
        params_ev: Dict[str, dict] = {}
        params_ice: Dict[str, dict] = {}

        if best_ev and best_ev in ev_models:
            joblib.dump(ev_models[best_ev], self.output_dir / "best_enhanced_ev_model.joblib")
        if best_ice and best_ice in ice_models:
            joblib.dump(ice_models[best_ice], self.output_dir / "best_enhanced_ice_model.joblib")

        # Capture model parameters for all models to mirror prior outputs
        for name, pipe in ev_models.items():
            try:
                params_ev[name] = pipe.named_steps["model"].get_params()
            except Exception:
                continue
        for name, pipe in ice_models.items():
            try:
                params_ice[name] = pipe.named_steps["model"].get_params()
            except Exception:
                continue

        (self.output_dir / "ev_model_parameters.json").write_text(json.dumps(params_ev, indent=2))
        (self.output_dir / "ice_model_parameters.json").write_text(json.dumps(params_ice, indent=2))


class MultiTargetPipeline:
    """Pipeline that runs separate analysis for each target variable."""
    
    def __init__(
        self,
        data_path: str | Path,
        base_output_dir: str | Path = "output",
        rank_by: str = "r2",
        enable_tuning: bool = True,
        enable_viz: bool = True,
        enable_fine_tuning: bool = True,
        tuning_metric: str = "mae",
        search_method: str = "random",
        random_iter: int = 50,
        optuna_trials: int = 50,
    ):
        self.data_path = data_path
        self.base_output_dir = Path(base_output_dir)
        self.rank_by = rank_by
        self.enable_tuning = enable_tuning
        self.enable_viz = enable_viz
        self.enable_fine_tuning = enable_fine_tuning
        self.tuning_metric = tuning_metric
        self.search_method = search_method
        self.random_iter = random_iter
        self.optuna_trials = optuna_trials
        
    def run(self) -> Dict[str, RunArtifacts]:
        """Run analysis for all target variables separately."""
        # Define target variables to analyze
        targets = {
            "maintenance_cost": "maintenance_cost_annual",
            "efficiency": "efficiency", 
            "mileage": "mileage_km"
        }
        
        logger.info("Starting multi-target analysis for {} targets", len(targets))
        
        all_artifacts = {}
        
        for target_name, target_column in targets.items():
            logger.info("=" * 60)
            logger.info("ANALYZING TARGET: {} ({})", target_name.upper(), target_column)
            logger.info("=" * 60)
            
            # Create separate output directory for this target
            target_output_dir = self.base_output_dir / target_name
            target_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run pipeline for this specific target
            pipeline = EnhancedPipeline(
                data_path=self.data_path,
                output_dir=str(target_output_dir),
                target_variable=target_column,
                rank_by=self.rank_by,
                enable_tuning=self.enable_tuning,
                enable_viz=self.enable_viz,
                enable_fine_tuning=self.enable_fine_tuning,
                tuning_metric=self.tuning_metric,
                search_method=self.search_method,
                random_iter=self.random_iter,
                optuna_trials=self.optuna_trials,
            )
            
            artifacts = pipeline.run()
            all_artifacts[target_name] = artifacts
            
            # Display performance summary for this target
            self._display_target_summary(target_name, target_column, target_output_dir)
            
            logger.info("Completed analysis for target '{}' | output_dir='{}'", target_name, target_output_dir)
        
        # Display overall summary
        self._display_overall_summary(targets, all_artifacts)
        
        logger.info("Multi-target analysis completed successfully | {} targets analyzed", len(targets))
        return all_artifacts
    
    def _display_target_summary(self, target_name: str, target_column: str, target_output_dir: Path) -> None:
        """Display performance summary for a specific target."""
        logger.info("📊 PERFORMANCE SUMMARY FOR TARGET: {}", target_name.upper())
        
        # Check for results files
        ev_rankings_file = target_output_dir / "enhanced_ev_model_rankings.csv"
        ice_rankings_file = target_output_dir / "enhanced_ice_model_rankings.csv"
        
        if ev_rankings_file.exists():
            try:
                ev_rankings = pd.read_csv(ev_rankings_file, index_col=0)
                logger.debug("EV rankings columns: {}", list(ev_rankings.columns))
                best_ev = ev_rankings.iloc[0]
                best_ev_name = ev_rankings.index[0]
                
                # Handle different possible column names
                r2_col = 'test_r2' if 'test_r2' in ev_rankings.columns else 'r2'
                mae_col = 'test_mae' if 'test_mae' in ev_rankings.columns else 'mae'
                
                logger.info("🔋 Best EV Model: {} | R²={:.4f} | MAE={:.4f}", 
                           best_ev_name, best_ev[r2_col], best_ev[mae_col])
            except Exception as e:
                logger.error("Error reading EV rankings: {}", e)
        
        if ice_rankings_file.exists():
            try:
                ice_rankings = pd.read_csv(ice_rankings_file, index_col=0)
                logger.debug("ICE rankings columns: {}", list(ice_rankings.columns))
                best_ice = ice_rankings.iloc[0]
                best_ice_name = ice_rankings.index[0]
                
                # Handle different possible column names
                r2_col = 'test_r2' if 'test_r2' in ice_rankings.columns else 'r2'
                mae_col = 'test_mae' if 'test_mae' in ice_rankings.columns else 'mae'
                
                logger.info("⛽ Best ICE Model: {} | R²={:.4f} | MAE={:.4f}", 
                           best_ice_name, best_ice[r2_col], best_ice[mae_col])
            except Exception as e:
                logger.error("Error reading ICE rankings: {}", e)
        
        logger.info("📁 Output Directory: {}", target_output_dir)
        logger.info("")
    
    def _display_overall_summary(self, targets: dict, all_artifacts: dict) -> None:
        """Display overall summary of multi-target analysis."""
        logger.info("🎯 MULTI-TARGET ANALYSIS COMPLETE")
        logger.info("=" * 60)
        
        for target_name in targets.keys():
            target_dir = self.base_output_dir / target_name
            logger.info("📊 {}: {}", target_name.upper(), target_dir)
            
        logger.info("")
        logger.info("🔍 Each target directory contains:")
        logger.info("  • Model rankings and best models")
        logger.info("  • Feature importance and selection results") 
        logger.info("  • Comprehensive visualizations and plots")
        logger.info("  • Performance metrics and final reports")
        logger.info("  • Cross-validation stability analysis")