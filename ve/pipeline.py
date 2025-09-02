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

from .data import load_and_prepare_data
from .features import ORIGINAL_FEATURES, TARGET, engineer_features, select_features
from .models import get_models, get_random_search_spaces
from .training import TrainResults, rankings_dataframe, train_models_for_vehicle_type
from .viz import cv_stability_plots, simple_dashboard
from .reporting import RunContext, save_core_artifacts, write_final_report_markdown
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
        rank_by: str = "r2",
        enable_tuning: bool = True,
        full_viz: bool = True,
        fine_tune_top2: bool = True,
        tuning_metric: str = "mae",
        search_method: str = "random",
        random_iter: int = 50,
        optuna_trials: int = 50,
    ) -> None:
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.rank_by = rank_by
        self.enable_tuning = enable_tuning
        self.full_viz = full_viz
        self.fine_tune_top2 = fine_tune_top2
        self.tuning_metric = tuning_metric
        self.search_method = search_method
        self.random_iter = random_iter
        self.optuna_trials = optuna_trials
        self.output_dir.mkdir(exist_ok=True)

    def run(self) -> None:
        logger.info("Starting enhanced pipeline run | data_path='{}' | rank_by='{}' | full_viz={} | fine_tune_top2={} | enable_tuning={}", self.data_path, self.rank_by, self.full_viz, self.fine_tune_top2, self.enable_tuning)
        # Load and split data
        ds = load_and_prepare_data(self.data_path)

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
        ev_train = train_models_for_vehicle_type(ev_eng, ev_features, models, search_spaces)
        logger.info("Training models for ICE cohort")
        ice_train = train_models_for_vehicle_type(ice_eng, ice_features, models, search_spaces)

        # Optional fine-tuning of top two and merging results
        if self.fine_tune_top2:
            try:
                from .tuning import finetune_top_two

                def merge_tuned(train: TrainResults, data: pd.DataFrame, feats: list[str]):
                    tuned_list = finetune_top_two(
                        data,
                        feats,
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
        if self.full_viz:
            from .viz import advanced_viz
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
