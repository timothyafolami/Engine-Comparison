#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ve.pipeline import EnhancedPipeline
from ve.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run enhanced vehicle efficiency analysis (modular pipeline)")
    p.add_argument("--data-path", default="data/vehicle_comparison_dataset_030417.csv", help="Path to input CSV")
    p.add_argument("--output-dir", default="output", help="Directory to write results")
    p.add_argument("--rank-by", choices=["r2", "mae"], default="r2", help="Metric to rank models")
    p.add_argument("--no-tuning", action="store_true", help="Disable hyperparameter tuning (RandomizedSearch)")
    p.add_argument("--no-full-viz", action="store_true", help="Disable full dashboards and 15 individual plots (enabled by default)")
    p.add_argument("--no-fine-tune-top2", action="store_true", help="Disable fine-tuning of top 2 models (enabled by default)")
    p.add_argument("--tuning-metric", choices=["r2", "mae"], default="mae", help="Metric to optimize during fine-tuning")
    p.add_argument("--search", choices=["random", "grid"], default="random", help="Sklearn search method for non-boosting models")
    p.add_argument("--random-iter", type=int, default=50, help="RandomizedSearchCV iterations")
    p.add_argument("--optuna-trials", type=int, default=50, help="Optuna trials for boosting models")
    p.add_argument("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    # Logging first so subsequent steps are captured
    setup_logging(args.output_dir, level=args.log_level)

    pipe = EnhancedPipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        rank_by=args.rank_by,
        enable_tuning=(not args.no_tuning),
        full_viz=(not args.no_full_viz),
        fine_tune_top2=(not args.no_fine_tune_top2),
        tuning_metric=args.tuning_metric,
        search_method=args.search,
        random_iter=args.random_iter,
        optuna_trials=args.optuna_trials,
    )
    pipe.run()


if __name__ == "__main__":
    main()
