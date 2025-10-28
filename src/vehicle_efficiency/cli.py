#!/usr/bin/env python3
"""Command-line interface for the vehicle efficiency analysis pipeline."""

import argparse
import sys
from pathlib import Path

from loguru import logger

from .core.pipeline import EnhancedPipeline, MultiTargetPipeline
from .utils.logging import setup_logging


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Vehicle Efficiency Analysis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the input CSV data file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output",
        help="Output directory for results and artifacts"
    )
    
    parser.add_argument(
        "--rank-by",
        type=str,
        choices=["r2", "mae"],
        default="r2",
        help="Metric to rank models by"
    )
    
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Disable hyperparameter tuning"
    )
    
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable advanced visualizations"
    )
    
    parser.add_argument(
        "--no-fine-tune",
        action="store_true",
        help="Disable fine-tuning of top 2 models"
    )
    
    parser.add_argument(
        "--multi-target",
        action="store_true",
        help="Run analysis for all target variables (maintenance_cost, efficiency, mileage) separately"
    )
    
    parser.add_argument(
        "--tuning-metric",
        type=str,
        choices=["mae", "r2"],
        default="mae",
        help="Metric for hyperparameter tuning"
    )
    
    parser.add_argument(
        "--search-method",
        type=str,
        choices=["random", "optuna"],
        default="random",
        help="Search method for hyperparameter tuning"
    )
    
    parser.add_argument(
        "--random-iter",
        type=int,
        default=50,
        help="Number of iterations for random search"
    )
    
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=50,
        help="Number of trials for Optuna optimization"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Input file does not exist: {data_path}")
        sys.exit(1)
    
    if not data_path.suffix.lower() == '.csv':
        logger.error(f"Input file must be a CSV file: {data_path}")
        sys.exit(1)
    
    # Setup logging
    output_dir = Path(args.output_dir)
    setup_logging(output_dir, level=args.log_level)
    
    logger.info("Starting Vehicle Efficiency Analysis Pipeline")
    logger.info(f"Input data: {data_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Multi-target analysis: {args.multi_target}")
    logger.info(f"Ranking metric: {args.rank_by}")
    logger.info(f"Tuning enabled: {not args.no_tuning}")
    logger.info(f"Advanced visualizations: {not args.no_viz}")
    logger.info(f"Fine-tuning enabled: {not args.no_fine_tune}")
    
    try:
        if args.multi_target:
            # Create and run multi-target pipeline
            pipeline = MultiTargetPipeline(
                data_path=str(data_path),
                base_output_dir=str(output_dir),
                rank_by=args.rank_by,
                enable_tuning=not args.no_tuning,
                enable_viz=not args.no_viz,
                enable_fine_tuning=not args.no_fine_tune,
                tuning_metric=args.tuning_metric,
                search_method=args.search_method,
                random_iter=args.random_iter,
                optuna_trials=args.optuna_trials,
            )
        else:
            # Create and run single-target pipeline
            pipeline = EnhancedPipeline(
                data_path=str(data_path),
                output_dir=str(output_dir),
                rank_by=args.rank_by,
                enable_tuning=not args.no_tuning,
                enable_viz=not args.no_viz,
                enable_fine_tuning=not args.no_fine_tune,
                tuning_metric=args.tuning_metric,
                search_method=args.search_method,
                random_iter=args.random_iter,
                optuna_trials=args.optuna_trials,
            )
        
        pipeline.run()
        
        logger.success("Pipeline completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()