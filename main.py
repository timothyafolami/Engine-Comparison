#!/usr/bin/env python3
"""
Main entry point for the Vehicle Efficiency Analysis Pipeline.

This script provides a simple way to run the pipeline with default settings.
For more advanced options, use the CLI: python -m vehicle_efficiency.cli
"""

import sys
import os
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vehicle_efficiency.core.pipeline import EnhancedPipeline, MultiTargetPipeline
from vehicle_efficiency.utils.logging import setup_logging
from loguru import logger


def main():
    """Run the pipeline with default settings."""
    # Default configuration
    data_path = "data/vehicle_comparison_dataset_030417.csv"
    output_dir = "output"
    
    # Check for multi-target mode
    multi_target = os.getenv("MULTI_TARGET", "").lower() in ("1", "true", "yes")
    
    # Check if data file exists
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please ensure 'data/vehicle_comparison_dataset_030417.csv' exists")
        logger.info("Or use the CLI with a custom path: python -m vehicle_efficiency.cli <path_to_data>")
        sys.exit(1)
    
    # Setup logging
    setup_logging(Path(output_dir))
    
    logger.info("🚗 Starting Vehicle Efficiency Analysis Pipeline")
    logger.info(f"📊 Data file: {data_path}")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"🎯 Multi-target analysis: {multi_target}")
    
    try:
        if multi_target:
            # Create and run multi-target pipeline
            pipeline = MultiTargetPipeline(
                data_path=data_path,
                base_output_dir=output_dir,
                rank_by="r2",
                enable_tuning=True,
                enable_viz=True,
                enable_fine_tuning=True,
                tuning_metric="mae",
                search_method="random",
                random_iter=50,
                optuna_trials=50,
            )
        else:
            # Create and run single-target pipeline
            pipeline = EnhancedPipeline(
                data_path=data_path,
                output_dir=output_dir,
                rank_by="r2",
                enable_tuning=True,
                enable_viz=True,
                enable_fine_tuning=True,
                tuning_metric="mae",
                search_method="random",
                random_iter=50,
                optuna_trials=50,
            )
        
        pipeline.run()
        
        logger.success("✅ Pipeline completed successfully!")
        if multi_target:
            logger.info(f"📈 Results saved to: {output_dir}/")
            logger.info("📋 Check the reports in each target directory:")
            logger.info(f"   - Maintenance Cost: {output_dir}/maintenance_cost/")
            logger.info(f"   - Efficiency: {output_dir}/efficiency/")
            logger.info(f"   - Mileage: {output_dir}/mileage/")
        else:
            logger.info(f"📈 Results saved to: {output_dir}/")
            logger.info(f"📋 Check the final report: {output_dir}/final_modeling_report.md")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()