#!/usr/bin/env python3
"""
Test script for multi-target vehicle efficiency analysis.
This script runs separate analyses for maintenance cost, efficiency, and mileage.
"""

from pathlib import Path
from src.vehicle_efficiency.core.pipeline import MultiTargetPipeline
from src.vehicle_efficiency.utils.logging import setup_logging
from loguru import logger

def main():
    """Run multi-target analysis."""
    # Setup paths
    data_path = "data/vehicle_comparison_dataset_030417.csv"
    output_dir = "output_multi_target"
    
    # Verify data file exists
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return
    
    # Setup logging
    setup_logging(Path(output_dir), level="INFO")
    
    logger.info("Starting multi-target vehicle efficiency analysis")
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {output_dir}")
    
    try:
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
            random_iter=30,  # Reduced for faster testing
            optuna_trials=30,
        )
        
        artifacts = pipeline.run()
        
        logger.success("Multi-target analysis completed successfully!")
        logger.info(f"Analyzed {len(artifacts)} target variables:")
        for target_name in artifacts.keys():
            logger.info(f"  - {target_name}: output/{target_name}/")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.exception("Full traceback:")

if __name__ == "__main__":
    main()