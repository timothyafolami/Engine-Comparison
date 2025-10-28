# Multi-Target Vehicle Efficiency Analysis

This document describes the new multi-target analysis functionality that allows you to analyze three different target variables separately:

1. **Maintenance Cost** (`maintenance_cost_annual`)
2. **Efficiency** (`mileage_km / energy_consumption`)
3. **Mileage** (`mileage_km`)

## Key Features

- **Separate Analysis**: Each target variable is analyzed independently with its own models and results
- **Dedicated Output Directories**: Results for each target are saved in separate folders
- **Complete Pipeline**: Each target gets the full treatment including feature engineering, model training, hyperparameter tuning, and visualization
- **Comparative Results**: Easy to compare model performance across different target variables

## Usage

### Option 1: Command Line Interface

```bash
# Run multi-target analysis using CLI
python -m src.vehicle_efficiency.cli data/vehicle_comparison_dataset_030417.csv --multi-target --output-dir output_multi_target

# Run single-target analysis (default behavior)
python -m src.vehicle_efficiency.cli data/vehicle_comparison_dataset_030417.csv --output-dir output_single
```

### Option 2: Main Script with Environment Variable

```bash
# Multi-target analysis
MULTI_TARGET=true python main.py

# Single-target analysis (default)
python main.py
```

### Option 3: Test Script

```bash
# Run the dedicated test script
python run_multi_target.py
```

### Option 4: Python Code

```python
from src.vehicle_efficiency.core.pipeline import MultiTargetPipeline

# Create multi-target pipeline
pipeline = MultiTargetPipeline(
    data_path="data/vehicle_comparison_dataset_030417.csv",
    base_output_dir="output_multi_target",
    rank_by="r2",
    enable_tuning=True,
    enable_viz=True,
    enable_fine_tuning=True,
)

# Run analysis for all targets
artifacts = pipeline.run()
```

## Output Structure

When running multi-target analysis, the output directory will contain: