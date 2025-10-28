"""Vehicle Efficiency Analysis Package."""

__version__ = "0.1.0"

# Core components
from .core.pipeline import EnhancedPipeline
from .core.training import TrainResults, train_models_for_vehicle_type
from .core.tuning import TunedModel, finetune_top_two

# Data processing
from .data.preprocessing import DatasetSplit, load_and_prepare_data
from .features.engineering import engineer_features, ORIGINAL_FEATURES, TARGET
from .features.selection import select_features

# Models
from .models.model_registry import get_models, get_random_search_spaces

# Utilities
from .utils.logging import setup_logging
from .utils.reporting import RunContext, save_core_artifacts
from .utils.visualization import simple_dashboard, advanced_viz

__all__ = [
    # Core
    "EnhancedPipeline",
    "TrainResults", 
    "train_models_for_vehicle_type",
    "TunedModel", 
    "finetune_top_two",
    
    # Data
    "DatasetSplit", 
    "load_and_prepare_data",
    "engineer_features", 
    "ORIGINAL_FEATURES", 
    "TARGET",
    "select_features",
    
    # Models
    "get_models", 
    "get_random_search_spaces",
    
    # Utils
    "setup_logging",
    "RunContext", 
    "save_core_artifacts",
    "simple_dashboard", 
    "advanced_viz",
]