"""
Service layer for ML model operations.
"""
import os
import joblib
import logging
from typing import Dict, Any
import numpy as np
"""
This module handles:
- Loading and caching the trained model
- Making predictions
- Model validation
- Feature preprocessing (if needed)
- Error handling for model operations
"""
import os
import joblib
from typing import Optional, Dict, Any
from pathlib import Path


class ModelService:
    """
    Service for managing ML models
    """
    
    def __init__(self):
        """Initialize the model service"""
        self.models_cache: Dict[str, Any] = {}
        self.models_path = self._get_models_path()
        self.data_path = self._get_data_path()
    
    def _get_models_path(self) -> Path:
        """
        Get the path to the models directory
        
        Returns:
            Path to models directory
        """
        # Default to models/ in project root
        return Path(__file__).parent.parent.parent / "models"
    
    def _get_data_path(self) -> Path:
        """
        Get the path to the data directory (where RF model might be)
        
        Returns:
            Path to data directory
        """
        # Default to data/ in project root
        return Path(__file__).parent.parent.parent / "data"
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """
        Load a model from disk (with caching)
        Searches in both models/ and data/ directories
        
        Args:
            model_name: Name of the model file (e.g., 'random_forest_model.pkl')
            
        Returns:
            Loaded model or None if not found
        """
        # Check cache first
        if model_name in self.models_cache:
            return self.models_cache[model_name]
        
        # Try to load from models/ first, then data/
        possible_paths = [
            self.models_path / model_name,
            self.data_path / model_name
        ]
        
        for model_file in possible_paths:
            if model_file.exists():
                try:
                    model = joblib.load(model_file)
                    self.models_cache[model_name] = model
                    return model
                except Exception as e:
                    print(f"Error loading model {model_name} from {model_file}: {e}")
                    continue
        
        return None
    
    def get_available_models(self) -> list:
        """
        Get list of available model files
        Searches in both models/ and data/ directories
        
        Returns:
            List of model filenames
        """
        # Exclude patterns (files that are NOT models)
        exclude_patterns = ["params", "results", "scaler", "best_params", "_results", "_scaler"]
        
        # Find all .pkl files that are models in both directories
        model_files = []
        search_paths = []
        
        if self.models_path.exists():
            search_paths.append(self.models_path)
        if self.data_path.exists():
            search_paths.append(self.data_path)
        
        for search_path in search_paths:
            for f in search_path.glob("*.pkl"):
                name_lower = f.name.lower()
                # Skip if contains exclude patterns
                if any(pattern in name_lower for pattern in exclude_patterns):
                    continue
                
                # Include if:
                # 1. Contains "model" in name, OR
                # 2. Starts with "rf_" and is not params/results (Random Forest), OR
                # 3. Starts with "random_forest", OR
                # 4. Matches common model naming patterns
                if ("model" in name_lower or 
                    (name_lower.startswith("rf_") and not any(ex in name_lower for ex in exclude_patterns)) or
                    name_lower.startswith("random_forest")):
                    if f.name not in model_files:
                        model_files.append(f.name)
        
        # Check for Random Forest in data/ directory
        rf_model_names = ["random_forest_model.pkl", "rf_model.pkl"]
        for rf_name in rf_model_names:
            if (self.data_path / rf_name).exists() and rf_name not in model_files:
                model_files.append(rf_name)
            if (self.models_path / rf_name).exists() and rf_name not in model_files:
                model_files.append(rf_name)
        
        return sorted(model_files)


# Global instance
model_service = ModelService()
