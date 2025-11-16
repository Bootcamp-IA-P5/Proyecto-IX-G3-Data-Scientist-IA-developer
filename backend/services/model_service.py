"""
Service layer for ML model operations.

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
    
    def _get_models_path(self) -> Path:
        """
        Get the path to the models directory
        
        Returns:
            Path to models directory
        """
        # Try different possible locations
        possible_paths = [
            Path(__file__).parent.parent.parent / "models",  # From backend/services/ -> models/
            Path(__file__).parent.parent.parent / "backend" / "models",  # backend/models/
            Path(__file__).parent.parent / "models",  # Alternative
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Default to models/ in project root
        return Path(__file__).parent.parent.parent / "models"
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """
        Load a model from disk (with caching)
        
        Args:
            model_name: Name of the model file (e.g., 'random_forest_model.pkl')
            
        Returns:
            Loaded model or None if not found
        """
        # Check cache first
        if model_name in self.models_cache:
            return self.models_cache[model_name]
        
        # Try to load from disk
        model_file = self.models_path / model_name
        
        if not model_file.exists():
            return None
        
        try:
            model = joblib.load(model_file)
            self.models_cache[model_name] = model
            return model
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
    
    def get_available_models(self) -> list:
        """
        Get list of available model files
        
        Returns:
            List of model filenames
        """
        if not self.models_path.exists():
            return []
        
        # Exclude patterns (files that are NOT models)
        exclude_patterns = ["params", "results", "scaler", "best_params", "_results", "_scaler"]
        
        # Find all .pkl files that are models
        model_files = []
        for f in self.models_path.glob("*.pkl"):
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
                model_files.append(f.name)
        
        # Check if Random Forest model exists with different names
        rf_model_names = ["random_forest_model.pkl", "rf_model.pkl"]
        for rf_name in rf_model_names:
            if (self.models_path / rf_name).exists() and rf_name not in model_files:
                model_files.append(rf_name)
        
        # If we have rf_best_params or rf_results but no model file,
        # it means RF was trained but model file might be missing
        # Check for RF indicators
        has_rf_params = (self.models_path / "rf_best_params.pkl").exists()
        has_rf_results = (self.models_path / "rf_results.pkl").exists()
        
        # If RF params/results exist but no model file, add expected name
        if (has_rf_params or has_rf_results) and not any("random_forest" in m.lower() or m.startswith("rf_") for m in model_files):
            # Try to find if model exists with any name
            rf_model_found = False
            for f in self.models_path.glob("*.pkl"):
                if "rf" in f.name.lower() and not any(ex in f.name.lower() for ex in exclude_patterns):
                    if f.name not in model_files:
                        model_files.append(f.name)
                        rf_model_found = True
                        break
            
            # If still not found, add expected name (even if file doesn't exist)
            # This helps identify that RF should be available
            if not rf_model_found:
                model_files.append("random_forest_model.pkl")
        
        return sorted(model_files)


# Global instance
model_service = ModelService()
