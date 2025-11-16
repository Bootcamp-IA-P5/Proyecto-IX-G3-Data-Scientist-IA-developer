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
        
        # Find all .pkl files that are models (not params or results)
        model_files = [
            f.name for f in self.models_path.glob("*.pkl")
            if "model" in f.name.lower() and "params" not in f.name.lower() and "results" not in f.name.lower()
        ]
        
        return sorted(model_files)


# Global instance
model_service = ModelService()
