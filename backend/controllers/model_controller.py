"""
Model controller

Handles model information and management logic.
"""
import pickle
from typing import List, Optional, Dict, Any
from pathlib import Path
from backend.services.model_service import model_service
from backend.schemas import ModelInfoResponse, ModelListResponse


class ModelController:
    """Controller for model operations"""
    
    @staticmethod
    def _get_model_type(model_name: str) -> str:
        """
        Determine model type from name
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Model type string
        """
        name_lower = model_name.lower()
        if "logistic" in name_lower or "lr" in name_lower:
            return "LogisticRegression"
        elif "random_forest" in name_lower or "rf" in name_lower:
            return "RandomForest"
        elif "xgboost" in name_lower or "xgb" in name_lower:
            return "XGBoost"
        else:
            return "Unknown"
    
    @staticmethod
    def _load_model_params(model_name: str) -> Optional[Dict[str, Any]]:
        """
        Load model parameters from best_params file
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Dictionary with parameters or None
        """
        models_path = model_service.models_path
        
        # Try different naming patterns
        base_name = model_name.replace("_model.pkl", "").replace(".pkl", "")
        param_files = [
            f"{base_name}_best_params.pkl",
            f"{base_name}_params.pkl",
            "rf_best_params.pkl" if "random_forest" in model_name.lower() else None,
            "logistic_regression_best_params.pkl" if "logistic" in model_name.lower() else None,
        ]
        
        for param_file in param_files:
            if param_file is None:
                continue
            param_path = models_path / param_file
            if param_path.exists():
                try:
                    with open(param_path, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    continue
        
        return None
    
    @staticmethod
    def _load_model_results(model_name: str) -> Optional[Dict[str, Any]]:
        """
        Load model results/metrics from results file
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Dictionary with results/metrics or None
        """
        models_path = model_service.models_path
        
        # Try different naming patterns
        base_name = model_name.replace("_model.pkl", "").replace(".pkl", "")
        result_files = [
            f"{base_name}_results.pkl",
            f"{base_name}_results.pkl",
            "rf_results.pkl" if "random_forest" in model_name.lower() else None,
            "logistic_regression_results.pkl" if "logistic" in model_name.lower() else None,
        ]
        
        for result_file in result_files:
            if result_file is None:
                continue
            result_path = models_path / result_file
            if result_path.exists():
                try:
                    with open(result_path, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    continue
        
        return None
    
    @staticmethod
    def _extract_metrics_from_results(results: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Extract metrics from results dictionary
        
        Args:
            results: Results dictionary from pickle file
            
        Returns:
            Dictionary with metrics or None
        """
        if not results:
            return None
        
        metrics = {}
        
        # Try to find test metrics (preferred) or validation metrics
        if "test_threshold_0.5" in results:
            test_metrics = results["test_threshold_0.5"]
            metrics = {
                "accuracy": test_metrics.get("accuracy", 0.0),
                "precision": test_metrics.get("precision", 0.0),
                "recall": test_metrics.get("recall", 0.0),
                "f1_score": test_metrics.get("f1_score", 0.0),
                "auc_roc": test_metrics.get("auc_roc", 0.0),
            }
        elif "validation_threshold_0.5" in results:
            val_metrics = results["validation_threshold_0.5"]
            metrics = {
                "accuracy": val_metrics.get("accuracy", 0.0),
                "precision": val_metrics.get("precision", 0.0),
                "recall": val_metrics.get("recall", 0.0),
                "f1_score": val_metrics.get("f1_score", 0.0),
                "auc_roc": val_metrics.get("auc_roc", 0.0),
            }
        
        return metrics if metrics else None
    
    @staticmethod
    def get_model_list() -> ModelListResponse:
        """
        Get list of available models
        
        Returns:
            ModelListResponse with list of model names
        """
        models = model_service.get_available_models()
        return ModelListResponse(
            models=models,
            total=len(models)
        )
    
    @staticmethod
    def get_model_info(model_name: str) -> ModelInfoResponse:
        """
        Get detailed information about a specific model
        
        Args:
            model_name: Name of the model file
            
        Returns:
            ModelInfoResponse with model information
            
        Raises:
            ValueError: If model is not found
        """
        # Check if model exists
        models = model_service.get_available_models()
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {models}")
        
        # Check if model is loaded in cache
        is_loaded = model_name in model_service.models_cache
        
        # Get model type
        model_type = ModelController._get_model_type(model_name)
        
        # Load parameters
        hyperparameters = ModelController._load_model_params(model_name)
        
        # Load results and extract metrics
        results = ModelController._load_model_results(model_name)
        metrics = ModelController._extract_metrics_from_results(results) if results else None
        
        # Get features (from preprocessing service if available)
        features_required = None
        try:
            from backend.services.preprocessing_service import preprocessing_service
            if preprocessing_service.expected_columns:
                features_required = preprocessing_service.expected_columns
        except:
            pass
        
        return ModelInfoResponse(
            model_name=model_name,
            model_type=model_type,
            is_loaded=is_loaded,
            features_required=features_required,
            hyperparameters=hyperparameters,
            metrics=metrics
        )


# Global instance
model_controller = ModelController()

