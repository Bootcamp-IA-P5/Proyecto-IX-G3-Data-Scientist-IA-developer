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
        Searches in both models/ and data/ directories
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Dictionary with parameters or None
        """
        models_path = model_service.models_path
        data_path = model_service.data_path
        
        # Try different naming patterns
        base_name = model_name.replace("_model.pkl", "").replace(".pkl", "")
        param_files = [
            f"{base_name}_best_params.pkl",
            f"{base_name}_params.pkl",
            "rf_best_params.pkl" if "random_forest" in model_name.lower() else None,
            "logistic_regression_best_params.pkl" if "logistic" in model_name.lower() else None,
            "xgboost_best_params_no_smote.pkl" if "xgboost" in model_name.lower() else None,
        ]
        
        # Search in both directories
        search_paths = [models_path, data_path]
        
        for param_file in param_files:
            if param_file is None:
                continue
            for search_path in search_paths:
                param_path = search_path / param_file
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
        Searches in both models/ and data/ directories
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Dictionary with results/metrics or None
        """
        models_path = model_service.models_path
        data_path = model_service.data_path
        
        # Try different naming patterns
        base_name = model_name.replace("_model.pkl", "").replace(".pkl", "")
        result_files = [
            f"{base_name}_results.pkl",
            "rf_results.pkl" if "random_forest" in model_name.lower() else None,
            "logistic_regression_results.pkl" if "logistic" in model_name.lower() else None,
            "xgboost_results_no_smote.pkl" if "xgboost" in model_name.lower() else None,
        ]
        
        # Search in both directories
        search_paths = [models_path, data_path]
        
        for result_file in result_files:
            if result_file is None:
                continue
            for search_path in search_paths:
                result_path = search_path / result_file
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
        Handles different structures from different models:
        - Logistic Regression: performance_metrics_threshold_0.5
        - Random Forest: test_threshold_0.5 or validation_threshold_0.5
        - XGBoost: test_threshold_optimal or validation_threshold_optimal
        
        Args:
            results: Results dictionary from pickle file
            
        Returns:
            Dictionary with metrics or None
        """
        if not results:
            return None
        
        metrics = {}
        
        # Priority order: test > validation, optimal > 0.5, performance_metrics
        # Helper function to convert numpy types
        def to_float(value):
            """Convert value to float, handling numpy types"""
            import numpy as np
            if value is None:
                return 0.0
            if isinstance(value, (np.integer, np.int64, np.int32)):
                return float(int(value))
            if isinstance(value, (np.floating, np.float64, np.float32)):
                return float(value)
            return float(value) if value is not None else 0.0
        
        # 1. Test metrics with optimal threshold (XGBoost)
        if "test_threshold_optimal" in results:
            test_metrics = results["test_threshold_optimal"]
            metrics = {
                "accuracy": to_float(test_metrics.get("accuracy", 0.0)),
                "precision": to_float(test_metrics.get("precision", 0.0)),
                "recall": to_float(test_metrics.get("recall", 0.0)),
                "f1_score": to_float(test_metrics.get("f1_score", 0.0)),
                "auc_roc": to_float(test_metrics.get("auc_roc", 0.0)),
            }
        # 2. Test metrics with 0.5 threshold (Random Forest)
        elif "test_threshold_0.5" in results:
            test_metrics = results["test_threshold_0.5"]
            metrics = {
                "accuracy": to_float(test_metrics.get("accuracy", 0.0)),
                "precision": to_float(test_metrics.get("precision", 0.0)),
                "recall": to_float(test_metrics.get("recall", 0.0)),
                "f1_score": to_float(test_metrics.get("f1_score", 0.0)),
                "auc_roc": to_float(test_metrics.get("auc_roc", 0.0)),
            }
        # 3. Performance metrics with 0.5 threshold (Logistic Regression)
        elif "performance_metrics_threshold_0.5" in results:
            perf_metrics = results["performance_metrics_threshold_0.5"]
            metrics = {
                "accuracy": to_float(perf_metrics.get("accuracy", 0.0)),
                "precision": to_float(perf_metrics.get("precision", 0.0)),
                "recall": to_float(perf_metrics.get("recall", 0.0)),
                "f1_score": to_float(perf_metrics.get("f1_score", 0.0)),
                "auc_roc": to_float(perf_metrics.get("auc_roc", 0.0)),
            }
        # 4. Validation metrics with optimal threshold (XGBoost fallback)
        elif "validation_threshold_optimal" in results:
            val_metrics = results["validation_threshold_optimal"]
            metrics = {
                "accuracy": to_float(val_metrics.get("accuracy", 0.0)),
                "precision": to_float(val_metrics.get("precision", 0.0)),
                "recall": to_float(val_metrics.get("recall", 0.0)),
                "f1_score": to_float(val_metrics.get("f1_score", 0.0)),
                "auc_roc": to_float(val_metrics.get("auc_roc", 0.0)),
            }
        # 5. Validation metrics with 0.5 threshold (Random Forest fallback)
        elif "validation_threshold_0.5" in results:
            val_metrics = results["validation_threshold_0.5"]
            metrics = {
                "accuracy": to_float(val_metrics.get("accuracy", 0.0)),
                "precision": to_float(val_metrics.get("precision", 0.0)),
                "recall": to_float(val_metrics.get("recall", 0.0)),
                "f1_score": to_float(val_metrics.get("f1_score", 0.0)),
                "auc_roc": to_float(val_metrics.get("auc_roc", 0.0)),
            }
        # 6. Performance metrics with optimal threshold (Logistic Regression fallback)
        elif "performance_metrics_threshold_optimal" in results:
            perf_metrics = results["performance_metrics_threshold_optimal"]
            metrics = {
                "accuracy": to_float(perf_metrics.get("accuracy", 0.0)),
                "precision": to_float(perf_metrics.get("precision", 0.0)),
                "recall": to_float(perf_metrics.get("recall", 0.0)),
                "f1_score": to_float(perf_metrics.get("f1_score", 0.0)),
                "auc_roc": to_float(perf_metrics.get("auc_roc", 0.0)),
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
        
        # Extract additional information from results
        feature_importance = None
        confusion_matrix = None
        optimal_threshold = None
        
        if results:
            # Extract feature importance
            if "feature_importance" in results:
                feature_importance = results["feature_importance"]
            elif "feature_importances" in results:
                # Convert array to list of dicts
                import numpy as np
                importances = results["feature_importances"]
                if isinstance(importances, (list, np.ndarray)):
                    # If we have feature names, use them
                    try:
                        from backend.services.preprocessing_service import preprocessing_service
                        feature_names = preprocessing_service.expected_columns or []
                        if feature_names and len(feature_names) == len(importances):
                            feature_importance = [
                                {"feature": name, "importance": float(imp)}
                                for name, imp in zip(feature_names, importances)
                            ]
                        else:
                            feature_importance = [
                                {"feature": f"feature_{i}", "importance": float(imp)}
                                for i, imp in enumerate(importances)
                            ]
                    except:
                        feature_importance = [
                            {"feature": f"feature_{i}", "importance": float(imp)}
                            for i, imp in enumerate(importances)
                        ]
            
            # Extract confusion matrix
            # Try direct keys first
            if "test_confusion_matrix" in results:
                confusion_matrix = results["test_confusion_matrix"]
            elif "confusion_matrix" in results:
                confusion_matrix = results["confusion_matrix"]
            elif "validation_confusion_matrix" in results:
                confusion_matrix = results["validation_confusion_matrix"]
            else:
                # Try to find in nested metrics (e.g., performance_metrics_threshold_0.5, test_threshold_0.5)
                for key in ["test_threshold_optimal", "test_threshold_0.5", 
                           "validation_threshold_optimal", "validation_threshold_0.5",
                           "performance_metrics_threshold_0.5", "performance_metrics_threshold_optimal"]:
                    if key in results and isinstance(results[key], dict):
                        if "confusion_matrix" in results[key]:
                            confusion_matrix = results[key]["confusion_matrix"]
                            break
            
            # Extract optimal threshold
            if "optimal_threshold" in results:
                optimal_threshold = results["optimal_threshold"]
            elif "best_threshold" in results:
                optimal_threshold = results["best_threshold"]
        
        # Get features (from preprocessing service if available)
        features_required = None
        try:
            from backend.services.preprocessing_service import preprocessing_service
            if preprocessing_service.expected_columns:
                features_required = preprocessing_service.expected_columns
        except:
            pass
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(value):
            """Convert numpy types and other non-serializable types to native Python types"""
            import numpy as np
            if value is None:
                return None
            if isinstance(value, (np.integer, np.int64, np.int32, np.int8, np.int16)):
                return int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
                return float(value)
            elif isinstance(value, (np.bool_, bool)):
                return bool(value)
            elif isinstance(value, (int, float, str, bool)):
                return value
            elif isinstance(value, (list, tuple)):
                return [convert_to_native(item) for item in value]
            elif isinstance(value, dict):
                return {str(k): convert_to_native(v) for k, v in value.items()}
            else:
                try:
                    # Try to convert to string as last resort
                    return str(value)
                except:
                    return None
        
        if hyperparameters:
            hyperparameters = {
                str(k): convert_to_native(v)
                for k, v in hyperparameters.items()
            }
        
        if metrics:
            metrics = {
                str(k): convert_to_native(v)
                for k, v in metrics.items()
            }
        
        # Convert feature importance and confusion matrix
        if feature_importance:
            if isinstance(feature_importance, list):
                feature_importance = [convert_to_native(item) for item in feature_importance]
            else:
                feature_importance = convert_to_native(feature_importance)
        
        # Process confusion matrix
        confusion_matrix_info = None
        if confusion_matrix is not None:
            # Convert confusion matrix to list of lists
            import numpy as np
            if isinstance(confusion_matrix, np.ndarray):
                confusion_matrix = confusion_matrix.tolist()
            elif isinstance(confusion_matrix, (list, tuple)):
                confusion_matrix = [[int(convert_to_native(cell)) for cell in row] for row in confusion_matrix]
            
            # Create detailed confusion matrix info
            if isinstance(confusion_matrix, list) and len(confusion_matrix) == 2 and len(confusion_matrix[0]) == 2:
                tn = int(confusion_matrix[0][0])  # True Negative
                fp = int(confusion_matrix[0][1])  # False Positive
                fn = int(confusion_matrix[1][0])  # False Negative
                tp = int(confusion_matrix[1][1])  # True Positive
                total = tn + fp + fn + tp
                
                from backend.schemas.model import ConfusionMatrixInfo
                confusion_matrix_info = ConfusionMatrixInfo(
                    matrix=confusion_matrix,
                    labels=["No Ictus", "Ictus"],
                    true_negative=tn,
                    false_positive=fp,
                    false_negative=fn,
                    true_positive=tp,
                    total=total,
                    accuracy=round((tn + tp) / total, 4) if total > 0 else 0.0,
                    error_rate=round((fp + fn) / total, 4) if total > 0 else 0.0
                )
        
        if optimal_threshold is not None:
            optimal_threshold = convert_to_native(optimal_threshold)
        
        return ModelInfoResponse(
            model_name=model_name,
            model_type=model_type,
            is_loaded=is_loaded,
            features_required=features_required,
            hyperparameters=hyperparameters,
            metrics=metrics,
            feature_importance=feature_importance,
            confusion_matrix=confusion_matrix,  # Keep for backward compatibility
            confusion_matrix_info=confusion_matrix_info,
            optimal_threshold=optimal_threshold
        )


# Global instance
model_controller = ModelController()

