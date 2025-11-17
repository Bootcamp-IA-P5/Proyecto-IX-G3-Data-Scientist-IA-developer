"""
Service layer for ML model operations.
"""
import os
import joblib
import logging
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

class ModelService:
    """Service for handling ML model operations"""
    
    def __init__(self, model_path: str = "/app/models"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and scaler"""
        try:
            # Load Random Forest model (winner model)
            model_file = os.path.join(self.model_path, "rf_results.pkl")
            if os.path.exists(model_file):
                model_data = joblib.load(model_file)
                self.model = model_data['model']
                logger.info("Random Forest model loaded successfully")
            else:
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            # Try to load scaler if exists
            scaler_file = os.path.join(self.model_path, "logistic_regression_scaler.pkl")
            if os.path.exists(scaler_file):
                self.scaler = joblib.load(scaler_file)
                logger.info("Scaler loaded successfully")
            else:
                logger.warning("Scaler not found, predictions may be less accurate")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction using the loaded model
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Dictionary with prediction and probability
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Convert features to array
            feature_values = [features[key] for key in sorted(features.keys())]
            X = np.array([feature_values])
            
            # Scale if scaler available
            if self.scaler:
                X = self.scaler.transform(X)
            
            # Make prediction
            prediction = int(self.model.predict(X)[0])
            probability = float(self.model.predict_proba(X)[0][1])  # Probability of positive class
            
            logger.info(f"Prediction made: {prediction} with probability {probability}")
            
            return {
                "prediction": prediction,
                "probability": probability
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
