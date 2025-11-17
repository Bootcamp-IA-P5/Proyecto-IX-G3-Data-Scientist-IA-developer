"""
Service layer for ML model operations.

This module handles:
- Loading and caching the trained model
- Making predictions
- Model validation
- Feature preprocessing (if needed)
- Error handling for model operations
"""
from pathlib import Path
from typing import Dict, Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

class ModelService:
    """Service class for managning ML model operations."""
    
    # Constructor
    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            base_path = Path(__file__).resolve().parents[2]
            model_path = base_path / "models/stroke_prediction_model.pkl"
            
        self.model_path = model_path
        self.model = None
        self.model_name = "Stroke Prediction Model"
        self._load_model()
        
    def _load_model(self):
        """Load the model from disk"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            self.model_loaded_at = datetime.now()
            print(f"Model loaded successfully from: {self.model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            raise
        
    def is_model_loaded(self):
        """Check if the model is loaded"""
        return self.model is not None
    
    def get_model_name(self):
        """Return the model name"""
        return self.model_name

    def get_model_info(self) -> Dict[str, str]:
        """Return model information"""
        if not self.is_model_loaded():
            return {"status": "not_loaded"}
        return {
            "status": "loaded",
            "model_path": str(self.model_path),
            "model_type": type(self.model).__name__,
            "loaded_at": self.model_loaded_at.isoformat() if self.model_loaded_at else None
        }
        
    def validate_features(self, df: pd.DataFrame)-> None:
        """Validate input features DataFrame"""
        expected_features = [
            'gender', 'age', 'hypertension', 'heart_disease',
            'ever_married', 'work_type', 'Residence_type',
            'avg_glucose_level', 'bmi', 'smoking_status'
        ]
            
        missing_features = []
        for feature in expected_features:
            if feature not in df.columns:
                missing_features.append(feature)
        if missing_features:
            raise ValueError(f"Missing required features: {', '.join(missing_features)}")
    
    def calculate_risk_level(self, stroke_probability: float) -> str:
        """Calculate risk level based on stroke probability
        Args:
            probability: Probability of stroke (0-1)
        Returns:
            Risk level as string: "Bajo", "Medio", or "Alto"
            """
        
        if stroke_probability < 0.2:
            return "Low"
        elif 0.2 <= stroke_probability < 0.5:
            return "Medium"
        else:
            return "High"
        
    def _calculate_confidence(self, probas: np.ndarray) -> float:
        """Calculate confidence as the max probability"""
        
        max_proba = max(probas)
        
        if max_proba >= 0.9:
            return "Hightest"
        elif max_proba >= 0.7:
            return "High"
        elif max_proba >= 0.5:
            return "Medium"
        else:
            return "Low"
        
    def predict(self, features:Dict[str, any]) -> Dict[str, any]:
        """Make predictions using the loaded model"""
        
        if not self.is_model_loaded():
            try:
                self._load_model()
            except Exception as e:
                raise ValueError(f"Model could not be loaded: {e}")
        try:
            # Convert features dict to DataFrame, validate and predict required features
            df = pd.DataFrame([features])
            self._validate_features(df)
            prediction = int(self.model.predict(df)[0])
            
            # Get probabilities
            probas = self.model.predict_proba(df)[0]
            stroke_probability = float(probas[1])
            
            # Calculate risk level
            risk_level = self.calculate_risk_level(stroke_probability)
            
            return {
                "prediction": prediction,
                "stroke_probability": round(stroke_probability, 4),
                "no_stroke_probability": round(probas[0], 4),
                "risk_level": risk_level,
                "confidence": self.calculate_confidence(probas)
            }
        except Exception as e:
            raise ValueError(f"Prediction failed: {e}")

    def batch_predict(self, features_list: list[Dict[str, any]]) -> list[Dict[str, any]]:
        """Make batch predictions using the loaded model"""
    
    
        if not self.is_model_loaded():
            try:
                self._load_model()
            except Exception as e:
                raise ValueError(f"Model could not be loaded: {e}")
        try:
            # Convert list of features dicts to DataFrame
            df = pd.DataFrame(features_list)
            self.validate_features(df)
            
            predictions = self.model.predict(df)
            probas = self.model.predict_proba(df)
            
            results = []
            for i in range(len(predictions)):
                stroke_probability = float(probas[i][1])
                risk_level = self.calculate_risk_level(stroke_probability)
                
                results.append({
                    "prediction": int(predictions[i]),
                    "stroke_probability": round(stroke_probability, 4),
                    "no_stroke_probability": round(probas[i][0], 4),
                    "risk_level": risk_level,
                    "confidence": self._calculate_confidence(probas[i])
                })
            return results
        except Exception as e:
            raise ValueError(f"Batch prediction failed: {e}")
        
    def get_model_service():
        """Factory function to get a singleton ModelService instance"""
        pass