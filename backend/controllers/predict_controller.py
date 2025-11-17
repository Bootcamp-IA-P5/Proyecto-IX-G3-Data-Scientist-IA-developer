"""
Prediction controller

Handles prediction logic and coordinates with model service.
"""
from typing import List, Optional
import numpy as np
from backend.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)
from backend.services.model_service import model_service
from backend.services.preprocessing_service import preprocessing_service


class PredictController:
    """Controller for prediction operations"""
    
    @staticmethod
    def _calculate_confidence(probability: float) -> str:
        """
        Calculate confidence level based on probability
        
        Args:
            probability: Prediction probability
            
        Returns:
            Confidence level string
        """
        if probability < 0.3 or probability > 0.7:
            return "High"
        elif 0.3 <= probability < 0.4 or 0.6 < probability <= 0.7:
            return "Medium"
        else:
            return "Low"
    
    @staticmethod
    def predict_single(request: PredictionRequest, model_name: Optional[str] = None) -> PredictionResponse:
        """
        Make a single prediction
        
        Args:
            request: Prediction request with patient data
            model_name: Optional specific model to use
            
        Returns:
            PredictionResponse with prediction results
            
        Raises:
            ValueError: If model is not found or prediction fails
        """
        # Determine which model to use
        final_model_name = model_name or request.model_name or "logistic_regression_model.pkl"
        
        # Load model
        model = model_service.load_model(final_model_name)
        if model is None:
            raise ValueError(f"Model '{final_model_name}' not found. Available models: {model_service.get_available_models()}")
        
        # Convert request to dict for preprocessing
        input_data = request.dict(exclude={'model_name'})
        
        # Preprocess input data
        try:
            X_preprocessed = preprocessing_service.preprocess(input_data)
        except Exception as e:
            raise ValueError(f"Error preprocessing input data: {str(e)}")
        
        # Make prediction
        try:
            probability = model.predict_proba(X_preprocessed)[0][1]  # Probability of class 1 (stroke)
            prediction = 1 if probability >= 0.5 else 0
        except Exception as e:
            raise ValueError(f"Error making prediction: {str(e)}")
        
        # Calculate confidence
        confidence = PredictController._calculate_confidence(probability)
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            model_used=final_model_name,
            confidence=confidence
        )
    
    @staticmethod
    def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
        """
        Make batch predictions
        
        Args:
            request: Batch prediction request with list of patient data
            
        Returns:
            BatchPredictionResponse with list of predictions
            
        Raises:
            ValueError: If model is not found or prediction fails
        """
        # Determine which model to use
        final_model_name = request.model_name or "logistic_regression_model.pkl"
        
        # Load model
        model = model_service.load_model(final_model_name)
        if model is None:
            raise ValueError(f"Model '{final_model_name}' not found. Available models: {model_service.get_available_models()}")
        
        # Convert requests to list of dicts for preprocessing
        input_data_list = [item.dict(exclude={'model_name'}) for item in request.data]
        
        # Preprocess batch input data
        try:
            X_preprocessed = preprocessing_service.preprocess_batch(input_data_list)
        except Exception as e:
            raise ValueError(f"Error preprocessing batch input data: {str(e)}")
        
        # Make batch predictions
        try:
            probabilities = model.predict_proba(X_preprocessed)[:, 1]  # Probabilities of class 1 (stroke)
            predictions_binary = (probabilities >= 0.5).astype(int)
        except Exception as e:
            raise ValueError(f"Error making batch predictions: {str(e)}")
        
        # Build response list
        predictions = [
            PredictionResponse(
                prediction=int(pred),
                probability=float(prob),
                model_used=final_model_name,
                confidence=PredictController._calculate_confidence(prob)
            )
            for pred, prob in zip(predictions_binary, probabilities)
        ]
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            model_used=final_model_name
        )


# Global instance
predict_controller = PredictController()

