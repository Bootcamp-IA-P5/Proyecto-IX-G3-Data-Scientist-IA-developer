"""
Prediction controller

Handles prediction logic and coordinates with model service.
"""
from typing import List, Optional
from backend.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)
from backend.services.model_service import model_service


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
        # TODO: Implement actual prediction logic in Step 5
        # For now, return placeholder
        return PredictionResponse(
            prediction=0,
            probability=0.5,
            model_used=model_name or "default",
            confidence="Medium"
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
        # TODO: Implement actual batch prediction logic in Step 5
        predictions = [
            PredictController.predict_single(item, request.model_name)
            for item in request.data
        ]
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            model_used=request.model_name or "default"
        )


# Global instance
predict_controller = PredictController()

