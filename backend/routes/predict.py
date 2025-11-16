"""
Prediction endpoints

Routes only define HTTP endpoints and call controllers for business logic.
"""
from typing import Optional
from fastapi import APIRouter, Query
from backend.controllers.predict_controller import predict_controller
from backend.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    model_name: Optional[str] = Query(None, description="Specific model to use")
) -> PredictionResponse:
    """
    Make a single prediction
    
    Args:
        request: Prediction request with patient data
        model_name: Optional specific model to use (query parameter)
        
    Returns:
        PredictionResponse with prediction results
    """
    # Use model_name from query if provided, otherwise from request
    final_model_name = model_name or request.model_name
    return predict_controller.predict_single(request, final_model_name)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Make batch predictions
    
    Args:
        request: Batch prediction request with list of patient data
        
    Returns:
        BatchPredictionResponse with list of predictions
    """
    return predict_controller.predict_batch(request)
