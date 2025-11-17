"""
Model endpoints

Routes for model information and management.
"""
from fastapi import APIRouter, HTTPException
from backend.controllers.model_controller import model_controller
from backend.schemas import ModelInfoResponse, ModelListResponse

router = APIRouter()


@router.get("/models", response_model=ModelListResponse)
async def get_models() -> ModelListResponse:
    """
    Get list of available models
    
    Returns:
        ModelListResponse with list of model names
    """
    return model_controller.get_model_list()


@router.get("/models/{model_name}", response_model=ModelInfoResponse)
async def get_model_info(model_name: str) -> ModelInfoResponse:
    """
    Get detailed information about a specific model
    
    Args:
        model_name: Name of the model file (e.g., 'logistic_regression_model.pkl')
        
    Returns:
        ModelInfoResponse with model information
        
    Raises:
        HTTPException: If model is not found
    """
    try:
        return model_controller.get_model_info(model_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

