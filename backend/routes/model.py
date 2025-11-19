"""
Model endpoints

Routes for model information and management.
"""
from fastapi import APIRouter, HTTPException
from backend.controllers.model_controller import model_controller
from backend.services.model_service import model_service
from backend.schemas import ModelInfoResponse, ModelListResponse
from backend.config import settings

router = APIRouter()


@router.get("/models", response_model=ModelListResponse)
async def get_models() -> ModelListResponse:
    """
    Get list of available models
    
    Returns:
        ModelListResponse with list of model names
    """
    return model_controller.get_model_list()


@router.get("/models/{model_name:path}", response_model=ModelInfoResponse)
async def get_model_info(model_name: str) -> ModelInfoResponse:
    """
    Get detailed information about a specific model
    
    Args:
        model_name: Name of the model file (e.g., 'logistic_regression_model.pkl' or 'logistic_regression_model')
                   Accepts both with and without .pkl extension
                   URL-encoded names are automatically decoded
        
    Returns:
        ModelInfoResponse with model information
        
    Raises:
        HTTPException: If model is not found or error occurs
    """
    import urllib.parse
    
    # Decode URL-encoded model name
    model_name = urllib.parse.unquote(model_name)
    
    # Normalize model name: ensure it has .pkl extension if not present
    # But first check if it exists as-is
    available_models = model_service.get_available_models()
    
    # Try exact match first
    if model_name not in available_models:
        # Try adding .pkl if not present
        if not model_name.endswith('.pkl'):
            model_name_with_pkl = f"{model_name}.pkl"
            if model_name_with_pkl in available_models:
                model_name = model_name_with_pkl
        # Try removing .pkl if present
        elif model_name.endswith('.pkl'):
            model_name_without_pkl = model_name[:-4]
            if model_name_without_pkl in available_models:
                model_name = model_name_without_pkl
    
    try:
        return model_controller.get_model_info(model_name)
    except ValueError as e:
        # Model not found
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' not found. Available models: {available_models}"
        )
    except Exception as e:
        # Any other error (500)
        import traceback
        error_details = str(e)
        # In production, don't expose full traceback
        if settings.DEBUG:
            error_details = f"{str(e)}\n{traceback.format_exc()}"
        
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model information: {error_details}"
        )

