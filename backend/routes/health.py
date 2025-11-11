"""
Health check endpoints
"""
from fastapi import APIRouter, HTTPException
from backend.services.model_service import get_model_service

# routers definition
router = APIRouter()

# model service instance
model_service = get_model_service()

# Health check endpoint
@router.get("/")
def read_root():
    """Health check endpoint"""
    
    return {"status": "ok",
            "model_loaded": model_service.is_model_loaded(),
            "model_name": model_service.get_model_name(),
            "message": "Stroke Prediction API is running."}
    
# Detailed health check endpoint
@router.get("/health")
def health_check():
    """Detailed health check endpoint with model information"""
    
    if not model_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_service.get_model_info()