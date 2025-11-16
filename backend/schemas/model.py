"""
Model information schemas
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of model (e.g., RandomForest)")
    is_loaded: bool = Field(..., description="Whether the model is currently loaded")
    features_required: Optional[List[str]] = Field(None, description="Required feature names")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Model hyperparameters")
    metrics: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")


class ModelListResponse(BaseModel):
    """Response model for list of available models"""
    models: List[str] = Field(..., description="List of available model names")
    total: int = Field(..., description="Total number of models")

