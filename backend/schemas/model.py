"""
Model information schemas
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_name: str
    model_type: str
    is_loaded: bool
    features_required: Optional[List[str]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    feature_importance: Optional[List[Dict[str, Any]]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    optimal_threshold: Optional[float] = None


class ModelListResponse(BaseModel):
    """Response model for list of available models"""
    models: List[str]
    total: int

