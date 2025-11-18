"""
Model information schemas
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class ConfusionMatrixInfo(BaseModel):
    """Detailed confusion matrix information"""
    matrix: List[List[int]]  # [[TN, FP], [FN, TP]]
    labels: List[str] = ["No Ictus", "Ictus"]  # Class labels
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    total: int
    accuracy: float  # (TN + TP) / Total
    error_rate: float  # (FP + FN) / Total


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_name: str
    model_type: str
    is_loaded: bool
    features_required: Optional[List[str]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    feature_importance: Optional[List[Dict[str, Any]]] = None
    confusion_matrix: Optional[List[List[int]]] = None  # Deprecated: use confusion_matrix_info
    confusion_matrix_info: Optional[ConfusionMatrixInfo] = None
    optimal_threshold: Optional[float] = None


class ModelListResponse(BaseModel):
    """Response model for list of available models"""
    models: List[str]
    total: int

