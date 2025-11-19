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


class ROCCurve(BaseModel):
    """ROC curve data - Muestra la capacidad del modelo para distinguir entre clases"""
    fpr: List[float]  # False Positive Rate (Tasa de Falsos Positivos)
    tpr: List[float]  # True Positive Rate (Tasa de Verdaderos Positivos / Recall)
    auc: Optional[float] = None  # Area Under Curve (Área bajo la curva, 0-1, mayor es mejor)
    description: str = "La curva ROC muestra la relación entre la tasa de verdaderos positivos (TPR) y la tasa de falsos positivos (FPR) para diferentes umbrales de decisión. Un AUC cercano a 1.0 indica un excelente modelo."


class PrecisionRecallCurve(BaseModel):
    """Precision-Recall curve data - Muestra el balance entre precisión y recall del modelo"""
    precision: List[float]  # Precisión (de los casos predichos como positivos, cuántos son realmente positivos)
    recall: List[float]  # Recall (de los casos realmente positivos, cuántos detectó el modelo)
    f1: Optional[float] = None  # F1 score at optimal threshold (balance entre precisión y recall)
    description: str = "La curva Precision-Recall muestra el balance entre precisión (exactitud de predicciones positivas) y recall (capacidad de detectar todos los casos positivos). Es especialmente útil cuando las clases están desbalanceadas."


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_config = {"protected_namespaces": ()}
    
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
    roc_curve: Optional[ROCCurve] = None
    precision_recall_curve: Optional[PrecisionRecallCurve] = None


class ModelListResponse(BaseModel):
    """Response model for list of available models"""
    models: List[str]
    total: int

