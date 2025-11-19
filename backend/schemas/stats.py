"""
Statistics schemas
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class StatsOverviewResponse(BaseModel):
    """Response model for general statistics"""
    total_predictions: int
    stroke_predictions: int
    no_stroke_predictions: int
    average_probability: float = Field(..., ge=0, le=1)


class RiskDistributionResponse(BaseModel):
    """Response model for risk distribution"""
    low_risk: int
    medium_risk: int
    high_risk: int
    distribution: Dict[str, int]


class ModelComparisonResponse(BaseModel):
    """Response model for model comparison"""
    models: List[str]
    metrics: Dict[str, Dict[str, float]]
    best_model: Optional[str] = None


class DashboardResponse(BaseModel):
    """Response model for dashboard - combines all relevant information"""
    # System status
    api_status: str
    models_loaded: int
    total_models: int
    available_models: List[str]
    
    # Prediction statistics
    total_predictions: int
    stroke_predictions: int
    no_stroke_predictions: int
    average_probability: float
    
    # Risk distribution
    risk_distribution: Dict[str, int]
    
    # Best model information (highlighted)
    best_model: Optional[str] = None
    best_model_metrics: Optional[Dict[str, float]] = None
    best_model_type: Optional[str] = None
    
    # Quick model comparison (top 3 metrics)
    model_comparison: Dict[str, Dict[str, float]]
    
    # Model performance summary
    model_performance_summary: Dict[str, Any]

