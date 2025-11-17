"""
Statistics schemas
"""
from typing import Optional, List, Dict
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

