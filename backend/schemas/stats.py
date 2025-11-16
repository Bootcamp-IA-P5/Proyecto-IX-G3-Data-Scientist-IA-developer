"""
Statistics schemas
"""
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class StatsOverviewResponse(BaseModel):
    """Response model for general statistics"""
    total_predictions: int = Field(..., description="Total predictions made")
    stroke_predictions: int = Field(..., description="Number of stroke predictions")
    no_stroke_predictions: int = Field(..., description="Number of no-stroke predictions")
    average_probability: float = Field(..., ge=0, le=1, description="Average stroke probability")


class RiskDistributionResponse(BaseModel):
    """Response model for risk distribution"""
    low_risk: int = Field(..., description="Number of low risk predictions")
    medium_risk: int = Field(..., description="Number of medium risk predictions")
    high_risk: int = Field(..., description="Number of high risk predictions")
    distribution: Dict[str, int] = Field(..., description="Risk distribution breakdown")


class ModelComparisonResponse(BaseModel):
    """Response model for model comparison"""
    models: List[str] = Field(..., description="List of compared models")
    metrics: Dict[str, Dict[str, float]] = Field(..., description="Metrics for each model")
    best_model: Optional[str] = Field(None, description="Best performing model")

