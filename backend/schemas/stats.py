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


# Dataset Statistics Schemas
class DatasetOverviewResponse(BaseModel):
    """Response model for dataset overview"""
    total_samples: int
    total_features: int
    stroke_cases: int
    no_stroke_cases: int
    class_balance: Dict[str, float]
    missing_values: int


class AgeDistribution(BaseModel):
    """Age distribution item"""
    range: str
    count: int
    stroke_rate: float


class DemographicsResponse(BaseModel):
    """Response model for demographics statistics"""
    age: Dict[str, Any]  # mean, median, std, distribution
    gender: Dict[str, Dict[str, Any]]
    marital_status: Dict[str, Dict[str, Any]]


class ClinicalStatsResponse(BaseModel):
    """Response model for clinical statistics"""
    hypertension: Dict[str, Dict[str, Any]]
    heart_disease: Dict[str, Dict[str, Any]]
    avg_glucose_level: Dict[str, Any]
    bmi: Dict[str, Any]
    smoking_status: Dict[str, Dict[str, Any]]


class CorrelationResponse(BaseModel):
    """Response model for correlation analysis"""
    correlation_matrix: Dict[str, float]
    top_risk_factors: List[Dict[str, Any]]


class HighRiskProfile(BaseModel):
    """High-risk profile"""
    id: int
    name: str
    criteria: str
    count: int
    stroke_rate: float
    avg_risk_score: float


class HighRiskProfilesResponse(BaseModel):
    """Response model for high-risk profiles"""
    profiles: List[HighRiskProfile]

