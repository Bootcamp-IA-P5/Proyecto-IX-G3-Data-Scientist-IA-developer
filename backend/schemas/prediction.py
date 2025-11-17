"""
Prediction schemas
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    age: float = Field(..., ge=0, le=120)
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)
    avg_glucose_level: float = Field(..., ge=0)
    bmi: float = Field(..., ge=0, le=100)
    gender: Literal["Male", "Female", "Other"] = Field(...)
    ever_married: Literal["Yes", "No"] = Field(...)
    work_type: str = Field(...)
    Residence_type: str = Field(...)
    smoking_status: str = Field(...)
    model_name: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    prediction: int = Field(..., ge=0, le=1)
    probability: float = Field(..., ge=0, le=1)
    model_used: str
    confidence: Literal["Low", "Medium", "High"]


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    data: List[PredictionRequest] = Field(..., min_length=1, max_length=100)
    model_name: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    total: int
    model_used: str

