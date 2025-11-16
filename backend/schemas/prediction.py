"""
Prediction schemas
"""
from typing import Optional, List
from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    age: float = Field(..., ge=0, le=120, description="Age of the patient")
    hypertension: int = Field(..., ge=0, le=1, description="Hypertension (0 or 1)")
    heart_disease: int = Field(..., ge=0, le=1, description="Heart disease (0 or 1)")
    avg_glucose_level: float = Field(..., ge=0, description="Average glucose level")
    bmi: float = Field(..., ge=0, le=100, description="Body Mass Index")
    gender: str = Field(..., description="Gender (Male/Female/Other)")
    ever_married: str = Field(..., description="Ever married (Yes/No)")
    work_type: str = Field(..., description="Work type")
    Residence_type: str = Field(..., description="Residence type (Urban/Rural)")
    smoking_status: str = Field(..., description="Smoking status")
    model_name: Optional[str] = Field(None, description="Specific model to use (optional)")

    @validator('gender')
    def validate_gender(cls, v):
        allowed = ['Male', 'Female', 'Other']
        if v not in allowed:
            raise ValueError(f'gender must be one of {allowed}')
        return v

    @validator('ever_married')
    def validate_ever_married(cls, v):
        allowed = ['Yes', 'No']
        if v not in allowed:
            raise ValueError(f'ever_married must be one of {allowed}')
        return v


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    prediction: int = Field(..., description="Predicted class (0 or 1)")
    probability: float = Field(..., ge=0, le=1, description="Probability of stroke")
    model_used: str = Field(..., description="Model used for prediction")
    confidence: str = Field(..., description="Confidence level (Low/Medium/High)")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    data: List[PredictionRequest] = Field(..., description="List of prediction requests")
    model_name: Optional[str] = Field(None, description="Specific model to use (optional)")

    @validator('data')
    def validate_data_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError('data list cannot be empty')
        if len(v) > 100:
            raise ValueError('batch size cannot exceed 100')
        return v


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total: int = Field(..., description="Total number of predictions")
    model_used: str = Field(..., description="Model used for predictions")

