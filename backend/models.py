from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator



class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Health status")
    message: str = Field(..., description="Status message")


class StatusResponse(BaseModel):
    """Response model for system status"""
    api_status: str = Field(..., description="API status")
    models_loaded: int = Field(..., description="Number of models loaded")
    available_models: List[str] = Field(..., description="List of available model names")



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


# ============================================================================
# PREDICTION MODELS
# ============================================================================

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


# ============================================================================
# STATISTICS MODELS
# ============================================================================

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


# ============================================================================
# ERROR MODELS
# ============================================================================

class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
