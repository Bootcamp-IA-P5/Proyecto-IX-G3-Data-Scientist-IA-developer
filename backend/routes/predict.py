"""
Prediction endpoint
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

from services.model_service import ModelService

router = APIRouter()
model_service = ModelService()
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    """Request model for stroke prediction"""
    features: Dict[str, Any]  # Features as key-value pairs
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "age": 67.0,
                    "hypertension": 1,
                    "heart_disease": 0,
                    "avg_glucose_level": 228.69,
                    "bmi": 36.6,
                    "gender_Male": 1,
                    "gender_Other": 0,
                    "ever_married_Yes": 1,
                    "work_type_Never_worked": 0,
                    "work_type_Private": 1,
                    "work_type_Self-employed": 0,
                    "work_type_children": 0,
                    "Residence_type_Urban": 1,
                    "smoking_status_formerly_smoked": 1,
                    "smoking_status_never_smoked": 0,
                    "smoking_status_smokes": 0
                }
            }
        }

class PredictionResponse(BaseModel):
    """Response model for prediction results"""
    prediction: int
    probability: float
    confidence: str

@router.post("/predict", response_model=PredictionResponse)
async def predict_stroke(request: PredictionRequest):
    """
    Predict stroke risk based on patient features
    
    Returns:
        - prediction: 0 (no stroke) or 1 (stroke risk)
        - probability: probability of stroke (0.0 to 1.0)
        - confidence: confidence level description
    """
    try:
        logger.info(f"Received prediction request with features: {list(request.features.keys())}")
        
        # Make prediction
        result = model_service.predict(request.features)
        
        # Determine confidence level
        prob = result["probability"]
        if prob < 0.3:
            confidence = "low"
        elif prob < 0.7:
            confidence = "medium"
        else:
            confidence = "high"
        
        return PredictionResponse(
            prediction=result["prediction"],
            probability=round(prob, 4),
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
