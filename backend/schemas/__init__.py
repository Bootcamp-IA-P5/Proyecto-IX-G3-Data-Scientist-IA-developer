"""
Schemas module for Pydantic models

This module exports all Pydantic models organized by domain.
Import from here to maintain backward compatibility.
"""
from backend.schemas.health import HealthResponse, StatusResponse
from backend.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)
from backend.schemas.model import ModelInfoResponse, ModelListResponse
from backend.schemas.stats import (
    StatsOverviewResponse,
    RiskDistributionResponse,
    ModelComparisonResponse
)
from backend.schemas.error import ErrorResponse

__all__ = [
    # Health
    "HealthResponse",
    "StatusResponse",
    # Prediction
    "PredictionRequest",
    "PredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    # Model
    "ModelInfoResponse",
    "ModelListResponse",
    # Stats
    "StatsOverviewResponse",
    "RiskDistributionResponse",
    "ModelComparisonResponse",
    # Error
    "ErrorResponse",
]

