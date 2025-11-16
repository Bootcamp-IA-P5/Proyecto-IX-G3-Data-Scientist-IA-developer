"""
Health and status schemas
"""
from typing import List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Health status")
    message: str = Field(..., description="Status message")


class StatusResponse(BaseModel):
    """Response model for system status"""
    api_status: str = Field(..., description="API status")
    models_loaded: int = Field(..., description="Number of models loaded")
    available_models: List[str] = Field(..., description="List of available model names")

