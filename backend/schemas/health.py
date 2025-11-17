"""
Health and status schemas
"""
from typing import List
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str


class StatusResponse(BaseModel):
    """Response model for system status"""
    api_status: str
    models_loaded: int
    available_models: List[str]

