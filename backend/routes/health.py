"""
Health check endpoints

Routes only define HTTP endpoints and call controllers for business logic.
"""
from fastapi import APIRouter
from backend.controllers.health_controller import health_controller
from backend.models import HealthResponse, StatusResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint
    
    Returns:
        HealthResponse with API status
    """
    return health_controller.get_health()


@router.get("/api/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """
    Get system and model status
    
    Returns:
        StatusResponse with system information
    """
    return health_controller.get_status()
