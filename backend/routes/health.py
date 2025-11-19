from fastapi import APIRouter
from backend.controllers.health_controller import health_controller
from backend.schemas import HealthResponse, StatusResponse, ControlCenterResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint
    
    Returns:
        HealthResponse with API status
    """
    return health_controller.get_health()


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """
    Get system and model status
    
    Returns:
        StatusResponse with system information
    """
    return health_controller.get_status()


@router.get("/control-center", response_model=ControlCenterResponse)
async def get_control_center() -> ControlCenterResponse:
    """
    Get comprehensive control center information
    
    Returns detailed system status including:
    - Component health (API, Models, Services, Storage)
    - Model health information
    - System resources
    - Performance metrics
    - Alerts and warnings
    - Configuration
    
    Returns:
        ControlCenterResponse with comprehensive system information
    """
    return health_controller.get_control_center()
