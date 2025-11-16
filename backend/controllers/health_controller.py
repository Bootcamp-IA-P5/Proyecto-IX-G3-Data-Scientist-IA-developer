"""
Health controller

Handles health check and system status logic.
"""
from typing import Dict, Any
from backend.models import HealthResponse, StatusResponse
from backend.services.model_service import model_service


class HealthController:
    """Controller for health-related operations"""
    
    @staticmethod
    def get_health() -> HealthResponse:
        """
        Get API health status
        
        Returns:
            HealthResponse with status information
        """
        return HealthResponse(
            status="healthy",
            message="API is running"
        )
    
    @staticmethod
    def get_status() -> StatusResponse:
        """
        Get system and model status
        
        Returns:
            StatusResponse with system information
        """
        available_models = model_service.get_available_models()
        models_loaded = len(model_service.models_cache)
        
        return StatusResponse(
            api_status="running",
            models_loaded=models_loaded,
            available_models=available_models
        )


# Global instance
health_controller = HealthController()

