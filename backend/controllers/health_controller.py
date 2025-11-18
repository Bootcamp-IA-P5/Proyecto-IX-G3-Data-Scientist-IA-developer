from typing import Dict, Any, List
from pathlib import Path
from backend.schemas import (
    HealthResponse, 
    StatusResponse,
    ControlCenterResponse,
    ComponentStatus,
    ModelHealthInfo
)
from backend.services.model_service import model_service
from backend.services.stats_service import stats_service
from backend.controllers.model_controller import model_controller
from backend.config import settings


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
    
    @staticmethod
    def get_control_center() -> ControlCenterResponse:
        """
        Get comprehensive control center information
        
        Returns:
            ControlCenterResponse with detailed system information
        """
        # Get basic status
        status = HealthController.get_status()
        available_models = model_service.get_available_models()
        models_loaded = len(model_service.models_cache)
        
        # Calculate component statuses
        components = []
        
        # API REST Status
        api_percentage = 100 if status.api_status == "running" else 0
        components.append(ComponentStatus(
            name="API REST",
            status="operational" if status.api_status == "running" else "error",
            percentage=api_percentage,
            message="API funcionando correctamente" if status.api_status == "running" else "API no disponible"
        ))
        
        # ML Model Status
        model_percentage = int((models_loaded / len(available_models) * 100)) if available_models else 0
        model_status = "operational" if models_loaded > 0 else "warning"
        components.append(ComponentStatus(
            name="Modelo ML",
            status=model_status,
            percentage=model_percentage,
            message=f"{models_loaded} de {len(available_models)} modelos cargados",
            details={"models_loaded": models_loaded, "total_models": len(available_models)}
        ))
        
        # Services Status
        services_status = "operational"
        services_message = "Todos los servicios operativos"
        # Check if stats service is working
        try:
            stats_service.get_overview_stats()
            services_percentage = 100
        except Exception:
            services_status = "warning"
            services_message = "Algunos servicios con problemas"
            services_percentage = 75
        
        components.append(ComponentStatus(
            name="Servicios",
            status=services_status,
            percentage=services_percentage,
            message=services_message
        ))
        
        # Storage Status
        models_path = model_service.models_path
        data_path = model_service.data_path
        
        total_storage_mb = 0.0
        models_storage_mb = 0.0
        
        # Calculate storage for models
        for search_path in [models_path, data_path]:
            if search_path.exists():
                for f in search_path.glob("*.pkl"):
                    if "model" in f.name.lower() or f.name.startswith("rf_") or f.name.startswith("random_forest"):
                        size_mb = f.stat().st_size / (1024 * 1024)
                        models_storage_mb += size_mb
                        total_storage_mb += size_mb
        
        # Estimate total storage (models + data files)
        for search_path in [models_path, data_path]:
            if search_path.exists():
                for f in search_path.glob("*"):
                    if f.is_file():
                        total_storage_mb += f.stat().st_size / (1024 * 1024)
        
        # Storage percentage (assuming 100MB as reasonable limit, adjust as needed)
        storage_limit_mb = 100.0
        storage_percentage = min(int((total_storage_mb / storage_limit_mb) * 100), 100)
        storage_status = "operational" if storage_percentage < 80 else "warning" if storage_percentage < 95 else "error"
        
        components.append(ComponentStatus(
            name="Almacenamiento",
            status=storage_status,
            percentage=storage_percentage,
            message=f"{total_storage_mb:.2f} MB utilizados",
            details={"total_mb": round(total_storage_mb, 2), "models_mb": round(models_storage_mb, 2)}
        ))
        
        # Get models health information
        models_health = []
        for model_name in available_models:
            is_loaded = model_name in model_service.models_cache
            is_available = True  # If it's in the list, it's available
            
            # Get file size
            file_size_mb = None
            for search_path in [models_path, data_path]:
                model_file = search_path / model_name
                if model_file.exists():
                    file_size_mb = model_file.stat().st_size / (1024 * 1024)
                    break
            
            # Check if metrics are available
            metrics_available = False
            try:
                model_info = model_controller.get_model_info(model_name)
                metrics_available = model_info.metrics is not None and len(model_info.metrics) > 0
            except Exception:
                pass
            
            status_str = "loaded" if is_loaded else "available"
            models_health.append(ModelHealthInfo(
                model_name=model_name,
                is_loaded=is_loaded,
                is_available=is_available,
                file_size_mb=round(file_size_mb, 2) if file_size_mb else None,
                status=status_str,
                metrics_available=metrics_available
            ))
        
        # Get prediction stats
        prediction_stats = stats_service.get_overview_stats()
        total_predictions = prediction_stats.get("total_predictions", 0)
        
        # Generate alerts and warnings
        alerts = []
        warnings = []
        
        if models_loaded == 0:
            warnings.append("Ningún modelo está cargado en memoria")
        
        if storage_percentage > 90:
            warnings.append(f"Almacenamiento al {storage_percentage}% de capacidad")
        
        if total_predictions == 0:
            warnings.append("No se han realizado predicciones aún")
        
        if api_percentage < 100:
            alerts.append("La API no está funcionando correctamente")
        
        # Configuration
        configuration = {
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
            "host": settings.HOST,
            "port": settings.PORT,
            "api_version": settings.API_VERSION,
            "models_directory": str(models_path),
            "data_directory": str(data_path)
        }
        
        return ControlCenterResponse(
            api_status=status.api_status,
            environment=settings.ENVIRONMENT,
            version=settings.API_VERSION,
            components=components,
            total_models=len(available_models),
            models_loaded=models_loaded,
            models_health=models_health,
            total_storage_mb=round(total_storage_mb, 2),
            models_storage_mb=round(models_storage_mb, 2),
            total_predictions=total_predictions,
            average_response_time_ms=None,  # Could be tracked in the future
            alerts=alerts,
            warnings=warnings,
            configuration=configuration
        )


# Global instance
health_controller = HealthController()

