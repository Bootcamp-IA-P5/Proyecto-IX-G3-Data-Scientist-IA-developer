"""
Health and status schemas
"""
from typing import List, Optional, Dict, Any
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


class ComponentStatus(BaseModel):
    """Status of a system component"""
    name: str
    status: str  # "operational", "warning", "error"
    percentage: int  # 0-100
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ModelHealthInfo(BaseModel):
    """Health information for a model"""
    model_name: str
    is_loaded: bool
    is_available: bool
    file_size_mb: Optional[float] = None
    status: str  # "available", "loaded", "error"
    metrics_available: bool = False


class ControlCenterResponse(BaseModel):
    """Comprehensive control center information"""
    # System overview
    api_status: str
    environment: str
    version: str
    
    # Component statuses
    components: List[ComponentStatus]
    
    # Models information
    total_models: int
    models_loaded: int
    models_health: List[ModelHealthInfo]
    
    # System resources
    total_storage_mb: float
    models_storage_mb: float
    
    # Performance metrics
    total_predictions: int
    average_response_time_ms: Optional[float] = None
    
    # Alerts and warnings
    alerts: List[str]
    warnings: List[str]
    
    # Configuration
    configuration: Dict[str, Any]

