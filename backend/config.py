"""
Configuration settings for the API
"""
import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_TITLE: str = "Stroke Prediction API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "API for stroke prediction using machine learning models"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", "8000"))  # Render uses PORT env var
    
    # CORS Settings - processed manually to avoid JSON parsing issues
    # This field is excluded from automatic parsing and handled in __init__
    _cors_origins_raw: str = ""
    
    @property
    def CORS_ORIGINS(self) -> List[str]:
        """Get CORS origins as a list, parsing from comma-separated string"""
        # Read from environment variable directly
        cors_env = os.getenv("CORS_ORIGINS", "")
        
        # Default origins
        default_origins = [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "https://proyecto-ix-g3-data-scientist-ia.onrender.com",  # Production frontend (without trailing slash)
        ]
        
        if not cors_env.strip():
            return default_origins
        
        # Split by comma and strip whitespace
        origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
        
        # Normalize URLs: remove trailing slashes and add both versions
        normalized_origins = []
        for origin in origins:
            # Remove trailing slash
            origin_clean = origin.rstrip("/")
            if origin_clean not in normalized_origins:
                normalized_origins.append(origin_clean)
            # Also add version with trailing slash for compatibility
            origin_with_slash = origin_clean + "/"
            if origin_with_slash not in normalized_origins:
                normalized_origins.append(origin_with_slash)
        
        # Always include production frontend (with and without trailing slash)
        default_prod = "https://proyecto-ix-g3-data-scientist-ia.onrender.com"
        if default_prod not in normalized_origins:
            normalized_origins.append(default_prod)
            normalized_origins.append(default_prod + "/")
        
        # Always include localhost origins for local development (even when testing against production backend)
        for local_origin in ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"]:
            if local_origin not in normalized_origins:
                normalized_origins.append(local_origin)
        
        return normalized_origins
    
    # Model Settings
    MODELS_DIR: Path = Path(__file__).parent.parent / "models"
    DATA_DIR: Path = Path(__file__).parent.parent / "src" / "data"
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignora campos extras del .env que no están en Settings


# Global settings instance
settings = Settings()

