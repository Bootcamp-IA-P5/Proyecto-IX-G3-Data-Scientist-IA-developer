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
        
        if not cors_env.strip():
            # Return default origins if empty
            return [
                "http://localhost:3000",
                "http://localhost:5173",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
                "https://proyecto-ix-g3-data-scientist-ia.onrender.com",  # Production frontend
            ]
        
        # Split by comma and strip whitespace
        origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
        # Always include production frontend if not already present
        default_prod = "https://proyecto-ix-g3-data-scientist-ia.onrender.com"
        if default_prod not in origins:
            origins.append(default_prod)
        return origins
    
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

