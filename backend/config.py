"""
Configuration settings for the API
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_TITLE: str = "Stroke Prediction API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "API for stroke prediction using machine learning models"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", "8000"))  # Render uses PORT env var
    
    # CORS Settings
    # Read from environment variable or use defaults
    # Format: comma-separated URLs, e.g., "http://localhost:3000,https://proyecto-ix-g3-data-scientist-ia.onrender.com"
    _cors_origins_env: str = os.getenv("CORS_ORIGINS", "")
    CORS_ORIGINS: list = (
        [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()]
        if _cors_origins_env
        else [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "https://proyecto-ix-g3-data-scientist-ia.onrender.com",  # Production frontend
        ]
    )
    
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

