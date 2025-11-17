"""
Health check endpoints
"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "stroke-prediction-api",
        "version": "1.0.0"
    }
