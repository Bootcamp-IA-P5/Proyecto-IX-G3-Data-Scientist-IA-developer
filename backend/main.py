"""
FastAPI application for Stroke Prediction API

This is the main entry point for the backend API.
It provides endpoints for model predictions, model information, and statistics.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import configuration and routes
from backend.config import settings
from backend.routes import health, predict

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/api", tags=["Prediction"])


@app.get("/")
async def root() -> dict:
    """
    Root endpoint with API information
    
    Returns:
        Dictionary with API information
    """
    return {
        "message": "Stroke Prediction API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )

