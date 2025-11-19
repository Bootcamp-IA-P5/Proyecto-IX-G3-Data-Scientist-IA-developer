from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

# Import configuration and routes
from backend.config import settings
from backend.routes import health, predict, model, stats

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Log CORS origins for debugging
cors_origins = settings.CORS_ORIGINS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"🌐 CORS Origins configured: {cors_origins}")

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,  # Use dynamic configuration from settings
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(model.router, tags=["Models"])
app.include_router(stats.router, tags=["Statistics"])


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

