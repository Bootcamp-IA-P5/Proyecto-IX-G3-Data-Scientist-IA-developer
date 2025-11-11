"""
FastAPI backend for stroke prediction.
MAin aplication file - configures the app and registers routes.
run: uvicorn backend.app:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import predict_router, health_router

# Create FastAPI app
app = FastAPI(
    title="Stroke Prediction API",
    description="API for predicting stroke risk using machine learning models.",
    version="1.0.0"
)
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(predict_router, prefix="/predict", tags=["Prediction"])
app.include_router(health_router, prefix="/health", tags=["Health"])
