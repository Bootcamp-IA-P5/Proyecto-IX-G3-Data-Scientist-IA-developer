from fastapi import FastAPI
from routes.health import router as health_router
from routes.predict import router as predict_router

app = FastAPI(
    title="Stroke Prediction API",
    description="API for stroke prediction using machine learning",
    version="1.0.0"
)

app.include_router(health_router)
app.include_router(predict_router)

@app.get("/")
async def root():
    return {"message": "Stroke Prediction API", "status": "running"}
