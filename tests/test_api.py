import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "message" in data

def test_predict_endpoint_valid_data():
    """Test prediction with valid data"""
    test_data = {
        "age": 65,
        "hypertension": 1,
        "heart_disease": 0,
        "avg_glucose_level": 150,
        "bmi": 28,
        "gender": "Male",
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "smoking_status": "never smoked"
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    
    # Verificar estructura de respuesta
    assert "prediction" in data
    assert "probability" in data
    assert "model_used" in data
    assert "confidence" in data
    
    # Verificar tipos de datos
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)
    assert isinstance(data["model_used"], str)
    assert isinstance(data["confidence"], str)

def test_predict_endpoint_missing_field():
    """Test prediction with missing required field"""
    incomplete_data = {
        "age": 65,
        "hypertension": 1
        # Faltan otros campos requeridos
    }
    
    response = client.post("/predict", json=incomplete_data)
    # Debería retornar error de validación
    assert response.status_code == 422

def test_predict_batch_endpoint():
    """Test batch prediction"""
    test_batch = {
        "data": [
            {
                "age": 65,
                "hypertension": 1,
                "heart_disease": 0,
                "avg_glucose_level": 150,
                "bmi": 28,
                "gender": "Male",
                "ever_married": "Yes",
                "work_type": "Private",
                "Residence_type": "Urban",
                "smoking_status": "never smoked"
            }
        ]
    }
    
    response = client.post("/predict/batch", json=test_batch)
    assert response.status_code == 200
    data = response.json()
    
    assert "predictions" in data
    assert "total" in data
    assert "model_used" in data
    assert len(data["predictions"]) == 1
