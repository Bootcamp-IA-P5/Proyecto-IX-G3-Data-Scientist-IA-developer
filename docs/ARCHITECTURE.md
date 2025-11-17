# Arquitectura del Backend

## 🏗️ Estructura de Capas

```
┌─────────────────────────────────────────────────────────┐
│                    HTTP Request                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  ROUTES (routes/)                                       │
│  - Define endpoints HTTP                                │
│  - Valida requests con Pydantic                        │
│  - NO contiene lógica de negocio                       │
│  - Llama a controllers                                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  CONTROLLERS (controllers/)                             │
│  - Contiene lógica de negocio                          │
│  - Coordina entre routes y services                    │
│  - Transforma datos si es necesario                    │
│  - Maneja errores de negocio                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  SERVICES (services/)                                   │
│  - Acceso a datos (modelos, archivos, DB)              │
│  - Operaciones de bajo nivel                           │
│  - Caché de modelos                                    │
│  - NO contiene lógica de negocio                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  MODELS (models.py)                                     │
│  - Modelos Pydantic para validación                 │
│  - Requests y Responses tipados                        │
│  - Validación automática                               │
└─────────────────────────────────────────────────────────┘
```

## 📋 Flujo de una Petición

### Ejemplo: POST /api/predict

1. **Request llega a FastAPI** (`main.py`)
   - FastAPI valida el formato HTTP
   - Enruta a `routes/predict.py`

2. **Route valida con Pydantic** (`routes/predict.py`)
   ```python
   @router.post("/predict", response_model=PredictionResponse)
   async def predict(request: PredictionRequest) -> PredictionResponse:
   ```
   - Valida que el request cumpla con `PredictionRequest`
   - Si no es válido, retorna error 422 automáticamente

3. **Controller ejecuta lógica** (`controllers/predict_controller.py`)
   ```python
   return predict_controller.predict_single(request)
   ```
   - Procesa la lógica de negocio
   - Llama a services si necesita datos/modelos

4. **Service accede a recursos** (`services/model_service.py`)
   ```python
   model = model_service.load_model("random_forest_model.pkl")
   ```
   - Carga el modelo desde disco
   - Usa caché si está disponible

5. **Response tipado** (`models.py`)
   - Controller retorna `PredictionResponse`
   - FastAPI valida y serializa automáticamente
   - Cliente recibe JSON válido

## 🎯 Principios de Diseño

### ✅ Separación de Responsabilidades

- **Routes**: Solo HTTP, validación, routing
- **Controllers**: Solo lógica de negocio
- **Services**: Solo acceso a recursos
- **Models**: Solo definición de datos

### ✅ Tipado Fuerte

- Todos los requests/responses están tipados con Pydantic
- FastAPI valida automáticamente
- Type hints en todas las funciones
- IDE autocompleta correctamente

### ✅ Sin Lógica en Routes

```python
# ❌ MAL - Lógica en route
@router.post("/predict")
async def predict(request: PredictionRequest):
    model = joblib.load("model.pkl")  # ❌ Acceso directo a recursos
    prediction = model.predict([...])  # ❌ Lógica de negocio
    return {"prediction": prediction}

# ✅ BIEN - Route solo llama a controller
@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    return predict_controller.predict_single(request)  # ✅ Delega
```

### ✅ Sin Lógica en Services

```python
# ❌ MAL - Lógica de negocio en service
def predict(self, data):
    if data.age > 65:  # ❌ Lógica de negocio
        return "high_risk"
    return "low_risk"

# ✅ BIEN - Service solo accede a recursos
def load_model(self, name):
    return joblib.load(f"models/{name}")  # ✅ Solo acceso
```

## 📦 Ejemplo Completo

### Request
```json
POST /api/predict
{
  "age": 65,
  "hypertension": 1,
  "heart_disease": 0,
  "avg_glucose_level": 95.0,
  "bmi": 28.5,
  "gender": "Male",
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "smoking_status": "formerly smoked"
}
```

### Flujo
1. `routes/predict.py` → Valida con `PredictionRequest`
2. `controllers/predict_controller.py` → Procesa lógica
3. `services/model_service.py` → Carga modelo
4. Controller → Hace predicción
5. Retorna `PredictionResponse` tipado

### Response
```json
{
  "prediction": 1,
  "probability": 0.75,
  "model_used": "random_forest_model",
  "confidence": "High"
}
```

## 🔍 Ventajas de esta Arquitectura

1. **Testeable**: Cada capa se puede testear independientemente
2. **Mantenible**: Cambios en una capa no afectan otras
3. **Escalable**: Fácil añadir nuevos endpoints/features
4. **Type-safe**: Pydantic valida todo automáticamente
5. **Documentado**: Swagger se genera automáticamente
6. **Limpio**: Código organizado y fácil de entender

