# Stroke Prediction API - Backend

FastAPI backend para la API de predicción de ictus.

## 🚀 Inicio Rápido

### 1. Instalar dependencias

```bash
cd backend
pip install -r requirements.txt
```

### 2. Ejecutar la API

```bash
# Opción 1: Desde la raíz del proyecto
python -m backend.main

# Opción 2: Usando uvicorn directamente
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Acceder a la documentación

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📁 Estructura del Proyecto

```
backend/
├── main.py                 # Punto de entrada (solo inicialización FastAPI)
├── config.py              # Configuración de la aplicación
├── models.py              # Modelos Pydantic (requests/responses)
├── routes/                # Endpoints HTTP (solo definen rutas)
│   ├── health.py         # Health check endpoints
│   └── predict.py        # Prediction endpoints
├── controllers/          # Lógica de negocio
│   ├── health_controller.py
│   └── predict_controller.py
└── services/             # Acceso a datos/modelos
    └── model_service.py  # Servicio de modelos ML
```

### 🏗️ Arquitectura

- **Routes**: Solo definen endpoints HTTP, validan con Pydantic, llaman a controllers
- **Controllers**: Contienen la lógica de negocio, coordinan con services
- **Services**: Acceso a datos, modelos, recursos externos
- **Models**: Todos los modelos Pydantic para validación y tipado

## 🔧 Configuración

La configuración se puede ajustar mediante variables de entorno o un archivo `.env`:

```env
ENVIRONMENT=development
DEBUG=True
HOST=0.0.0.0
PORT=8000
```

## 🌐 CORS

La API está configurada para aceptar peticiones desde:
- `http://localhost:3000` (React default)
- `http://localhost:5173` (Vite default)

Para añadir más orígenes, edita `backend/config.py`.

## 📝 Endpoints Disponibles

### Implementados
- `GET /health` - Health check
- `GET /` - Información de la API

### Por implementar
- `GET /api/status` - Estado del sistema
- `GET /api/models` - Listar modelos
- `GET /api/models/{model_name}/info` - Info del modelo
- `GET /api/models/{model_name}/features` - Features requeridas
- `POST /api/predict` - Predicción individual
- `POST /api/predict/batch` - Predicciones en lote
- `GET /api/stats/overview` - Estadísticas generales
- `GET /api/stats/risk-distribution` - Distribución de riesgo
- `GET /api/models/compare` - Comparar modelos

## 🐳 Docker

```bash
docker build -t stroke-api backend/
docker run -p 8000:8000 stroke-api
```

## 📦 Dependencias Principales

- **FastAPI**: Framework web moderno y rápido
- **Uvicorn**: Servidor ASGI
- **Pydantic**: Validación de datos
- **scikit-learn**: Modelos de ML
- **joblib**: Carga de modelos

