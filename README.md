# 🏥 Stroke Prediction API - Proyecto IX G3

Sistema de predicción de ictus (stroke) utilizando Machine Learning con modelos de Regresión Logística, Random Forest y XGBoost. API REST desarrollada con FastAPI y frontend en React + TypeScript.

## 🌐 Enlaces de Deployment

- **Frontend (React)**: [https://proyecto-ix-g3-data-scientist-ia.onrender.com/](https://proyecto-ix-g3-data-scientist-ia.onrender.com/)
- **Backend API (FastAPI)**: [https://proyecto-ix-g3-data-scientist-ia-78z0.onrender.com](https://proyecto-ix-g3-data-scientist-ia-78z0.onrender.com)
- **API Documentation (Swagger)**: [https://proyecto-ix-g3-data-scientist-ia-78z0.onrender.com/docs](https://proyecto-ix-g3-data-scientist-ia-78z0.onrender.com/docs)
- **Repositorio Frontend**: [https://github.com/Bootcamp-IA-P5/Proyecto-IX-G3-Data-Scientist-IA-developer--Frontend](https://github.com/Bootcamp-IA-P5/Proyecto-IX-G3-Data-Scientist-IA-developer--Frontend)

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Endpoints de la API](#-endpoints-de-la-api)
- [Base de Datos](#-base-de-datos)
- [Modelos de Machine Learning](#-modelos-de-machine-learning)
- [MLflow](#-mlflow)
- [Instalación y Configuración](#-instalación-y-configuración)
- [Deployment](#-deployment)
- [Uso de la API](#-uso-de-la-api)

## 🎯 Descripción

Este proyecto implementa un sistema completo de predicción de ictus cerebral utilizando técnicas de Machine Learning. El sistema permite:

- **Predicción individual**: Evaluar el riesgo de ictus de un paciente basándose en características demográficas y clínicas
- **Predicción por lotes**: Procesar múltiples pacientes simultáneamente
- **Análisis estadístico**: Visualizar estadísticas del dataset, correlaciones y perfiles de alto riesgo
- **Comparación de modelos**: Evaluar y comparar el rendimiento de diferentes modelos ML
- **Monitoreo del sistema**: Dashboard de control con métricas en tiempo real

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                         │
│  https://proyecto-ix-g3-data-scientist-ia.onrender.com/         │
└────────────────────────────┬────────────────────────────────────┘
                              │ HTTPS/REST API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND API (FastAPI)                         │
│  https://proyecto-ix-g3-data-scientist-ia-78z0.onrender.com     │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Routes     │  │ Controllers  │  │  Services    │          │
│  │              │  │              │  │              │          │
│  │ - health     │→ │ - health     │→ │ - model      │          │
│  │ - predict    │  │ - predict    │  │ - stats      │          │
│  │ - model      │  │ - model      │  │ - dataset    │          │
│  │ - stats      │  │ - stats      │  │ - preprocess │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                  │                  │                  │
│         └──────────────────┼──────────────────┘                 │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────┐              │
│  │         Machine Learning Models              │              │
│  │  - Logistic Regression (default)              │              │
│  │  - Random Forest                             │              │
│  │  - XGBoost                                   │              │
│  └──────────────────────────────────────────────┘              │
└────────────────────────────┬────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
        ┌──────────────────┐  ┌──────────────────┐
        │  PostgreSQL DB    │  │   MLflow (Local) │
        │  - patient_data   │  │   - Experiments │
        │  - predictions    │  │   - Models      │
        └──────────────────┘  └──────────────────┘
```

### Flujo de Datos

1. **Frontend** → Usuario ingresa datos del paciente
2. **API Route** → Recibe request HTTP y valida con Pydantic schemas
3. **Controller** → Orquesta la lógica de negocio
4. **Service** → Preprocesa datos y carga modelo ML
5. **Modelo ML** → Genera predicción y probabilidad
6. **Database** → Guarda datos del paciente y predicción
7. **Response** → Retorna resultado al frontend

## 🛠️ Tecnologías Utilizadas

### Backend
- **FastAPI 0.109.0**: Framework web moderno y rápido
- **Python 3.11**: Lenguaje de programación
- **Uvicorn**: Servidor ASGI de alto rendimiento
- **Pydantic 2.5.3**: Validación de datos y configuración

### Machine Learning
- **scikit-learn 1.7.2**: Algoritmos ML (Logistic Regression, Random Forest)
- **XGBoost 2.0.3**: Gradient boosting avanzado
- **Optuna 3.2.0**: Optimización de hiperparámetros
- **imbalanced-learn 0.14.0**: SMOTE para balanceo de clases
- **MLflow 3.6.0**: Tracking de experimentos y modelos

### Base de Datos
- **PostgreSQL**: Base de datos relacional (producción)
- **SQLite**: Base de datos local (desarrollo)
- **SQLAlchemy 2.0.23**: ORM para gestión de base de datos
- **Alembic 1.13.1**: Migraciones de base de datos

### Data Processing
- **pandas 2.2.0**: Manipulación y análisis de datos
- **numpy 1.26.4**: Computación numérica
- **joblib 1.3.2**: Serialización de modelos

### Deployment
- **Docker**: Containerización
- **Render**: Plataforma de deployment (PaaS)
- **Nginx**: Servidor web (frontend)

## 📁 Estructura del Proyecto

```
Proyecto-IX-G3-Data-Scientist-IA-developer/
├── backend/                      # Código del backend
│   ├── controllers/              # Lógica de controladores
│   │   ├── health_controller.py
│   │   ├── predict_controller.py
│   │   ├── model_controller.py
│   │   ├── stats_controller.py
│   │   └── dataset_statistics_controller.py
│   ├── routes/                   # Definición de endpoints
│   │   ├── health.py
│   │   ├── predict.py
│   │   ├── model.py
│   │   └── stats.py
│   ├── services/                 # Lógica de negocio
│   │   ├── model_service.py      # Gestión de modelos ML
│   │   ├── preprocessing_service.py  # Preprocesamiento
│   │   ├── stats_service.py      # Estadísticas de predicciones
│   │   └── dataset_statistics_service.py  # Estadísticas del dataset
│   ├── schemas/                  # Modelos Pydantic
│   │   ├── prediction.py
│   │   ├── model.py
│   │   ├── stats.py
│   │   └── health.py
│   ├── database/                 # Configuración de BD
│   │   ├── connection.py         # Conexión SQLAlchemy
│   │   ├── models.py             # Modelos de BD
│   │   └── crud.py               # Operaciones CRUD
│   ├── data/                     # Datos preprocesados
│   │   ├── X_test_scaled.pkl
│   │   ├── y_test.pkl
│   │   └── scaler.pkl
│   ├── config.py                 # Configuración de la aplicación
│   └── main.py                   # Punto de entrada FastAPI
│
├── models/                        # Modelos ML entrenados
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model_no_smote.pkl
│   └── [archivos de resultados y parámetros]
│
├── notebooks/                    # Jupyter notebooks
│   ├── stroke_preprocessing.ipynb
│   ├── stroke_logistic_regression.ipynb
│   ├── stroke_random_forest.ipynb
│   ├── stroke_xgboost.ipynb
│   └── stroke_eda_complete.ipynb
│
├── src/                          # Datos fuente
│   └── data/
│       └── stroke_dataset.csv    # Dataset original
│
├── data/                         # Datos preprocesados (raíz)
│   ├── X_test_scaled.pkl
│   ├── y_test.pkl
│   └── scaler.pkl
│
├── tests/                        # Tests unitarios
├── Dockerfile                    # Configuración Docker
├── .render.yaml                  # Configuración Render
├── requirements.txt              # Dependencias Python
└── README.md                     # Este archivo
```

## 🔌 Endpoints de la API

### Health & Status

#### `GET /health`
Verifica el estado de salud de la API.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "version": "1.0.0"
}
```

#### `GET /status`
Obtiene información del estado del sistema y modelos.

#### `GET /control-center`
Dashboard completo de control del sistema con métricas detalladas.

### Predicciones

#### `POST /predict`
Realiza una predicción individual de riesgo de ictus.

**Request:**
```json
{
  "age": 67,
  "hypertension": 1,
  "heart_disease": 0,
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "gender": "Male",
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "smoking_status": "formerly smoked",
  "model_name": "logistic_regression"  // opcional
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.85,
  "model_used": "logistic_regression_model.pkl",
  "confidence": "High"
}
```

#### `POST /predict/batch`
Realiza predicciones por lotes (múltiples pacientes).

**Request:**
```json
{
  "data": [
    { /* paciente 1 */ },
    { /* paciente 2 */ }
  ],
  "model_name": "logistic_regression"  // opcional
}
```

### Modelos

#### `GET /models`
Lista todos los modelos disponibles.

**Response:**
```json
{
  "models": [
    "logistic_regression_model.pkl",
    "random_forest_model.pkl",
    "xgboost_model_no_smote.pkl"
  ]
}
```

#### `GET /models/{model_name}`
Obtiene información detallada de un modelo específico.

**Response incluye:**
- Métricas de rendimiento (accuracy, precision, recall, F1, ROC-AUC)
- Hiperparámetros
- Feature importance
- Matriz de confusión
- Curvas ROC y Precision-Recall
- Umbral óptimo

### Estadísticas

#### `GET /stats/overview`
Estadísticas generales de las predicciones realizadas.

#### `GET /stats/risk-distribution`
Distribución de riesgo (bajo, medio, alto).

#### `GET /stats/models/compare`
Comparación de rendimiento entre modelos.

#### `GET /dashboard`
Dashboard consolidado con toda la información relevante.

### Estadísticas del Dataset

#### `GET /statistics/overview`
Vista general del dataset original (muestras, características, balance de clases).

#### `GET /statistics/demographics`
Estadísticas demográficas (edad, género, estado civil).

#### `GET /statistics/clinical`
Estadísticas clínicas (hipertensión, enfermedad cardíaca, glucosa, BMI, tabaquismo).

#### `GET /statistics/correlations`
Matriz de correlación y factores de riesgo principales.

#### `GET /statistics/high-risk-profiles`
Perfiles de alto riesgo identificados en el dataset.

## 🗄️ Base de Datos

### PostgreSQL (Producción)

El sistema utiliza PostgreSQL en producción para almacenar:

#### Tabla: `patient_data`
Almacena los datos RAW de los pacientes (sin transformar).

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `id` | Integer | PK, autoincrement |
| `created_at` | DateTime | Timestamp automático |
| `age` | Integer | Edad del paciente |
| `gender` | String(10) | Género (Male/Female/Other) |
| `hypertension` | Boolean | Hipertensión (0/1) |
| `heart_disease` | Boolean | Enfermedad cardíaca (0/1) |
| `ever_married` | String(3) | Estado civil (Yes/No) |
| `work_type` | String(20) | Tipo de trabajo |
| `residence_type` | String(10) | Tipo de residencia (Urban/Rural) |
| `avg_glucose_level` | Float | Nivel promedio de glucosa |
| `bmi` | Float | Índice de masa corporal |
| `smoking_status` | String(20) | Estado de tabaquismo |

#### Tabla: `predictions`
Almacena los resultados de las predicciones.

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `id` | Integer | PK, autoincrement |
| `patient_data_id` | Integer | FK a `patient_data.id` |
| `created_at` | DateTime | Timestamp automático |
| `model_name` | String(50) | Modelo utilizado |
| `prediction` | Integer | Resultado (0=No stroke, 1=Stroke) |
| `probability` | Float | Probabilidad (0.0-1.0) |
| `risk_level` | String(10) | Nivel de riesgo (Low/Medium/High) |

### SQLite (Desarrollo)

Para desarrollo local, el sistema utiliza SQLite como fallback automático si `DATABASE_URL` no está configurado.

## 🤖 Modelos de Machine Learning

### Modelos Entrenados

1. **Logistic Regression** (Modelo por defecto)
   - Archivo: `logistic_regression_model.pkl`
   - **Métricas destacadas:**
     - Recall: 100% (prioritario en contexto médico)
     - Accuracy: ~85%
     - ROC-AUC: ~0.83

2. **Random Forest**
   - Archivo: `random_forest_model.pkl`
   - **Características:**
     - Feature importance disponible
     - Mejor accuracy general

3. **XGBoost**
   - Archivo: `xgboost_model_no_smote.pkl`
   - **Características:**
     - Optimizado con Optuna
     - Sin SMOTE (mejor rendimiento)

### Pipeline de Preprocesamiento

1. **Feature Engineering**
   - Categorización de edad
   - Categorización de glucosa
   - Categorización de BMI
   - Transformación de variables categóricas

2. **Encoding**
   - Label Encoding para variables categóricas
   - One-Hot Encoding donde es necesario

3. **Scaling**
   - StandardScaler para normalización

4. **Balanceo de Clases**
   - SMOTE aplicado en algunos modelos
   - Estrategia de balanceo según modelo

### Selección del Modelo

El modelo **Logistic Regression** se selecciona como predeterminado debido a:
- **100% de Recall**: Detecta todos los casos positivos (crítico en medicina)
- Interpretabilidad: Fácil de explicar a profesionales médicos
- Rendimiento estable y confiable

## 📊 MLflow

MLflow se utiliza para el tracking de experimentos y gestión del ciclo de vida de modelos ML.

### Configuración

```bash
# Iniciar MLflow UI
mlflow ui --backend-store-uri ./notebooks/mlruns \
          --default-artifact-root ./notebooks/mlruns \
          --host 0.0.0.0 \
          --port 5000
```

### Funcionalidades

- **Tracking de Experimentos**: Registro de hiperparámetros, métricas y artefactos
- **Model Registry**: Gestión de versiones de modelos
- **Reproducibilidad**: Logging completo de entornos y dependencias

### Estructura MLflow

```
notebooks/mlruns/
├── 0/                    # Experimento por defecto
│   ├── meta.yaml
│   └── [runs]/
│       ├── [run_id]/
│       │   ├── metrics/
│       │   ├── params/
│       │   └── artifacts/
│       │       └── model.pkl
```

## 🚀 Instalación y Configuración

### Requisitos Previos

- Python 3.11+
- PostgreSQL (producción) o SQLite (desarrollo)
- Git

### Instalación Local

1. **Clonar el repositorio**
```bash
git clone https://github.com/Bootcamp-IA-P5/Proyecto-IX-G3-Data-Scientist-IA-developer.git
cd Proyecto-IX-G3-Data-Scientist-IA-developer
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**
Crear archivo `.env`:
```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/stroke_db
# O dejar vacío para usar SQLite local

# Environment
ENVIRONMENT=development
DEBUG=True

# CORS (opcional, tiene valores por defecto)
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Port (opcional)
PORT=8000
```

5. **Inicializar base de datos**
```bash
python -c "from backend.database.connection import init_db; init_db()"
```

6. **Ejecutar servidor**
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

La API estará disponible en `http://localhost:8000`
Documentación interactiva en `http://localhost:8000/docs`

## 🐳 Deployment

### Docker

El proyecto incluye un `Dockerfile` optimizado para deployment:

```bash
# Build
docker build -t stroke-prediction-api .

# Run
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e CORS_ORIGINS=https://... \
  stroke-prediction-api
```

### Render

El proyecto está configurado para deployment automático en Render:

1. **Configuración en `.render.yaml`**
   - Runtime: Docker
   - Auto-deploy: Habilitado

2. **Variables de Entorno en Render Dashboard**
   - `DATABASE_URL`: URL de PostgreSQL
   - `CORS_ORIGINS`: Orígenes permitidos (comma-separated)
   - `ENVIRONMENT`: production
   - `DEBUG`: false

3. **Deployment Automático**
   - Push a `feat/deploy` → Deploy automático
   - Build usando Dockerfile
   - Health checks automáticos

## 📖 Uso de la API

### Ejemplo con cURL

```bash
# Health check
curl https://proyecto-ix-g3-data-scientist-ia-78z0.onrender.com/health

# Predicción
curl -X POST https://proyecto-ix-g3-data-scientist-ia-78z0.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 67,
    "hypertension": 1,
    "heart_disease": 0,
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "gender": "Male",
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "smoking_status": "formerly smoked"
  }'

# Listar modelos
curl https://proyecto-ix-g3-data-scientist-ia-78z0.onrender.com/models

# Información de modelo
curl https://proyecto-ix-g3-data-scientist-ia-78z0.onrender.com/models/logistic_regression_model.pkl
```

### Ejemplo con Python

```python
import requests

# Predicción
response = requests.post(
    "https://proyecto-ix-g3-data-scientist-ia-78z0.onrender.com/predict",
    json={
        "age": 67,
        "hypertension": 1,
        "heart_disease": 0,
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "gender": "Male",
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "smoking_status": "formerly smoked"
    }
)

result = response.json()
print(f"Predicción: {result['prediction']}")
print(f"Probabilidad: {result['probability']}")
print(f"Confianza: {result['confidence']}")
```

## 👥 Contribuidores

- **Backend Development**: Bootcamp IA P5 - Grupo 3
- **Frontend Development**: [Repositorio Frontend](https://github.com/Bootcamp-IA-P5/Proyecto-IX-G3-Data-Scientist-IA-developer--Frontend)

## 📝 Licencia

Este proyecto es parte del Bootcamp IA P5.

## 🔗 Enlaces Útiles

- [Documentación FastAPI](https://fastapi.tiangolo.com/)
- [Documentación MLflow](https://mlflow.org/docs/latest/index.html)
- [Documentación XGBoost](https://xgboost.readthedocs.io/)
- [Render Documentation](https://render.com/docs)

---

**Versión**: 1.0.0  
**Última actualización**: Enero 2025
