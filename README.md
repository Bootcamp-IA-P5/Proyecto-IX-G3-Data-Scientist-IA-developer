# 🏥 Stroke Prediction - Machine Learning Project

[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Sistema completo de predicción de accidentes cerebrovasculares (stroke) utilizando machine learning, con API REST, interfaz web y contenedorización completa.

## 📋 Tabla de Contenidos

- [🏗️ Arquitectura](#-arquitectura)
- [✨ Características](#-características)
- [🚀 Instalación](#-instalación)
- [🐳 Docker](#-docker)
- [📖 Uso de la API](#-uso-de-la-api)
- [🔧 Desarrollo](#-desarrollo)
- [📊 Modelos Disponibles](#-modelos-disponibles)
- [🤝 Contribución](#-contribución)
- [📄 Licencia](#-licencia)

## 🏗️ Arquitectura

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Nginx Proxy   │    │   Backend API   │
│   (React/TypeScript) │◄──►│   (Port 80)    │◄──►│   (FastAPI)    │
│   (Port 3000)   │    │                 │    │   (Port 8000)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
│                       │                       │
└───────────────────────┼───────────────────────┘
▼
┌─────────────────┐
│   ML Models     │
│   (Scikit-learn)│
└─────────────────┘


### Componentes

- **Backend (FastAPI)**: API REST con modelos de machine learning
- **Frontend (React/TypeScript)**: Interfaz web para predicciones
- **Nginx**: Proxy reverso y balanceo de carga
- **Docker**: Contenedorización completa del sistema

## ✨ Características

- 🔬 **Modelos de ML**: Regresión Logística, Random Forest, XGBoost
- 📊 **Preprocesamiento**: Feature engineering y normalización automática
- 🔄 **API REST**: Endpoints para predicciones individuales y batch
- 🐳 **Docker Ready**: Despliegue completo con un comando
- 📚 **Documentación**: API docs automática con Swagger/OpenAPI
- 🏥 **Médico**: Enfoque en predicción de stroke con features clínicas
- ⚡ **Alta Performance**: Modelos optimizados y cache inteligente

## 🚀 Instalación

### Prerrequisitos

- Docker y Docker Compose
- 4GB RAM mínimo
- 2GB espacio en disco

### Instalación Rápida

```bash
# Clonar repositorio
git clone <tu-repo>
cd Proyecto-IX-G3-Data-Scientist-IA-developer

# Construir y ejecutar
docker-compose up --build

### Acceder a la aplicación

- __API__: [](http://localhost:8000)<http://localhost:8000>
- __Documentación API__: [](http://localhost:8000/docs)<http://localhost:8000/docs>
- __Frontend__: [](http://localhost)<http://localhost> (cuando esté integrado)
- __Health Check__: [](http://localhost:8000/health)<http://localhost:8000/health>

## 🐳 Docker

### Servicios Disponibles
# Solo backend
docker-compose up backend

# Backend + Frontend (cuando esté disponible)
docker-compose --profile frontend up

# Producción con Nginx
docker-compose --profile production up

### Estructura de Contenedores

- __backend__: API FastAPI con modelos ML
- __frontend__: Interfaz React/TypeScript (opcional)
- __nginx__: Proxy reverso para producción

### Variables de Entorno
# Archivo .env
ENVIRONMENT=production
DEBUG=false
HOST=0.0.0.0
PORT=8000

## 📖 Uso de la API

### Predicción Individual
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'

__Respuesta:__

```
```

{
  "prediction": 1,
  "probability": 0.704,
  "model_used": "logistic_regression_model.pkl",
  "confidence": "High"
}

### Predicción Batch

```
```
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'

### Health Check

```
```
curl http://localhost:8000/health
# {"status": "healthy", "message": "API is running"}

## 🔧 Desarrollo

### Configuración del Entorno

```
```
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar en desarrollo
cd backend
uvicorn main:app --reload

### Estructura del Proyecto

```
```
Proyecto-IX-G3-Data-Scientist-IA-developer/
├── backend/                 # API FastAPI
│   ├── main.py             # Punto de entrada
│   ├── config.py           # Configuración
│   ├── routes/             # Endpoints
│   ├── controllers/        # Lógica de negocio
│   ├── services/           # Servicios (ML, preprocessing)
│   └── schemas/            # Modelos de datos
├── data/                   # Datos de entrenamiento y preprocessing
├── models/                 # Modelos entrenados (.pkl)
├── frontend/               # Interfaz React/TypeScript (futuro)
├── notebooks/              # Jupyter notebooks de análisis
├── visualizations/         # Gráficos y visualizaciones
├── docker-compose.yml      # Configuración Docker
├── requirements.txt        # Dependencias Python
└── README.md              # Esta documentación


## 📊 Modelos Disponibles

| Modelo | Archivo | Estado | Precisión | |--------|---------|--------|-----------| | Regresión Logística | `logistic_regression_model.pkl` | ✅ Activo | 85.2% | | Random Forest | `random_forest_model.pkl` | 🔄 Disponible | 87.1% | | XGBoost | `xgboost_model_no_smote.pkl` | 🔄 Disponible | 86.8% |

### Features Utilizadas

- __Demográficos__: Edad, género, estado civil
- __Clínicos__: Hipertensión, enfermedad cardíaca, nivel de glucosa
- __Antropométricos__: BMI, tipo de residencia
- __Hábitos__: Tipo de trabajo, estado de fumador
- __Ingeniería de Features__: Categorías de edad/glucosa/BMI, riesgo compuesto

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Guías de Contribución

- Sigue PEP 8 para código Python
- Añade tests para nuevas funcionalidades
- Actualiza documentación según cambios
- Usa commits descriptivos

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🙏 Agradecimientos

- Dataset: [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- Framework: [FastAPI](https://fastapi.tiangolo.com) y [Scikit-learn](https://scikit-learn.org)
- Contenedorización: [Docker](https://docker.com)

---

__Desarrollado con ❤️ por el equipo de Data Science e IA__
