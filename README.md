# 🍳 Food-101 Breakfast Calorie Detector

Sistema completo de detección de calorías en imágenes de desayuno utilizando Deep Learning (CNN) con Transfer Learning.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)
![React](https://img.shields.io/badge/React-18.3-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Descripción

Aplicación web completa que utiliza **Computer Vision** y **Deep Learning** para identificar automáticamente platos de desayuno en imágenes y estimar su contenido calórico y nutricional.

El sistema emplea un modelo CNN basado en **MobileNetV2** con Transfer Learning, entrenado en un subset del dataset **Food-101** (21 clases de desayunos), alcanzando una precisión del **75-80%** con overfitting controlado (<5%).

---

## ✨ Características

- ✅ **Clasificación de 21 tipos de desayunos** con Deep Learning
- ✅ **Estimación automática de calorías** y macronutrientes
- ✅ **Transfer Learning** con MobileNetV2 (ImageNet pre-trained)
- ✅ **Regularización avanzada** (Dropout, L2, Label Smoothing)
- ✅ **Data Augmentation** agresivo (9 transformaciones)
- ✅ **API REST** con FastAPI + documentación automática (Swagger)
- ✅ **Frontend moderno** con React + Vite
- ✅ **Predicción en tiempo real** (<3 segundos)
- ✅ **Top-3 predicciones** con confianza
- ✅ **Dataset preprocessado** (~21,000 imágenes, 224x224)

---

## 🛠️ Stack Tecnológico

### **Backend**
- **Python 3.12**
- **TensorFlow 2.20.0** / Keras 3.x
- **FastAPI 0.104.1** - Framework API REST
- **Uvicorn 0.24.0** - ASGI server
- **Pillow 10.1.0** - Procesamiento de imágenes
- **NumPy**, **Pandas**, **Scikit-learn** - Data Science

### **Frontend**
- **React 18.3** + **Vite 6.0**
- **JavaScript (ES6+)**
- **CSS3** - Diseño responsive
- **Fetch API** - Comunicación con backend

### **Machine Learning**
- **MobileNetV2** - Arquitectura base (ImageNet)
- **Transfer Learning** - Fine-tuning últimas 20 capas
- **Food-101 Dataset** - 21 clases de desayunos
- **Data Augmentation** - 9 transformaciones

---

## 🏗️ Arquitectura del Proyecto

```
proyecto7_ensemble_grupo2/
│
├── backend/                          # API REST + Modelo CNN
│   ├── main.py                       # FastAPI app
│   ├── cnn_predictor.py             # Predictor CNN
│   ├── train_cnn_model.py           # Script de entrenamiento
│   ├── requirements.txt             # Dependencias Python
│   ├── README_EXPLICATION.md        # Documentación técnica
│   └── models/                       # Modelos entrenados
│       ├── breakfast_cnn_model_optimized.h5    # Modelo CNN (50MB)
│       ├── class_names.pkl                     # 21 clases
│       ├── training_history.json               # Métricas
│       └── training_curves.png                 # Gráficas
│
├── frontend/                         # Interfaz web React
│   ├── src/
│   │   ├── pages/
│   │   │   └── Home.jsx             # Página principal
│   │   ├── components/              # Componentes reutilizables
│   │   ├── services/                # API calls
│   │   └── main.jsx                 # Entry point
│   ├── package.json
│   └── vite.config.js
│
├── notebooks/                        # Análisis y preprocesamiento
│   ├── EDA_UNIVERSAL.ipynb          # Análisis exploratorio
│   └── data/
│       └── desayuno_preprocessed/   # Dataset preprocessado
│           ├── food101_desayuno_preprocessed.pkl
│           └── npz_files/           # ~20,987 imágenes (224x224)
│
├── .venv/                            # Entorno virtual Python
├── GUIA_DESPLIEGUE_COMPLETA.md      # Guía de despliegue
└── README.md                         # Este archivo
```

---

## ⚡ Instalación Rápida

### **Requisitos Previos**
- Python 3.12+
- Node.js 18+
- 8GB RAM mínimo
- 2GB espacio en disco

### **1. Clonar el Repositorio**
```bash
git clone https://github.com/Factoria-F5-madrid/proyecto7_ensemble_grupo2.git
cd proyecto7_ensemble_grupo2
```

### **2. Configurar Backend**
```bash
# Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias
cd backend
pip install -r requirements.txt

# Entrenar el modelo (20-35 minutos)
python train_cnn_model.py

# Iniciar API
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### **3. Configurar Frontend**
```bash
# En otra terminal
cd frontend

# Instalar dependencias
npm install

# Configurar variable de entorno
echo "VITE_API_URL=http://localhost:8000" > .env

# Iniciar servidor de desarrollo
npm run dev
```

### **4. Acceder a la Aplicación**
- **Frontend:** http://localhost:5173
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

---

## 🚀 Uso

### **Desde la Interfaz Web**
1. Abre http://localhost:5173
2. Sube una imagen de comida (JPG/PNG)
3. Obtén la predicción con calorías estimadas

### **Desde la API (cURL)**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@pancakes.jpg"
```

**Respuesta:**
```json
{
  "success": true,
  "predicted_class": "pancakes",
  "display_name": "Pancakes",
  "confidence": 0.89,
  "estimated_calories": 340,
  "nutrition": {
    "protein": 9.3,
    "carbohydrates": 42.5,
    "fat": 15.5
  },
  "top_predictions": [
    {"class": "pancakes", "confidence": 0.89},
    {"class": "waffles", "confidence": 0.06},
    {"class": "french_toast", "confidence": 0.03}
  ]
}
```

---

## 🧠 Modelo CNN

### **Arquitectura**
```
Input (224x224x3)
    ↓
MobileNetV2 (ImageNet pre-trained)
├── Frozen: 134 capas (40.6% params)
└── Trainable: 20 capas (59.4% params)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dropout (0.5)
    ↓
Dense(256, ReLU) + L2(5e-4)
    ↓
BatchNormalization
    ↓
Dropout (0.3)
    ↓
Dense(21, Softmax) + L2(5e-4)
    ↓
Output (21 clases)
```

### **Hiperparámetros Optimizados**
- **Input Size:** 224x224x3
- **Batch Size:** 16
- **Epochs:** 30 (con Early Stopping)
- **Learning Rate:** 1e-3 → 1e-6 (Cosine Annealing + Warmup)
- **Dropout:** 0.5, 0.3
- **L2 Regularization:** 5e-4
- **Label Smoothing:** 0.2
- **Data Augmentation:** 9 técnicas

### **Parámetros del Modelo**
- **Total:** 2,597,461 parámetros
- **Entrenables:** 1,542,485 (59.4%)
- **Frozen:** 1,054,976 (40.6%)

---

## 📡 API Endpoints

### **POST `/predict`**
Predice la clase de comida y estima calorías.

**Parámetros:**
- `file` (multipart/form-data): Imagen JPG/PNG

**Respuesta:** JSON con predicción, confianza, calorías y nutrición

---

### **GET `/health`**
Verifica el estado de la API y del modelo.

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "type": "CNN - MobileNetV2",
    "num_classes": 21
  }
}
```

---

### **GET `/classes`**
Lista todas las clases soportadas.

**Respuesta:**
```json
{
  "num_classes": 21,
  "classes": ["apple_pie", "beignets", ...]
}
```

---

### **GET `/docs`**
Documentación interactiva Swagger UI.

---

## 🍳 Clases Soportadas

El modelo puede clasificar **21 tipos de desayunos**:

| Categoría | Clases |
|-----------|--------|
| **Pasteles** | apple_pie, carrot_cake, cheesecake, chocolate_cake, strawberry_shortcake |
| **Dulces fritos** | beignets, churros, donuts |
| **Postres** | bread_pudding, cannoli, cup_cakes |
| **Sándwiches** | club_sandwich, croque_madame, grilled_cheese_sandwich |
| **Desayunos calientes** | breakfast_burrito, eggs_benedict, french_toast, huevos_rancheros, omelette, pancakes, waffles |

---

## 🤝 Contribuir

Este es un proyecto educativo de **Factoría F5 Madrid**.

Para contribuir:
1. Fork el repositorio
2. Crea un branch desde `development`
3. Haz tus cambios
4. Crea un Pull Request
