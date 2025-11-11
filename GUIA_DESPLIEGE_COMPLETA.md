# Guía Simple del Proyecto - Reconocimiento de Comidas de Desayuno

## ¿Qué hace este proyecto?

Este proyecto es una **aplicación web** que reconoce fotos de comida de desayuno y te dice:
- **Qué comida es** (ejemplo: "pancakes", "waffles", "omelette")
- **Cuántas calorías tiene**
- **Información nutricional** (proteínas, carbohidratos, grasas)

**Es como Shazam, pero para comida de desayuno!**

---

## ¿Cómo funciona? (Explicación simple)

```
Usuario sube foto de desayuno
        |
        v
Inteligencia Artificial analiza la imagen
        |
        v
Sistema dice: "Esto es un waffle"
        |
        v
Muestra calorías y nutrición
```

---

## Estructura del Proyecto (¿Dónde está cada cosa?)

### 1. **backend/** - El cerebro
Aquí está la Inteligencia Artificial que reconoce las fotos:
- **`cnn_predictor.py`**: El programa que predice qué comida es
- **`train_cnn_model.py`**: El programa que entrena la IA
- **`main.py`**: El servidor que recibe las fotos desde la web
- **`models/`**: Aquí se guarda el cerebro entrenado (archivo `.h5`)

### 2. **frontend/** - La interfaz web
Aquí está la página web que ven los usuarios:
- **`src/pages/Home.jsx`**: La página principal donde subes fotos
- **`src/services/`**: El código que conecta con el backend
- **`src/components/`**: Botones, menús, etc.

### 3. **notebooks/** - El laboratorio
Aquí se analiza la comida antes de entrenar la IA:
- **`EDA_UNIVERSAL.ipynb`**: Análisis de las 20,987 fotos de entrenamiento
- **`data/`**: Las fotos organizadas (21 tipos de desayuno)

---

## ¿Qué comidas reconoce?

Reconoce **21 tipos de comida de desayuno**:

| Categoría | Comidas |
|-----------|---------|
| **Dulces** | Pancakes, Waffles, French Toast, Donuts, Cupcakes |
| **Postres** | Apple Pie, Cheesecake, Carrot Cake, Chocolate Cake, Churros, Cannoli, Bread Pudding, Strawberry Shortcake, Beignets |
| **Salados** | Omelette, Eggs Benedict, Huevos Rancheros, Croque Madame, Breakfast Burrito |
| **Sandwiches** | Grilled Cheese Sandwich, Club Sandwich |

**Total: 20,987 fotos** para entrenar la IA

---

## FLUJO DE DESPLIEGUE COMPLETO (Desde Cero)

### **FASE 1: Configuración del Entorno Virtual (venv)**

**Paso 1.1: Crear entorno virtual**
```bash
# Desde la raíz del proyecto
python -m venv venv
```

**Paso 1.2: Activar entorno virtual**

**En macOS/Linux:**
```bash
source venv/bin/activate
```

**En Windows:**
```bash
venv\Scripts\activate
```

Deberías ver `(venv)` al inicio de tu terminal.

**Paso 1.3: Instalar dependencias del backend**
```bash
cd backend/
pip install -r requirements.txt
```

**Instalará:**
- TensorFlow 2.20.0 (Deep Learning)
- FastAPI 0.104.1 (API REST)
- Uvicorn 0.24.0 (Servidor ASGI)
- Pillow 10.1.0 (Procesamiento de imágenes)
- NumPy 1.26.2 (Cálculos numéricos)
- Scikit-learn (Métricas)

---

### **FASE 2: Generar Dataset con Notebooks**

**Paso 2.1: Descargar imágenes de Food-101**

Descarga el dataset desde Kaggle:
- URL: https://www.kaggle.com/dansbecker/food-101
- Extrae las 21 clases de desayuno en: `notebooks/data/images_clean/`

**Estructura esperada:**
```
notebooks/data/images_clean/
├── apple_pie/        (750 imágenes)
├── beignets/         (750 imágenes)
├── breakfast_burrito/ (750 imágenes)
├── ...               (18 clases más)
└── waffles/          (750 imágenes)
```

**Paso 2.2: Ejecutar el notebook EDA_UNIVERSAL.ipynb**

```bash
cd notebooks/
jupyter notebook EDA_UNIVERSAL.ipynb
# O abrirlo en VS Code y ejecutar todas las celdas
```

**Lo que hace el notebook:**
1. Carga las 20,987 imágenes desde `images_clean/`
2. Las redimensiona a 224x224 píxeles (tamaño nativo de MobileNetV2)
3. Las convierte a formato uint8 (0-255, sin normalizar)
4. Las divide en batches y guarda en archivos NPZ
5. Genera metadata (PKL) con información del dataset
6. Crea visualizaciones y estadísticas

**Tiempo estimado:** 15-30 minutos

**Archivos generados:**
```
notebooks/data/desayuno_preprocessed/
├── food101_desayuno_preprocessed.pkl  (Metadata v2.0)
├── dataset_desayuno_summary.csv       (Resumen por clase)
├── dataset_summary.png                (Visualización)
├── load_dataset_example.py            (Ejemplo de carga)
├── stats.npy                          (Estadísticas)
└── npz_files/                         (1,050 archivos NPZ)
    ├── c000_batch0000.npz             (20 imágenes cada uno)
    ├── c000_batch0001.npz
    └── ...
```

**Paso 2.3: Verificar dataset generado**

```bash
# Verificar cantidad de archivos NPZ
ls notebooks/data/desayuno_preprocessed/npz_files/ | wc -l
# Debería mostrar: 1050

# Verificar metadata
python -c "
import pickle
with open('notebooks/data/desayuno_preprocessed/food101_desayuno_preprocessed.pkl', 'rb') as f:
    data = pickle.load(f)
print(f'Versión: {data[\"stats\"][\"version\"]}')
print(f'Clases: {len(data[\"class_names\"])}')
print(f'Imágenes: {data[\"stats\"][\"total_imagenes\"]}')
print(f'Tamaño: {data[\"stats\"][\"target_size\"]}x{data[\"stats\"][\"target_size\"]}')
"
```

**Salida esperada:**
```
Versión: 2.0
Clases: 21
Imágenes: 20987
Tamaño: 224x224
```

---

### **FASE 3: Entrenar la Red Neuronal**

**Paso 3.1: Iniciar entrenamiento**

```bash
cd backend/
python train_cnn_model.py
```

**Lo que hace el script:**
1. Carga el dataset desde `../notebooks/data/desayuno_preprocessed/`
2. Divide en Train (80%) y Validation (20%)
3. Aplica Data Augmentation (Mixup, CutMix, rotaciones, etc.)
4. Carga MobileNetV2 pre-entrenado (ImageNet)
5. Añade capas personalizadas (Dropout, Dense, Batch Normalization)
6. Entrena durante 60 épocas con Early Stopping
7. Guarda el mejor modelo en `models/breakfast_cnn_model_optimized.h5`

**Parámetros de regularización (ULTRA-OPTIMIZATION v2):**
- Dropout: 0.55 / 0.35
- L2 Regularization: 7e-4
- Label Smoothing: 0.15
- Mixup Alpha: 0.3
- CutMix Alpha: 0.3
- Early Stopping Patience: 18 épocas

**Tiempo estimado:** 2-4 horas (depende de tu GPU/CPU)

**Requisitos de hardware:**
- **RAM:** 8GB mínimo (16GB recomendado)
- **GPU:** Opcional pero recomendada (acelera 10x)
- **Disco:** 5GB libres

**Paso 3.2: Monitorear entrenamiento**

Durante el entrenamiento verás algo como:

```
Epoch 1/60
834/834 - loss: 2.3456 - accuracy: 0.4567 - val_loss: 1.9876 - val_accuracy: 0.5432
Epoch 2/60
834/834 - loss: 1.8765 - accuracy: 0.5678 - val_loss: 1.7654 - val_accuracy: 0.6123
...
Epoch 42/60
834/834 - loss: 0.4321 - accuracy: 0.8765 - val_loss: 0.6789 - val_accuracy: 0.7890
Early stopping triggered!
```

**Indicadores de buen entrenamiento:**
- Accuracy de entrenamiento: 85-92%
- Accuracy de validación: 78-83%
- Gap (overfitting): 8-12%
- Loss bajando consistentemente

**Paso 3.3: Archivos generados**

```
backend/models/
├── breakfast_cnn_model_optimized.h5   (50MB - Modelo entrenado)
├── class_names.pkl                    (Nombres de las 21 clases)
├── training_history.json              (Historial de entrenamiento)
├── training_curves.png                (Gráficas de accuracy/loss)
├── confusion_matrix.png               (Matriz de confusión)
├── confusion_matrix.csv               (Datos de confusión)
├── metrics_per_class.json             (Precision/Recall/F1 por clase)
└── error_analysis.json                (Top-10 confusiones)
```

**Paso 3.4: Verificar modelo entrenado**

```bash
# Verificar que el archivo .h5 existe
ls -lh backend/models/breakfast_cnn_model_optimized.h5

# Probar predicción con el modelo
cd backend/
python -c "
from cnn_predictor import CNNPredictor
predictor = CNNPredictor()
print('Modelo cargado correctamente!')
print(f'Clases disponibles: {len(predictor.class_names)}')
"
```

---

### **FASE 4: Desplegar Backend y Frontend**

**Paso 4.1: Arrancar el Backend** (Terminal 1)

```bash
cd backend/
python main.py
```

Deberías ver:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

**Endpoints disponibles:**
- GET `/` - Documentación de la API
- POST `/predict` - Predecir imagen (acepta multipart/form-data)
- GET `/health` - Estado del servidor
- GET `/classes` - Lista de las 21 clases

**Paso 4.2: Arrancar el Frontend** (Terminal 2)

```bash
cd frontend/
npm install          # Solo la primera vez
npm run dev
```

Deberías ver:
```
VITE v7.1.7  ready in 543 ms
➜  Local:   http://localhost:5173/
```

**Paso 4.3: Probar la aplicación**

1. Abre **http://localhost:5173** en tu navegador
2. Sube una foto de desayuno
3. Espera 2-3 segundos
4. Verás la predicción con calorías y macronutrientes

---

### **RESUMEN DEL FLUJO COMPLETO**

```
PASO 1: Configurar venv
    python -m venv venv
    source venv/bin/activate  (macOS/Linux)
    cd backend/ && pip install -r requirements.txt
        |
        v
PASO 2: Generar dataset con notebooks
    Descargar Food-101 → notebooks/data/images_clean/
    Ejecutar EDA_UNIVERSAL.ipynb
    Genera: notebooks/data/desayuno_preprocessed/
        |
        v
PASO 3: Entrenar red neuronal
    cd backend/
    python train_cnn_model.py
    Genera: models/breakfast_cnn_model_optimized.h5
        |
        v
PASO 4: Desplegar aplicación
    Terminal 1: python main.py (backend)
    Terminal 2: npm run dev (frontend)
    Abrir: http://localhost:5173
```

**Tiempo total:** 3-5 horas (la mayoría es entrenamiento)

---

## Opción Rápida (Si ya tienes el modelo entrenado)

Si ya tienes el archivo `backend/models/breakfast_cnn_model_optimized.h5`:

**Paso 1: Activar venv e instalar dependencias**
```bash
source venv/bin/activate  # O venv\Scripts\activate en Windows
cd backend/
pip install -r requirements.txt
```

**Paso 2: Arrancar Backend** (Terminal 1)
```bash
cd backend/
python main.py
```

**Paso 3: Arrancar Frontend** (Terminal 2)
```bash
cd frontend/
npm install
npm run dev
```

**Paso 4: Usar la aplicación**
- Abre http://localhost:5173
- Sube una foto de desayuno
- Verás la predicción con calorías

---

## ¿Qué tan buena es la IA?

### **Métricas actuales:**

| Métrica | Valor | ¿Qué significa? |
|---------|-------|-----------------|
| **Precisión de Entrenamiento** | 91% | De 100 fotos de entrenamiento, acierta 91 |
| **Precisión de Validación** | 78-83% | De 100 fotos nuevas, acierta 78-83 |
| **Overfitting** | 8-12% | La IA no "memoriza", generaliza bien |

**Traducción:** La IA es bastante buena! Acierta 8 de cada 10 fotos nuevas.

### **¿Cuándo se equivoca?**

La IA a veces confunde comidas que se parecen mucho:
- Pancakes <-> Waffles (ambos son redondos y tienen miel)
- Chocolate Cake <-> Bread Pudding (ambos son marrones)
- Omelette <-> Eggs Benedict (ambos tienen huevo)

**Es normal! Incluso los humanos nos confundimos a veces.**

---

## ¿Cómo se entrena la IA?

### **El proceso (simplificado):**

```
1. Descargar 20,987 fotos de comida (Food-101 dataset)
        |
        v
2. Organizar en 21 carpetas (una por cada comida)
        |
        v
3. Analizar las fotos (notebook EDA_UNIVERSAL.ipynb)
        |
        v
4. Entrenar la IA con las fotos (train_cnn_model.py)
        |
        v
5. Guardar el cerebro entrenado (archivo .h5)
        |
        v
6. Usar el cerebro para predecir nuevas fotos
```

### **¿Cuánto tarda entrenar?**

- **Tiempo:** 2-4 horas
- **Requiere:** Computadora potente (GPU recomendada)
- **Genera:** 1 archivo de 50MB (`breakfast_cnn_model_optimized.h5`)

**NOTA: Normalmente NO necesitas re-entrenar. El modelo ya está listo!**

---

## ¿Cómo se calculan las calorías?

### **Fuente de datos:**
Usamos información nutricional del **USDA** (Departamento de Agricultura de EE.UU.):
- Base de datos pública: **FoodData Central**
- Datos verificados científicamente

### **Proceso de cálculo:**

```python
# Ejemplo: Pancakes
Calorías por 100g = 227 kcal (dato USDA)
Porción estándar = 150g

Calorías totales = (227 kcal / 100g) × 150g = 340.5 kcal
```

**Nota:** Asumimos porciones de 150g (promedio estándar)

### **¿Y los macronutrientes?**

```
Proteínas = (gramos / 100g) × 150g
Carbohidratos = (gramos / 100g) × 150g
Grasas = (gramos / 100g) × 150g
```

**Ejemplo real (Pancakes):**
- Proteínas: 6.0g → 9.0g en 150g
- Carbohidratos: 34.0g → 51.0g en 150g
- Grasas: 9.0g → 13.5g en 150g

---

## La Tecnología (Explicación NO técnica)

### **¿Qué es MobileNetV2?**
Es un "cerebro pre-entrenado" que ya sabe reconocer objetos en fotos:
- Aprendió con **1,000,000 de fotos** (ImageNet dataset)
- Ya sabe detectar: formas, colores, texturas, bordes
- **Nosotros lo reutilizamos** para comida de desayuno

**Analogía:** Es como contratar a un experto en arte para que reconozca fotografías de comida. Ya sabe de imágenes, solo le enseñamos las comidas específicas.

### **¿Qué es Transfer Learning?**
En lugar de empezar desde cero, **reutilizamos conocimiento**:

```
MobileNetV2 (pre-entrenado)
    | YA SABE
    v
- Detectar bordes
- Reconocer colores
- Identificar formas
    | LE ENSEÑAMOS
    v
- Qué es un pancake
- Qué es un waffle
- Qué es un omelette
    | RESULTADO
    v
Cerebro especializado en desayunos!
```

**Ventaja:** En lugar de 1 semana de entrenamiento, tomó 3 horas

---

## ¿Qué es el "Overfitting"?

### **Problema:**
Imagina que estudias para un examen **memorizando** las preguntas exactas:
- En el examen de práctica sacas 100/100
- En el examen real sacas 60/100

**Eso es overfitting:** El modelo "memoriza" en lugar de "entender"

### **Solución aplicada:**
Usamos técnicas avanzadas para que la IA **generalice** (entienda patrones reales):

| Técnica | ¿Qué hace? | Analogía |
|---------|------------|----------|
| **Dropout** | Apaga neuronas aleatoriamente durante entrenamiento | "Estudiar con distracciones para aprender mejor" |
| **Data Augmentation** | Rota, voltea, cambia brillo de fotos | "Practicar con diferentes versiones del mismo problema" |
| **Mixup** | Mezcla dos fotos durante entrenamiento | "Estudiar casos híbridos" |
| **Early Stopping** | Para el entrenamiento cuando ya no mejora | "Dejar de estudiar cuando ya dominas el tema" |

### **Resultado:**
- **Antes:** Overfitting del 21% (memorizaba mucho)
- **Ahora:** Overfitting del 8-12% (generaliza bien)

---

## ¿Qué notebook EDA usar?

### **Respuesta:** `EDA_UNIVERSAL.ipynb` (v2.0)

**¿Por qué?**
- Es la versión actualizada (v2.0)
- Las imágenes son del tamaño correcto (224x224)
- Compatible con el modelo actual
- Contiene análisis completo de las 21 clases

**IMPORTANTE: NO uses versiones antiguas** (si encuentras archivos v1.0)

### **¿Qué hace el notebook?**

```
1. Carga 20,987 fotos de Food-101
2. Las organiza en 21 categorías
3. Las redimensiona a 224x224 píxeles
4. Calcula estadísticas (cuántas fotos por categoría)
5. Genera visualizaciones
6. Guarda todo listo para entrenar la IA
```

**Tiempo de ejecución:** 15-30 minutos

---

## Archivos importantes

### **Backend:**
- `models/breakfast_cnn_model_optimized.h5` (50MB) - El cerebro de la IA
- `models/class_names.pkl` - Los nombres de las 21 comidas
- `cnn_predictor.py` - El programa que hace predicciones

### **Frontend:**
- `src/pages/Home.jsx` - Página principal
- `src/services/PredictCaloriesServices.js` - Conecta con el backend

### **Notebooks:**
- `EDA_UNIVERSAL.ipynb` - Análisis de datos (v2.0)
- `data/desayuno_preprocessed/` - Datos procesados listos para usar

---

## Preguntas Frecuentes

### **1. ¿Necesito instalar algo?**

**Backend:**
```bash
Python 3.12
pip install tensorflow fastapi uvicorn pillow numpy
```

**Frontend:**
```bash
Node.js 18+
npm install
```

### **2. ¿Funciona en mi computadora?**

**Requisitos mínimos:**
- **RAM:** 8GB (16GB recomendado)
- **Disco:** 5GB libres
- **Sistema:** Windows, Mac o Linux
- **Python:** 3.10+ (recomendado 3.12)
- **Node.js:** 18+

### **3. ¿Puedo usar otras comidas (no solo desayuno)?**

Actualmente solo funciona con **21 tipos de desayuno** del Food-101 dataset.

Para agregar más comidas necesitarías:
1. Conseguir fotos de esas comidas (mínimo 500 por categoría)
2. Re-entrenar el modelo
3. Actualizar la base de datos nutricional

### **4. ¿Qué tan segura es la aplicación?**

- No guarda las fotos que subes
- Solo procesa la foto en memoria
- No requiere registro de usuario
- No envía datos a servidores externos

### **5. ¿Puedo desplegar esto en producción?**

**Sí, pero necesitas:**
- Un servidor (AWS, Google Cloud, Heroku, etc.)
- HTTPS (certificado SSL)
- Docker (recomendado para despliegue)
- Presupuesto para hosting (~$20-50/mes)

---

## Aprende más

### **Conceptos clave:**
- **CNN (Convolutional Neural Network):** Red neuronal para imágenes
- **Transfer Learning:** Reutilizar conocimiento pre-entrenado
- **Overfitting:** Memorizar en lugar de generalizar
- **Data Augmentation:** Aumentar variedad de datos
- **Accuracy:** Porcentaje de aciertos

### **Recursos recomendados:**
- [3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk) (YouTube)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Food-101 Dataset](https://www.kaggle.com/dansbecker/food-101)

---

## ¿Necesitas ayuda?

### **Problemas comunes:**

**"El backend no arranca"**
```bash
# Verifica que Python esté instalado
python --version

# Reinstala dependencias
cd backend/
pip install -r requirements.txt
```

**"El frontend no carga"**
```bash
# Verifica que Node.js esté instalado
node --version

# Reinstala dependencias
cd frontend/
rm -rf node_modules/
npm install
```

**"El modelo no encuentra el archivo .h5"**
```bash
# Verifica que el archivo exista
ls backend/models/breakfast_cnn_model_optimized.h5

# Si no existe, debes entrenar el modelo:
cd backend/
python train_cnn_model.py
```

---

## Resumen Final

### **¿Qué hace el proyecto?**
Reconoce 21 tipos de desayuno en fotos y calcula calorías/nutrición.

### **¿Cómo usarlo?**
1. `cd backend/ && python main.py` (Terminal 1)
2. `cd frontend/ && npm run dev` (Terminal 2)
3. Abre http://localhost:5173

### **¿Qué tan bueno es?**
- Precisión: 78-83%
- Overfitting: 8-12%
- 20,987 fotos de entrenamiento

### **¿Qué notebook usar?**
**`EDA_UNIVERSAL.ipynb` (v2.0)**

### **¿Tecnologías?**
- Backend: Python + TensorFlow + FastAPI
- Frontend: React + Vite
- IA: MobileNetV2 + Transfer Learning

---

## Equipo

Este proyecto fue desarrollado por **Grupo 2 - Ensemble**

**Versión de la Guía:** 1.0
**Última actualización:** 2024
**Documentación completa:** Ver `README.md` y `README_EXPLICATION.md`

---

**Gracias por usar nuestra aplicación!**

