# 📊 MLflow Integration - Guía Completa

## 📋 Tabla de Contenidos

1. [¿Qué es MLflow?](#qué-es-mlflow)
2. [Setup e Instalación](#setup-e-instalación)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Cómo Funciona en Este Proyecto](#cómo-funciona-en-este-proyecto)
5. [Cómo Usar MLflow](#cómo-usar-mlflow)
6. [Hacer Múltiples Experimentos](#hacer-múltiples-experimentos)
7. [Troubleshooting](#troubleshooting)
8. [Referencias](#referencias)

---

## ¿Qué es MLflow?

**MLflow** es una plataforma open-source para gestionar el ciclo de vida completo de Machine Learning. Te permite:

- ✅ **Tracking de experimentos**: Registrar parámetros, métricas y modelos
- ✅ **Comparar experimentos**: Ver qué configuraciones funcionan mejor
- ✅ **Reproducibilidad**: Guardar exactamente qué código y parámetros usaste
- ✅ **Versionado de modelos**: Gestionar diferentes versiones de tus modelos
- ✅ **Deployment**: Facilitar el despliegue de modelos en producción

### Conceptos Clave

- **Experimento (Experiment)**: Agrupa múltiples runs relacionados (ej: "Random_Forest_Stroke_Prediction")
- **Run**: Una ejecución individual del script (cada vez que entrenas)
- **Parámetros**: Valores de configuración (n_estimators, max_depth, etc.)
- **Métricas**: Resultados numéricos (accuracy, f1_score, etc.)
- **Artifacts**: Archivos guardados (gráficos, modelos, CSVs)
- **Tags**: Etiquetas para identificar y filtrar runs

---

## Setup e Instalación

### 1. Instalar MLflow

```bash
pip install mlflow
```

O desde requirements.txt:
```bash
pip install -r requirements.txt
```

### 2. Verificar Instalación

```bash
python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')"
```

### 3. Configurar .gitignore

Asegúrate de que `mlruns/` está en `.gitignore`:

```
# MLflow tracking
mlruns/
```

---

## Estructura del Proyecto

```
proyecto/
├── notebooks/
│   ├── train_random_forest.py    ← Script con MLflow integrado
│   └── mlruns/                    ← MLflow crea esto automáticamente
│       └── 0/                       ← ID del experimento
│           └── [hash]/              ← Cada run tiene su carpeta
│               ├── metrics/         ← Métricas registradas
│               ├── params/           ← Parámetros registrados
│               ├── artifacts/       ← Gráficos, modelos, etc.
│               └── meta.yaml        ← Metadata del run
├── models/                          ← Modelos para producción (pickle)
├── data/                            ← Datos preprocesados (generados por preprocessing)
└── requirements.txt                 ← Incluye mlflow
```

---

## Cómo Funciona en Este Proyecto

### Script: `notebooks/train_random_forest.py`

El script está completamente integrado con MLflow. Aquí está lo que hace:

#### 1. **Configuración del Experimento** (línea 48)
```python
mlflow.set_experiment("Random_Forest_Stroke_Prediction")
```
- Crea o selecciona el experimento
- Todos los runs irán a este experimento

#### 2. **Inicio del Run** (línea 177)
```python
with mlflow.start_run():
    # Todo el código de entrenamiento aquí
```
- Inicia un nuevo run automáticamente
- Todo lo que registres va a este run

#### 3. **Tags** (líneas 182-187)
```python
mlflow.set_tag("model_type", "RandomForest")
mlflow.set_tag("use_smote", "False")
mlflow.set_tag("dataset", "stroke_dataset")
mlflow.set_tag("task", "binary_classification")
mlflow.set_tag("target", "stroke_prediction")
```
- Identifican el tipo de experimento
- Permiten filtrar runs fácilmente

#### 4. **Registro de Parámetros** (líneas 199-201)
```python
mlflow.log_param("n_estimators", best_params['n_estimators'])
mlflow.log_param("max_depth", best_params['max_depth'])
mlflow.log_param("min_samples_split", best_params['min_samples_split'])
```
- Registra los hiperparámetros encontrados por Optuna

#### 5. **Registro de Métricas** (líneas 266-267, 389-390)
```python
mlflow.log_metric("test_accuracy", test_accuracy)
mlflow.log_metric("test_f1_score", test_f1)
mlflow.log_metric("test_accuracy_optimal", test_accuracy_opt)
mlflow.log_metric("test_f1_score_optimal", test_f1_opt)
```
- Registra métricas de evaluación
- Con threshold 0.5 y threshold óptimo

#### 6. **Guardar Artifacts** (gráficos)
```python
mlflow.log_artifact(curves_path, "plots")           # Curvas ROC/PR
mlflow.log_artifact(feature_importance_path, "plots")  # Feature importance
mlflow.log_artifact(feature_importance_csv, "data")    # CSV de features
```
- Guarda gráficos y archivos para análisis

#### 7. **Guardar Modelo** (líneas 508-513)
```python
mlflow.sklearn.log_model(
    rf_model,
    "model",
    registered_model_name="RandomForest_Stroke_Prediction"
)
```
- Guarda el modelo entrenado en MLflow
- Permite versionado y carga posterior

---

## Cómo Usar MLflow

### Paso 1: Preparar los Datos

**IMPORTANTE**: Antes de ejecutar el script, necesitas los datos preprocesados.

Ejecuta el notebook de preprocessing:
```bash
# Abre notebooks/stroke_preprocessing.ipynb
# Ejecuta todas las celdas
# Esto generará los archivos .pkl en data/
```

Los archivos necesarios:
- `data/X_train_balanced.pkl`
- `data/y_train_balanced.pkl`
- `data/X_val_scaled.pkl`
- `data/y_val.pkl`
- `data/X_test_scaled.pkl`
- `data/y_test.pkl`

### Paso 2: Ejecutar el Script

```bash
cd notebooks
python train_random_forest.py
```

**Qué verás:**
```
================================================================================
🌲 RANDOM FOREST - PREDICCIÓN DE ICTUS
================================================================================
📊 MLflow experiment: Random_Forest_Stroke_Prediction
📂 CARGA DE DATOS
✅ Datos cargados desde: ../data
...
📊 MLFLOW: Run iniciado
...
✅ Curvas guardadas en MLflow como artifact
✅ Feature importance guardada en MLflow como artifact
✅ Modelo guardado en MLflow
📊 MLFLOW: Run completado y registrado
```

### Paso 3: Abrir MLflow UI

En una **nueva terminal**, desde la raíz del proyecto:

```bash
mlflow ui
```

Verás:
```
[INFO] Starting gunicorn 20.1.0
[INFO] Listening at: http://127.0.0.1:5000
```

Abre en tu navegador: **http://localhost:5000**

### Paso 4: Explorar en MLflow UI

#### Página Principal
- Lista de experimentos
- Click en "Random_Forest_Stroke_Prediction"

#### Vista del Experimento
- Tabla con todos tus runs
- Columnas: fecha, parámetros, métricas
- Ordena por cualquier columna (click en el header)

#### Detalles de un Run
- Click en cualquier run para ver:
  - **Parámetros**: Todos los hiperparámetros usados
  - **Métricas**: Todas las métricas registradas
  - **Tags**: Etiquetas del run
  - **Artifacts**: 
    - `plots/random_forest_curves.png` - Gráficos ROC/PR
    - `plots/feature_importance_rf.png` - Importancia de features
    - `data/feature_importance.csv` - CSV con importancia
    - `model/` - Modelo completo (descargable)

#### Comparar Runs
1. Selecciona 2+ runs (checkboxes)
2. Click en "Compare"
3. Verás comparación lado a lado
4. Gráficos comparando métricas

---

## Hacer Múltiples Experimentos

### Ejemplo: Probar diferentes n_estimators

#### Método 1: Modificar el Script Manualmente

**Para probar n_estimators = 50:**

1. Modifica línea 177:
```python
with mlflow.start_run(run_name="n_estimators_50"):
```

2. Modifica línea 115 (en la función objective):
```python
'n_estimators': 50,  # Fijar en 50
# Comenta: 'n_estimators': trial.suggest_int('n_estimators', 50, 300),
```

3. Ejecuta:
```bash
python train_random_forest.py
```

4. Repite para otros valores (100, 200) cambiando el nombre y el valor

#### Método 2: Usar un Loop (Avanzado)

Puedes modificar el script para hacer múltiples runs automáticamente:

```python
n_estimators_values = [50, 100, 200]

for n_est in n_estimators_values:
    with mlflow.start_run(run_name=f"n_estimators_{n_est}"):
        mlflow.set_tag("n_estimators_test", str(n_est))
        # ... resto del código con n_estimators fijado en n_est
```

### Comparar Resultados

1. Abre MLflow UI
2. Ve a tu experimento
3. Selecciona los 3 runs (n_estimators_50, 100, 200)
4. Click en "Compare"
5. Ordena por `test_f1_score` para ver cuál es mejor

---

## Qué se Registra en MLflow

### Parámetros (3)
- `n_estimators`: Número de árboles
- `max_depth`: Profundidad máxima
- `min_samples_split`: Mínimo de muestras para split

### Métricas (4)
- `test_accuracy`: Accuracy en test (threshold 0.5)
- `test_f1_score`: F1-Score en test (threshold 0.5)
- `test_accuracy_optimal`: Accuracy con threshold óptimo
- `test_f1_score_optimal`: F1-Score con threshold óptimo

### Tags (5)
- `model_type`: "RandomForest"
- `use_smote`: "False"
- `dataset`: "stroke_dataset"
- `task`: "binary_classification"
- `target`: "stroke_prediction"

### Artifacts (4)
- `plots/random_forest_curves.png` - Curvas ROC y Precision-Recall
- `plots/feature_importance_rf.png` - Gráfico de importancia
- `data/feature_importance.csv` - CSV con importancia de features
- `model/` - Modelo entrenado completo

---

## Troubleshooting

### Error: "No module named 'mlflow'"

**Solución:**
```bash
pip install mlflow
```

### Error: "No se encontró la carpeta de datos preprocesados"

**Solución:**
1. Ejecuta `notebooks/stroke_preprocessing.ipynb`
2. O verifica que los archivos `.pkl` estén en `data/` o `backend/data/`

### Error: "mlflow ui: command not found"

**Solución:**
```bash
# Asegúrate de que MLflow está instalado
pip install mlflow

# O usa:
python -m mlflow ui
```

### No veo mi experimento en MLflow UI

**Solución:**
1. Verifica que el script terminó correctamente
2. Refresca la página (F5)
3. Verifica que estás en la carpeta correcta:
```bash
ls mlruns/  # Debe mostrar carpetas con números
```

### Los artifacts no aparecen

**Solución:**
1. Verifica que los gráficos se generaron correctamente
2. Revisa los paths en el script
3. Verifica permisos de escritura

### MLflow UI no se abre

**Solución:**
1. Verifica que el puerto 5000 no está en uso:
```bash
lsof -i :5000
```
2. Usa otro puerto:
```bash
mlflow ui --port 5001
```

---

## Comandos Útiles

### Ver experimentos desde terminal
```bash
mlflow experiments list
```

### Ver runs de un experimento
```bash
mlflow runs list --experiment-id 0
```

### Cargar modelo desde MLflow
```python
import mlflow
model = mlflow.sklearn.load_model("runs:/<run_id>/model")
```

### Exportar datos de MLflow
```bash
# Exportar a CSV
mlflow export-metrics --experiment-id 0 --output-file metrics.csv
```

---

## Estructura de mlruns/

```
mlruns/
└── 0/                          ← ID del experimento
    ├── meta.yaml               ← Metadata del experimento
    └── [hash-del-run]/         ← Cada run tiene un hash único
        ├── metrics/
        │   ├── test_accuracy
        │   ├── test_f1_score
        │   └── ...
        ├── params/
        │   ├── n_estimators
        │   ├── max_depth
        │   └── ...
        ├── tags/
        │   ├── model_type
        │   ├── use_smote
        │   └── ...
        ├── artifacts/
        │   ├── plots/
        │   │   ├── random_forest_curves.png
        │   │   └── feature_importance_rf.png
        │   ├── data/
        │   │   └── feature_importance.csv
        │   └── model/
        │       └── [archivos del modelo]
        └── meta.yaml            ← Metadata del run
```

---

## Mejores Prácticas

### 1. Nombres de Runs
- Usa nombres descriptivos: `"n_estimators_50"`, `"with_smote_v1"`
- Evita nombres genéricos como `"run1"`, `"test"`

### 2. Tags
- Usa tags consistentes para poder filtrar
- Ejemplo: siempre usa `use_smote: "True"` o `"False"` (no mezcles True/true)

### 3. Métricas
- Registra métricas en el mismo conjunto de datos (test)
- Usa nombres consistentes: `test_accuracy`, no `accuracy_test`

### 4. Artifacts
- Organiza artifacts en carpetas: `plots/`, `data/`, `models/`
- No guardes archivos muy grandes (mejor comprimir)

### 5. Experimentos
- Un experimento por tipo de modelo
- Ejemplo: "Random_Forest_Stroke", "XGBoost_Stroke", etc.

---

## Próximos Pasos

### Mejoras Futuras

1. **Tracking Remoto**
   - Configurar servidor MLflow
   - Backend store (PostgreSQL)
   - Artifact store (S3, Azure Blob)

2. **Autologging**
   - Usar `mlflow.sklearn.autolog()` para registro automático

3. **Model Registry**
   - Registrar modelos para producción
   - Gestión de versiones

4. **Integración CI/CD**
   - Registrar experimentos automáticamente
   - Comparar modelos en cada commit

---

## Referencias

- [Documentación oficial de MLflow](https://www.mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html)
- [MLflow Models](https://www.mlflow.org/docs/latest/models.html)

---

## Resumen Rápido

```bash
# 1. Instalar
pip install mlflow

# 2. Generar datos (si no existen)
# Ejecutar notebooks/stroke_preprocessing.ipynb

# 3. Entrenar modelo
cd notebooks
python train_random_forest.py

# 4. Ver resultados
cd ..
mlflow ui
# Abrir http://localhost:5000
```

---

## Script de Verificación

Antes de ejecutar el entrenamiento, puedes verificar que todo esté listo:

```bash
cd notebooks
python verify_mlflow_setup.py
```

Este script verifica:
- ✅ Todas las librerías instaladas
- ✅ Datos preprocesados disponibles
- ✅ Estructura de carpetas correcta
- ✅ MLflow funcionando
- ✅ Script de entrenamiento con MLflow integrado

---

## Checklist de Implementación

### ✅ Completado

- [x] MLflow instalado en requirements.txt
- [x] mlruns/ en .gitignore
- [x] Script train_random_forest.py con MLflow integrado
- [x] Registro de parámetros (3)
- [x] Registro de métricas (4)
- [x] Guardado de artifacts (gráficos y CSV)
- [x] Guardado de modelo en MLflow
- [x] Tags para identificar runs
- [x] Script de verificación
- [x] README completo

### ⏳ Pendiente (depende de ti)

- [ ] Ejecutar stroke_preprocessing.ipynb para generar datos
- [ ] Ejecutar train_random_forest.py
- [ ] Abrir MLflow UI y verificar resultados

---

**¿Preguntas?** Revisa la sección de Troubleshooting o consulta la documentación oficial de MLflow.


