# EXPLICACIÓN: EDA vs PREPROCESAMIENTO

## ¿Cuál es la diferencia entre los 2 archivos?

### **ARCHIVO #1: `stroke_eda_complete.ipynb`**

**¿Qué es?** Análisis Exploratorio de Datos (EDA)

**¿Qué hace?**
-  ANALIZAR y ENTENDER el dataset original
-  Ver distribuciones, correlaciones, calidad de datos
-  Identificar patrones y problemas (desbalanceo)
-  Decidir QUÉ hacer en el preprocesamiento

**Resultado:** Sabemos que:
- age es la variable MÁS importante (correlación 0.246)
- Hay desbalanceo SEVERO: 19:1 (95% sin stroke, 5% con stroke)
- gender y Residence_type NO sirven (correlación 0.009 y 0.016)
- Necesitamos crear nuevas variables y balancear clases

---

### **ARCHIVO #2: `stroke_preprocessing.ipynb`**

**¿Qué es?** Preprocesamiento de Datos

**¿Qué hace?**
- TRANSFORMAR los datos para que los modelos ML puedan aprender
- Crear 8 nuevas variables (feature engineering)
-  Eliminar variables inútiles
-  Convertir texto a números (encoding)
-  Dividir en Train/Validation/Test (60/20/20)
-  Normalizar con StandardScaler
-  **Balancear con SMOTE** (SOLO en Train)

**Resultado:** Datasets listos para entrenar modelos:
-  X_train_balanced.pkl (4,258 filas) ← **Balanceado con SMOTE**
-  X_val_scaled.pkl (996 filas) ← **SIN SMOTE**
-  X_test_scaled.pkl (997 filas) ← **SIN SMOTE**

---

## ANALOGÍA DEL ESTUDIANTE:

### **TRAIN = Material de Estudio (60%)**

Aquí **APRENDES**, puedes hacer lo que quieras para aprender mejor:

```
ANTES de SMOTE:
  Sin Stroke: 2,839 casos (95%)  ████████████████████
  Con Stroke: 149 casos (5%)     █

DESPUÉS de SMOTE:
  Sin Stroke: 2,839 casos (67%)  █████████████
  Con Stroke: 1,419 casos (33%)  ███████
```

**¿Por qué añadimos casos sintéticos?**
- El modelo necesita VER SUFICIENTES EJEMPLOS de ambas clases
- Con solo 149 casos de stroke, el modelo NO aprende bien
- SMOTE crea 1,270 casos sintéticos adicionales
- Ahora el modelo ve ~10x más ejemplos de stroke y puede aprender a detectarlos

---

###  **VALIDATION

Aquí **EVALÚAS** tu progreso antes del examen final:

```
Sin Stroke: 947 casos (95%)  ████████████████████
Con Stroke: 49 casos (5%)    █
```

**¿Por qué NO añadimos casos sintéticos?**
- ❌ Queremos ver el rendimiento REAL
- ❌ Si añadimos datos falsos, nos engañamos
- ✅ Mantenemos la distribución REAL del mundo (95% vs 5%)
- ✅ Así sabemos si el modelo funcionará en la vida real

---

### **TEST = Examen Final (20%)**

Aquí **MIDES** el rendimiento final REAL:

```
Sin Stroke: 947 casos (95%)  ████████████████████
Con Stroke: 50 casos (5%)    █
```

**¿Por qué NO añadimos casos sintéticos?**
- ❌ Es la evaluación FINAL, debe ser 100% realista
- ❌ Los datos sintéticos distorsionarían la medición
- ✅ Evaluamos con la distribución REAL del mundo
- ✅ Así sabemos el rendimiento VERDADERO que tendrá con pacientes reales

---

## **¿QUÉ PASARÍA SI USÁRAMOS SMOTE EN TODO?**

### ❌ **ESCENARIO ERRÓNEO:**

```python
# ❌ MAL - NUNCA HACER ESTO
X_train_balanced = smote(X_train)   # ✅ OK
X_val_balanced = smote(X_val)       # ❌ ERROR
X_test_balanced = smote(X_test)     # ❌ ERROR

# Resultado:
# - F1 en test: 0.85 ✅ (PARECE GENIAL pero es MENTIRA)
# - En pacientes reales: F1 = 0.40 ❌ (DESASTRE)
```

**Problema:** Estás evaluando en datos FALSOS (sintéticos), no sabes cómo funciona en la realidad.

---

### ✅ **ESCENARIO CORRECTO:**

```python
# ✅ BIEN - FORMA CORRECTA
X_train_balanced = smote(X_train)   # ✅ Solo aquí
# X_val NO se toca - distribución real 95% vs 5%
# X_test NO se toca - distribución real 95% vs 5%

# Resultado:
# - F1 en test: 0.70 ✅ (REALISTA)
# - En pacientes reales: F1 = 0.70 ✅ (COINCIDE)
```

**Beneficio:** Sabes EXACTAMENTE cómo funcionará en la vida real.

---

## **RESUMEN EN 3 PUNTOS:**

1. **SMOTE es para ENTRENAR, no para EVALUAR**
   - ✅ Uso: Solo en Train
   - ❌ NO usar: Validation ni Test

2. **Validation/Test deben ser REALES**
   - ✅ Mantienen distribución original (95% vs 5%)
   - ✅ Reflejan cómo será en pacientes reales

3. **Si modificamos Validation/Test = TRAMPA**
   - ❌ Resultados artificialmente inflados
   - ❌ No sabemos el rendimiento real
   - ❌ Modelo fallará en producción

---

## **TABLA RESUMEN:**

| Conjunto | Tamaño Original | Después SMOTE | ¿Por qué? |
|----------|----------------|---------------|-----------|
| **TRAIN** | 2,988 (95% vs 5%) | 4,258 (67% vs 33%) | Para que el modelo APRENDA con ejemplos balanceados |
| **VALIDATION** | 996 (95% vs 5%) | 996 (95% vs 5%) | Para EVALUAR en condiciones reales |
| **TEST** | 997 (95% vs 5%) | 997 (95% vs 5%) | Para medir rendimiento FINAL en datos reales |

---

# RESUMEN DEL DATASET

## Dataset Original

- **Archivo:** `stroke_dataset.csv`
- **Tamaño:** 4,981 pacientes × 11 variables
- **Calidad:** ✅ Excelente (0% nulos, 0% duplicados)
- **Problema:** ⚠️ Desbalanceo SEVERO (19:1)

## Top 5 Variables Más Importantes

| # | Variable | Correlación | Insight |
|---|----------|-------------|---------|
| 🥇 | **age** | 0.246 | Edad promedio con stroke: **67.8 años** vs sin stroke: **42.1 años** |
| 🥈 | **heart_disease** | 0.135 | Con enfermedad: **17%** stroke vs sin ella: **4%** |
| 🥉 | **avg_glucose_level** | 0.133 | Con stroke: **132 mg/dL** vs sin stroke: **104 mg/dL** |
| 4️⃣ | **hypertension** | 0.132 | Con hipertensión: **14%** stroke vs sin ella: **4%** |
| 5️⃣ | **ever_married** | 0.108 | Casados: **7%** stroke vs solteros: **2%** |

## Variables Eliminadas (No sirven)

- ❌ **gender** (correlación: 0.009) - No aporta información
- ❌ **Residence_type** (correlación: 0.016) - No aporta información

## Features Creadas (Feature Engineering)

1. **age_group** - Categorías: Child, Young_Adult, Adult, Senior
2. **glucose_category** - Normal, Prediabetes, Diabetes
3. **bmi_category** - Underweight, Normal, Overweight, Obese
4. **has_smoked** - Binaria: fumó alguna vez (sí/no)
5. **risk_score** - Score compuesto de factores de riesgo
6. **age_x_hypertension** - Interacción edad × hipertensión
7. **age_x_heart_disease** - Interacción edad × enfermedad cardíaca
8. **glucose_x_bmi** - Interacción glucosa × BMI

**Total Features Final:** 25 (de 11 originales)

## Archivos Generados

```
data/
├── X_train_balanced.pkl    # 4,258 × 25 (834 KB) - TRAIN balanceado
├── y_train_balanced.pkl    # 4,258 (34 KB)
├── X_val_scaled.pkl        # 996 × 25 (203 KB) - VALIDATION original
├── y_val.pkl               # 996 (24 KB)
├── X_test_scaled.pkl       # 997 × 25 (204 KB) - TEST original
├── y_test.pkl              # 997 (24 KB)
└── scaler.pkl              # StandardScaler (1.6 KB)
```

---

## ¿Por qué hay 7 archivos separados y en formato .pkl?

### ¿Por qué separar X (características) de y (target)?

**Regla fundamental de ML:**
> "Las características (X) se separan del objetivo (y) porque el modelo aprende de X para predecir y"

**Analogía del examen:**
- **X** = Las PREGUNTAS del examen (edad, glucosa, BMI, etc.)
- **y** = Las RESPUESTAS correctas (stroke: 0 o 1)
- El modelo ve solo X y debe adivinar y
- Luego comparamos con las respuestas correctas

**Por eso hay 6 archivos:**
- X_train + y_train (entrenamiento)
- X_val + y_val (validación)
- X_test + y_test (evaluación final)

### ¿Qué es scaler.pkl?
- Es el objeto StandardScaler usado para normalizar los datos
- Guarda la media y desviación estándar de cada variable
- Necesario para normalizar nuevos datos en producción

### ** Explicación detallada de cada archivo:**

#### ** X_train_balanced.pkl (Características de entrenamiento)**

**¿Qué contiene?**
- Las 25 variables (edad, glucosa, BMI, etc.) de cada paciente
- 4,258 filas (pacientes)
- **NO** incluye la columna `stroke`
- Tamaño: 834 KB

**¿Para qué sirve?**
- El modelo aprende de ESTAS características para predecir stroke
- Ya está balanceado con SMOTE (67% sin stroke, 33% con stroke)
- Ya está normalizado con StandardScaler

---

#### ** y_train_balanced.pkl (Target de entrenamiento)**

**¿Qué contiene?**
- SOLO la columna `stroke` (0 o 1)
- 4,258 valores
- Las "respuestas correctas" que el modelo debe aprender a predecir
- Tamaño: 34 KB

**¿Para qué sirve?**
- Es lo que el modelo intenta predecir
- Durante el entrenamiento, el modelo compara sus predicciones con estos valores
- Balanceado: 2,839 (0) + 1,419 (1) = ratio 2:1

---

#### ** X_val_scaled.pkl (Características de validación)**

**¿Qué contiene?**
- Las 25 variables de 996 pacientes
- **NO** incluye `stroke`
- **SIN** SMOTE (distribución real: 95% vs 5%)
- Tamaño: 203 KB

**¿Para qué sirve?**
- Evaluar el modelo durante el entrenamiento
- Ajustar hiperparámetros
- Ver si hay overfitting (comparando Train vs Validation)

---

#### ** y_val.pkl (Target de validación)**

**¿Qué contiene?**
- SOLO `stroke` de esos 996 pacientes
- Distribución real: 947 (0) + 49 (1) = ratio 19:1
- Tamaño: 24 KB

**¿Para qué sirve?**
- Comparar las predicciones del modelo con la realidad
- Calcular métricas (F1, Recall, Precision, etc.)
- Medir rendimiento en datos reales (no balanceados)

---

#### ** X_test_scaled.pkl (Características de test)**

**¿Qué contiene?**
- Las 25 variables de 997 pacientes
- **NO** incluye `stroke`
- **SIN** SMOTE (distribución real: 95% vs 5%)
- Tamaño: 204 KB

**¿Para qué sirve?**
- Evaluación FINAL del modelo
- **NO** se usa durante el entrenamiento
- Solo se usa al final para medir el rendimiento verdadero

---

#### ** y_test.pkl (Target de test)**

**¿Qué contiene?**
- SOLO `stroke` de esos 997 pacientes
- Distribución real: 947 (0) + 50 (1) = ratio 19:1
- Tamaño: 24 KB

**¿Para qué sirve?**
- Las respuestas correctas para la evaluación final
- Se comparan con las predicciones del modelo entrenado
- Determina el rendimiento REAL en producción

---

#### ** scaler.pkl (Normalizador)**

**¿Qué contiene?**
- El objeto StandardScaler entrenado
- Guarda la media y desviación estándar de cada una de las 25 variables
- Tamaño: 1.6 KB

**¿Para qué sirve?**
- Para normalizar datos nuevos en producción
- Ejemplo: Si llega un nuevo paciente con edad=65, el scaler lo convierte a escala normalizada
- **CRÍTICO** para que el modelo funcione correctamente con datos nuevos
- Sin esto, el modelo recibiría datos en escala diferente y fallaría

### ** Cómo cargar los archivos:**

```python
import pickle
import pandas as pd

# Cargar datos de entrenamiento
X_train = pd.read_pickle('data/X_train_balanced.pkl')
y_train = pd.read_pickle('data/y_train_balanced.pkl')

# Cargar scaler
with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print(f"X_train shape: {X_train.shape}")  # (4258, 25)
print(f"y_train shape: {y_train.shape}")  # (4258,)
```

---

## Cómo desplegar y ejecutar este repositorio

### 1) Preparar entorno virtual (recomendado)

Abre una terminal (zsh) en la raíz del repositorio y ejecuta:

```bash
# Crear entorno virtual (usa Python 3)
python3 -m venv .venv

# Activarlo (zsh)
source .venv/bin/activate

# Actualizar pip y instalar dependencias desde requirements.txt
pip install -U pip
pip install -r requirements.txt
```

### 2) Estructura de la carpeta `data` y archivos esperados

Antes de ejecutar los notebooks, la carpeta `data/` debe contener el archivo fuente original:

- `data/stroke_dataset.csv` → el CSV original con las 11 columnas del dataset (4,981 filas).

Después de ejecutar el notebook de preprocesamiento (ver más abajo), se generarán los archivos `.pkl` que el resto del flujo asume:

```
data/
├── X_train_balanced.pkl    # 4,258 × 25 (TRAIN balanceado con SMOTE)
├── y_train_balanced.pkl    # 4,258 (target train)
├── X_val_scaled.pkl        # 996 × 25 (VALIDATION sin SMOTE)
├── y_val.pkl               # 996 (target val)
├── X_test_scaled.pkl       # 997 × 25 (TEST sin SMOTE)
├── y_test.pkl              # 997 (target test)
└── scaler.pkl              # StandardScaler usado para normalizar
```


### 3) Orden recomendado para abrir/ejecutar los notebooks

1. `stroke_eda_complete.ipynb` (EDA)
    - Objetivo: entender el dataset, revisar distribuciones, correlaciones y decidir transformaciones.
    - Recomendación: ejecutar celda por celda, revisar gráficas y outputs. No debería modificar archivos en `data/` salvo que haya celdas explícitas para guardado de artefactos.

2. `stroke_preprocessing.ipynb` (Preprocesamiento)
    - Objetivo: transformar el CSV original en los `.pkl` listados arriba.
    - Antes de ejecutar: asegúrate de que `data/stroke_dataset.csv` está presente.
    - Recomendación: ejecutar las celdas en orden. El notebook realiza:
       - Feature engineering (nuevas variables)
       - Encoding y limpieza
       - División Train/Val/Test (60/20/20)
       - Escalado con StandardScaler
       - SMOTE solo en el conjunto de entrenamiento
       - Guardado de `X_*.pkl`, `y_*.pkl` y `scaler.pkl` en `data/`

3. Validación / entrenamiento de modelos
    - Una vez generados los `.pkl`, puedes usar scripts o notebooks de modelado que carguen `data/X_train_balanced.pkl`, `data/y_train_balanced.pkl`, etc.

### 4) Comandos útiles para abrir Jupyter

Con el entorno activado:

```bash
# Abrir Jupyter Lab (recomendado)
jupyter lab

# o abrir Jupyter Notebook
jupyter notebook
```

Abre los notebooks en el navegador y usa la opción "Run -> Run All Cells" si confías en el flujo; sino ejecuta celda a celda para inspeccionar resultados.

### 5) Verificaciones rápidas después del preprocesamiento

- Confirmar que `data/X_train_balanced.pkl` existe y tiene ~4258 filas.
- Confirmar que `data/X_val_scaled.pkl` y `data/X_test_scaled.pkl` existen y mantienen la distribución original (≈95% sin stroke, ≈5% con stroke).
- Confirmar que `data/scaler.pkl` existe. Este archivo es necesario para normalizar datos nuevos en producción.

