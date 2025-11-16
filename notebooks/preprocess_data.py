"""
Script completo de preprocessing para Stroke Prediction
Ejecuta todo el pipeline de preprocessing y guarda los datos procesados

Uso:
    cd notebooks
    python preprocess_data.py
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🏥 STROKE PREDICTION - PREPROCESAMIENTO COMPLETO")
print("="*80)

print("\n" + "="*80)
print("📂 PASO 1: CARGA DE DATOS")
print("="*80)

# Buscar el CSV en diferentes ubicaciones
csv_paths = [
    '../src/data/stroke_dataset.csv',
    '../data/stroke_dataset.csv',
    'src/data/stroke_dataset.csv',
    'data/stroke_dataset.csv'
]

csv_path = None
for path in csv_paths:
    if os.path.exists(path):
        csv_path = path
        break

if csv_path is None:
    print("❌ Error: No se encontró stroke_dataset.csv")
    print("   Buscado en:", csv_paths)
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"✅ Dataset cargado: {csv_path}")
print(f"   Filas: {df.shape[0]:,}")
print(f"   Columnas: {df.shape[1]}")
print(f"   Valores nulos: {df.isnull().sum().sum()}")
print(f"   Duplicados: {df.duplicated().sum()}")


print("\n" + "="*80)
print("🎯 PASO 2: FEATURE ENGINEERING")
print("="*80)

df_processed = df.copy()

# 1. Categorías de edad
df_processed['age_group'] = pd.cut(
    df_processed['age'],
    bins=[0, 18, 40, 60, 100],
    labels=['Child', 'Young_Adult', 'Adult', 'Senior']
)

# 2. Categorías de glucosa
df_processed['glucose_category'] = pd.cut(
    df_processed['avg_glucose_level'],
    bins=[0, 100, 125, 300],
    labels=['Normal', 'Prediabetes', 'Diabetes']
)

# 3. Categorías de BMI
df_processed['bmi_category'] = pd.cut(
    df_processed['bmi'],
    bins=[0, 18.5, 25, 30, 100],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)

# 4. Smoking binario
df_processed['has_smoked'] = df_processed['smoking_status'].apply(
    lambda x: 1 if x in ['smokes', 'formerly smoked'] else 0
)

# 5. Risk score compuesto
df_processed['risk_score'] = (
    df_processed['age'] * 0.3 +
    df_processed['hypertension'] * 20 +
    df_processed['heart_disease'] * 25 +
    df_processed['avg_glucose_level'] * 0.1 +
    df_processed['bmi'] * 0.5
)

# 6. Interacciones
df_processed['age_x_hypertension'] = df_processed['age'] * df_processed['hypertension']
df_processed['age_x_heart_disease'] = df_processed['age'] * df_processed['heart_disease']
df_processed['glucose_x_bmi'] = df_processed['avg_glucose_level'] * df_processed['bmi']

print("✅ Feature engineering completado")
print(f"   Nuevas features: 8")
print(f"   Total columnas: {df_processed.shape[1]}")


print("\n" + "="*80)
print("✂️ PASO 3: ELIMINACIÓN DE FEATURES IRRELEVANTES")
print("="*80)

# Eliminar features con baja correlación (según EDA)
features_to_drop = ['gender', 'Residence_type']
df_processed = df_processed.drop(columns=[f for f in features_to_drop if f in df_processed.columns])
print(f"✅ Features eliminadas: {features_to_drop}")


print("\n" + "="*80)
print("🔢 PASO 4: ENCODING DE CATEGÓRICAS")
print("="*80)

# Separar target
y = df_processed['stroke']
X = df_processed.drop('stroke', axis=1)

# Label Encoding para binarias
binary_features = ['ever_married']
le = LabelEncoder()
for feat in binary_features:
    if feat in X.columns:
        X[feat] = le.fit_transform(X[feat])

# One-Hot Encoding para categóricas
categorical_features = ['work_type', 'smoking_status', 'age_group', 'glucose_category', 'bmi_category']
categorical_features = [f for f in categorical_features if f in X.columns]

X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
print(f"✅ Encoding completado")
print(f"   Shape final: {X.shape}")

print("\n" + "="*80)
print("📊 PASO 5: TRAIN/VALIDATION/TEST SPLIT")
print("="*80)

# Split estratificado 60/20/20
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"✅ Split completado:")
print(f"   Train: {X_train.shape[0]:,} muestras")
print(f"   Validation: {X_val.shape[0]:,} muestras")
print(f"   Test: {X_test.shape[0]:,} muestras")


print("\n" + "="*80)
print("📏 PASO 6: NORMALIZACIÓN (StandardScaler)")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convertir de nuevo a DataFrame para mantener nombres de columnas
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("✅ Normalización completada")


print("\n" + "="*80)
print("⚖️ PASO 7: BALANCEO CON SMOTE")
print("="*80)

smote = SMOTE(random_state=42, sampling_strategy=0.5)  # Ratio 2:1
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"✅ SMOTE aplicado:")
print(f"   Train original: {len(y_train):,} muestras")
print(f"   Train balanceado: {len(y_train_balanced):,} muestras")
print(f"   Muestras sintéticas: {len(y_train_balanced) - len(y_train):,}")


print("\n" + "="*80)
print("💾 PASO 8: GUARDAR DATOS PROCESADOS")
print("="*80)

# Crear carpetas si no existen
os.makedirs('../data', exist_ok=True)
os.makedirs('../backend/data', exist_ok=True)

# Datos a guardar
datasets = {
    'X_train_balanced': X_train_balanced,
    'y_train_balanced': y_train_balanced,
    'X_val_scaled': X_val_scaled,
    'y_val': y_val,
    'X_test_scaled': X_test_scaled,
    'y_test': y_test,
    'scaler': scaler
}

# Guardar en ambas ubicaciones
for location in ['../data', '../backend/data']:
    print(f"\n📂 Guardando en {location}/")
    for name, data in datasets.items():
        file_path = f'{location}/{name}.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"   ✅ {name}.pkl")

print("\n" + "="*80)
print("🎉 PREPROCESAMIENTO COMPLETADO")
print("="*80)
print(f"\n📊 DATASETS FINALES:")
print(f"   Train (balanceado): {X_train_balanced.shape}")
print(f"   Validation: {X_val_scaled.shape}")
print(f"   Test: {X_test_scaled.shape}")
print(f"\n✅ Datos guardados en:")
print(f"   - data/")
print(f"   - backend/data/")
print("\n🎯 PRÓXIMO PASO: Entrenar modelos")
print("="*80)

