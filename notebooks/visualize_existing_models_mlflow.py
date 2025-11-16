"""
Script para visualizar modelos existentes en MLflow UI
Carga modelos ya entrenados y los registra en MLflow para visualización
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import mlflow
import mlflow.sklearn
import joblib
from datetime import datetime

print("="*80)
print("📊 VISUALIZACIÓN DE MODELOS EXISTENTES EN MLFLOW")
print("="*80)

# Configurar MLflow
mlflow.set_experiment("Existing_Models_Visualization")

# Ruta a modelos
models_path = '../models'
notebooks_path = '../notebooks'

print("\n" + "="*80)
print("🔍 BUSCANDO MODELOS EXISTENTES")
print("="*80)

# Lista de modelos disponibles
available_models = []

# 1. Random Forest (buscar con diferentes nombres)
rf_model_files = [
    f'{models_path}/random_forest_model.pkl',
    f'{models_path}/rf_model.pkl',
    f'{models_path}/rf_random_forest.pkl'
]

rf_model_file = None
for file in rf_model_files:
    if os.path.exists(file):
        rf_model_file = file
        break

if rf_model_file or os.path.exists(f'{models_path}/rf_best_params.pkl'):
    # Si no hay modelo pero hay parámetros, usar parámetros
    model_file = rf_model_file if rf_model_file else None
    
    available_models.append({
        'name': 'Random Forest',
        'model_file': model_file,
        'params_file': f'{models_path}/rf_best_params.pkl',
        'results_file': f'{models_path}/rf_results.pkl',
        'model_id': 'random_forest'
    })
    print("✅ Random Forest encontrado")

# 2. Logistic Regression
if os.path.exists(f'{models_path}/logistic_regression_model.pkl'):
    available_models.append({
        'name': 'Logistic Regression',
        'model_file': f'{models_path}/logistic_regression_model.pkl',
        'params_file': f'{models_path}/logistic_regression_best_params.pkl',
        'results_file': f'{models_path}/logistic_regression_results.pkl',
        'model_id': 'logistic_regression'
    })
    print("✅ Logistic Regression encontrado")

# 3. XGBoost
if os.path.exists(f'{models_path}/xgboost_model_no_smote.pkl'):
    available_models.append({
        'name': 'XGBoost',
        'model_file': f'{models_path}/xgboost_model_no_smote.pkl',
        'params_file': f'{models_path}/xgboost_best_params_no_smote.pkl',
        'results_file': f'{models_path}/xgboost_results_no_smote.pkl',
        'model_id': 'xgboost'
    })
    print("✅ XGBoost encontrado")

if len(available_models) == 0:
    print("❌ No se encontraron modelos en models/")
    print("   Buscado en:", models_path)
    sys.exit(1)

print(f"\n📊 Total de modelos encontrados: {len(available_models)}")

# Registrar cada modelo en MLflow
for model_info in available_models:
    print("\n" + "="*80)
    print(f"📝 REGISTRANDO: {model_info['name']}")
    print("="*80)
    
    try:
        with mlflow.start_run(run_name=f"{model_info['model_id']}_existing"):
            # Tags
            mlflow.set_tag("model_type", model_info['name'])
            mlflow.set_tag("source", "existing_model")
            mlflow.set_tag("dataset", "stroke_dataset")
            mlflow.set_tag("task", "binary_classification")
            
            # Cargar y registrar parámetros
            if os.path.exists(model_info['params_file']):
                try:
                    with open(model_info['params_file'], 'rb') as f:
                        params = pickle.load(f)
                    
                    if isinstance(params, dict):
                        for key, value in params.items():
                            if isinstance(value, (int, float, str, bool)):
                                mlflow.log_param(key, value)
                        print(f"   ✅ Parámetros registrados: {len(params)} parámetros")
                except Exception as e:
                    print(f"   ⚠️  Error cargando parámetros: {e}")
            
            # Cargar y registrar métricas
            if os.path.exists(model_info['results_file']):
                try:
                    with open(model_info['results_file'], 'rb') as f:
                        results = pickle.load(f)
                    
                    if isinstance(results, dict):
                        # Buscar métricas en diferentes estructuras
                        metrics_found = 0
                        
                        # Estructura 1: test_threshold_optimal
                        if 'test_threshold_optimal' in results:
                            test_opt = results['test_threshold_optimal']
                            if isinstance(test_opt, dict):
                                for key, value in test_opt.items():
                                    if isinstance(value, (int, float)):
                                        mlflow.log_metric(f"test_{key}", value)
                                        metrics_found += 1
                        
                        # Estructura 2: test_threshold_0.5
                        if 'test_threshold_0.5' in results:
                            test_05 = results['test_threshold_0.5']
                            if isinstance(test_05, dict):
                                for key, value in test_05.items():
                                    if isinstance(value, (int, float)):
                                        mlflow.log_metric(f"test_{key}_threshold_05", value)
                                        metrics_found += 1
                        
                        # Estructura 3: métricas directas
                        metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
                        for key in metric_keys:
                            if key in results and isinstance(results[key], (int, float)):
                                mlflow.log_metric(key, results[key])
                                metrics_found += 1
                        
                        print(f"   ✅ Métricas registradas: {metrics_found} métricas")
                except Exception as e:
                    print(f"   ⚠️  Error cargando métricas: {e}")
            
            # Cargar y registrar modelo
            if model_info['model_file'] and os.path.exists(model_info['model_file']):
                try:
                    model = joblib.load(model_info['model_file'])
                    
                    # Registrar modelo en MLflow
                    mlflow.sklearn.log_model(
                        model,
                        "model",
                        registered_model_name=f"{model_info['name']}_Stroke_Prediction"
                    )
                    print(f"   ✅ Modelo registrado en MLflow")
                except Exception as e:
                    print(f"   ⚠️  Error cargando modelo: {e}")
            else:
                print(f"   ⚠️  Archivo de modelo no encontrado (solo parámetros y métricas)")
            
            # Guardar información del modelo como artifact
            model_info_text = f"""
Modelo: {model_info['name']}
Archivo: {model_info['model_file']}
Fecha de registro: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            info_path = f'/tmp/{model_info["model_id"]}_info.txt'
            with open(info_path, 'w') as f:
                f.write(model_info_text)
            
            mlflow.log_artifact(info_path, "info")
            os.remove(info_path)
            
            print(f"   ✅ {model_info['name']} registrado exitosamente")
            
    except Exception as e:
        print(f"   ❌ Error registrando {model_info['name']}: {e}")
        continue

print("\n" + "="*80)
print("🎉 VISUALIZACIÓN COMPLETADA")
print("="*80)
print(f"\n✅ {len(available_models)} modelo(s) registrado(s) en MLflow")
print("\n📋 PRÓXIMOS PASOS:")
print("   1. Abre otra terminal")
print("   2. Ejecuta: python -m mlflow ui")
print("   3. Abre en navegador: http://localhost:5000")
print("   4. Busca el experimento: 'Existing_Models_Visualization'")
print("="*80)

