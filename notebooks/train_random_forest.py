"""
Script para entrenar Random Forest con Optuna y K-Folds
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Para guardar gráficos sin display
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# Hyperparameter Optimization
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

# MLflow Tracking
import mlflow
import mlflow.sklearn

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("🌲 RANDOM FOREST - PREDICCIÓN DE ICTUS")
print("="*80)
print("\n✅ Librerías importadas correctamente")
print(f"📊 Optuna version: {optuna.__version__}")
print(f"🔢 Scikit-learn version: {__import__('sklearn').__version__}")

# ============================================================================
# CONFIGURAR MLFLOW
# ============================================================================
mlflow.set_experiment("Random_Forest_Stroke_Prediction")
print(f"📊 MLflow experiment: Random_Forest_Stroke_Prediction")

# ============================================================================
# CARGA DE DATOS
# ============================================================================
print("\n" + "="*80)
print("📂 CARGA DE DATOS")
print("="*80)

try:
    with open('../data/X_train_balanced.pkl', 'rb') as f:
        X_train_balanced = pickle.load(f)
    with open('../data/y_train_balanced.pkl', 'rb') as f:
        y_train_balanced = pickle.load(f)
    with open('../data/X_val_scaled.pkl', 'rb') as f:
        X_val_scaled = pickle.load(f)
    with open('../data/y_val.pkl', 'rb') as f:
        y_val = pickle.load(f)
    with open('../data/X_test_scaled.pkl', 'rb') as f:
        X_test_scaled = pickle.load(f)
    with open('../data/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    print("✅ Datos cargados desde archivos pickle")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print("\n📊 RESUMEN DE DATOS:")
print(f"   Train: {X_train_balanced.shape[0]:,} muestras, {X_train_balanced.shape[1]} features")
print(f"   Validation: {X_val_scaled.shape[0]:,} muestras, {X_val_scaled.shape[1]} features")
print(f"   Test: {X_test_scaled.shape[0]:,} muestras, {X_test_scaled.shape[1]} features")
print(f"\n   Train balanceado - Stroke 0: {(y_train_balanced == 0).sum():,}, Stroke 1: {(y_train_balanced == 1).sum():,}")
print(f"   Validation original - Stroke 0: {(y_val == 0).sum():,}, Stroke 1: {(y_val == 1).sum():,}")
print(f"   Test original - Stroke 0: {(y_test == 0).sum():,}, Stroke 1: {(y_test == 1).sum():,}")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
print("\n" + "="*80)
print("⚙️ CONFIGURACIÓN")
print("="*80)

N_FOLDS = 5
RANDOM_STATE = 42
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

print(f"✅ Configuración:")
print(f"   • K-Folds: {N_FOLDS}")
print(f"   • Random State: {RANDOM_STATE}")
print(f"   • Métrica objetivo: F1-Score (balance entre Precision y Recall)")
print(f"   • Métricas adicionales: Recall, AUC-ROC")

# ============================================================================
# FUNCIÓN OBJETIVO PARA OPTUNA
# ============================================================================
print("\n" + "="*80)
print("🎯 FUNCIÓN OBJETIVO PARA OPTUNA")
print("="*80)

def objective(trial):
    """
    Función objetivo para Optuna.
    Optimiza hiperparámetros de Random Forest usando K-Folds CV.
    """
    # Hiperparámetros a optimizar
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': 'balanced',  # Balanceo automático de clases
        'random_state': RANDOM_STATE,
        'n_jobs': -1  # Usar todos los cores
    }
    
    # Crear modelo
    model = RandomForestClassifier(**params)
    
    # K-Folds Cross-Validation con F1-Score
    cv_scores = cross_val_score(
        model, 
        X_train_balanced, 
        y_train_balanced,
        cv=skf,
        scoring='f1',
        n_jobs=-1
    )
    
    # Retornar el promedio de F1-Score
    return cv_scores.mean()

print("✅ Función objetivo definida")
print("   Optimizando: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap")

# ============================================================================
# OPTIMIZACIÓN CON OPTUNA
# ============================================================================
print("\n" + "="*80)
print("🚀 OPTIMIZACIÓN CON OPTUNA")
print("="*80)
print(f"📊 Datos: {X_train_balanced.shape[0]:,} muestras, {X_train_balanced.shape[1]} features")
print(f"🔄 K-Folds: {N_FOLDS}")
print(f"⏱️  Esto puede tomar varios minutos...\n")

# Crear estudio de Optuna
study = optuna.create_study(
    direction='maximize',  # Maximizar F1-Score
    study_name='random_forest_stroke_prediction',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
)

# Ejecutar optimización
N_TRIALS = 50  # Número de trials
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\n" + "="*80)
print("✅ OPTIMIZACIÓN COMPLETADA")
print("="*80)
print(f"\n🏆 MEJOR F1-SCORE: {study.best_value:.4f}")
print(f"\n📋 MEJORES HIPERPARÁMETROS:")
for key, value in study.best_params.items():
    print(f"   • {key}: {value}")

# ============================================================================
# INICIAR MLFLOW RUN
# ============================================================================
with mlflow.start_run():
    print("\n" + "="*80)
    print("📊 MLFLOW: Run iniciado")
    print("="*80)

    # ============================================================================
    # ENTRENAR MODELO FINAL
    # ============================================================================
    print("\n" + "="*80)
    print("🎓 ENTRENAR MODELO FINAL")
    print("="*80)

    # Obtener mejores parámetros
    best_params = study.best_params.copy()
    best_params['class_weight'] = 'balanced'
    best_params['random_state'] = RANDOM_STATE
    best_params['n_jobs'] = -1

    print("🎓 ENTRENANDO MODELO FINAL...")
    print(f"📋 Parámetros: {best_params}\n")
    
    # Registrar parámetros en MLflow
    mlflow.log_param("n_estimators", best_params['n_estimators'])
    mlflow.log_param("max_depth", best_params['max_depth'])
    mlflow.log_param("min_samples_split", best_params['min_samples_split'])

    # Crear y entrenar modelo
    rf_model = RandomForestClassifier(**best_params)
    rf_model.fit(X_train_balanced, y_train_balanced)

    print("✅ Modelo entrenado correctamente")
    print(f"   • Número de árboles: {rf_model.n_estimators}")
    print(f"   • Features importantes: {rf_model.n_features_in_}")

    # ============================================================================
    # EVALUACIÓN EN VALIDATION SET
    # ============================================================================
    print("\n" + "="*80)
    print("📊 EVALUACIÓN EN VALIDATION SET")
    print("="*80)

    # Predicciones
    y_val_pred = rf_model.predict(X_val_scaled)
    y_val_pred_proba = rf_model.predict_proba(X_val_scaled)[:, 1]

    # Métricas
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_pred_proba)

    print(f"\n📊 MÉTRICAS:")
    print(f"   Accuracy:  {val_accuracy:.4f}")
    print(f"   Precision: {val_precision:.4f}")
    print(f"   Recall:    {val_recall:.4f} ⭐ (MÉTRICA PRINCIPAL)")
    print(f"   F1-Score:  {val_f1:.4f} ⭐")
    print(f"   AUC-ROC:   {val_auc:.4f} ⭐")

    print(f"\n📋 MATRIZ DE CONFUSIÓN:")
    cm_val = confusion_matrix(y_val, y_val_pred)
    print(cm_val)
    print(f"\n   Verdaderos Negativos: {cm_val[0,0]}")
    print(f"   Falsos Positivos:     {cm_val[0,1]}")
    print(f"   Falsos Negativos:     {cm_val[1,0]} ⚠️  (CRÍTICO - pacientes en riesgo no detectados)")
    print(f"   Verdaderos Positivos: {cm_val[1,1]}")

    print(f"\n📄 REPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_val, y_val_pred, target_names=['No Stroke', 'Stroke']))

    # ============================================================================
    # EVALUACIÓN EN TEST SET
    # ============================================================================
    print("\n" + "="*80)
    print("🧪 EVALUACIÓN EN TEST SET (DATOS FINALES)")
    print("="*80)

    # Predicciones
    y_test_pred = rf_model.predict(X_test_scaled)
    y_test_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

    # Métricas
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    # Registrar métricas de test (threshold 0.5) en MLflow
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1_score", test_f1)

    print(f"\n📊 MÉTRICAS FINALES:")
    print(f"   Accuracy:  {test_accuracy:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall:    {test_recall:.4f} ⭐ (MÉTRICA PRINCIPAL)")
    print(f"   F1-Score:  {test_f1:.4f} ⭐")
    print(f"   AUC-ROC:   {test_auc:.4f} ⭐")

    print(f"\n📋 MATRIZ DE CONFUSIÓN:")
    cm_test = confusion_matrix(y_test, y_test_pred)
    print(cm_test)
    print(f"\n   Verdaderos Negativos: {cm_test[0,0]}")
    print(f"   Falsos Positivos:     {cm_test[0,1]}")
    print(f"   Falsos Negativos:     {cm_test[1,0]} ⚠️  (CRÍTICO)")
    print(f"   Verdaderos Positivos: {cm_test[1,1]}")

    print(f"\n📄 REPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_test, y_test_pred, target_names=['No Stroke', 'Stroke']))

    # ============================================================================
    # AJUSTE DE THRESHOLD ÓPTIMO
    # ============================================================================
    print("\n" + "="*80)
    print("🎯 AJUSTE DE THRESHOLD ÓPTIMO")
    print("="*80)
    print("Buscando threshold que maximice Recall (mínimo 0.70) manteniendo F1 razonable...\n")

    # Probar diferentes thresholds en validation set
    thresholds = np.arange(0.1, 0.6, 0.05)
    best_threshold = 0.5
    best_recall = 0
    best_f1 = 0
    best_metrics = None

    results_threshold = []

    for threshold in thresholds:
        # Predicciones con threshold ajustado
        y_val_pred_thresh = (y_val_pred_proba >= threshold).astype(int)
        
        # Calcular métricas
        recall_thresh = recall_score(y_val, y_val_pred_thresh)
        precision_thresh = precision_score(y_val, y_val_pred_thresh)
        f1_thresh = f1_score(y_val, y_val_pred_thresh)
        
        results_threshold.append({
            'threshold': threshold,
            'recall': recall_thresh,
            'precision': precision_thresh,
            'f1': f1_thresh
        })
        
        # Buscar threshold que maximice Recall (objetivo: >0.70) con F1 razonable (>0.40)
        if recall_thresh >= 0.70 and f1_thresh > best_f1:
            best_threshold = threshold
            best_recall = recall_thresh
            best_f1 = f1_thresh
            best_metrics = {
                'recall': recall_thresh,
                'precision': precision_thresh,
                'f1': f1_thresh
            }

    # Si no encontramos threshold con Recall >0.70, usar el que maximice F1 con Recall >0.50
    if best_recall < 0.70:
        print("⚠️  No se encontró threshold con Recall >0.70. Buscando mejor compromiso...\n")
        for result in results_threshold:
            if result['recall'] >= 0.50 and result['f1'] > best_f1:
                best_threshold = result['threshold']
                best_recall = result['recall']
                best_f1 = result['f1']
                best_metrics = {
                    'recall': result['recall'],
                    'precision': result['precision'],
                    'f1': result['f1']
                }

    print(f"✅ THRESHOLD ÓPTIMO ENCONTRADO: {best_threshold:.3f}")
    print(f"   Validation - Recall: {best_metrics['recall']:.4f}, Precision: {best_metrics['precision']:.4f}, F1: {best_metrics['f1']:.4f}")

    # Tabla de resultados
    print(f"\n📊 RESULTADOS POR THRESHOLD (Top 10 mejores F1):")
    df_threshold = pd.DataFrame(results_threshold).sort_values('f1', ascending=False)
    print(df_threshold.head(10).to_string(index=False))

    # ============================================================================
    # RE-EVALUACIÓN CON THRESHOLD ÓPTIMO
    # ============================================================================
    print("\n" + "="*80)
    print("🔄 RE-EVALUACIÓN CON THRESHOLD ÓPTIMO")
    print("="*80)

    # Validation con threshold óptimo
    y_val_pred_optimal = (y_val_pred_proba >= best_threshold).astype(int)
    val_recall_opt = recall_score(y_val, y_val_pred_optimal)
    val_precision_opt = precision_score(y_val, y_val_pred_optimal)
    val_f1_opt = f1_score(y_val, y_val_pred_optimal)
    val_accuracy_opt = accuracy_score(y_val, y_val_pred_optimal)

    print(f"\n📊 VALIDATION SET (Threshold = {best_threshold:.3f}):")
    print(f"   Accuracy:  {val_accuracy_opt:.4f}")
    print(f"   Precision: {val_precision_opt:.4f}")
    print(f"   Recall:    {val_recall_opt:.4f} ⭐ (MÉTRICA PRINCIPAL)")
    print(f"   F1-Score:  {val_f1_opt:.4f} ⭐")

    cm_val_opt = confusion_matrix(y_val, y_val_pred_optimal)
    print(f"\n📋 MATRIZ DE CONFUSIÓN:")
    print(cm_val_opt)
    print(f"\n   Verdaderos Negativos: {cm_val_opt[0,0]}")
    print(f"   Falsos Positivos:     {cm_val_opt[0,1]}")
    print(f"   Falsos Negativos:     {cm_val_opt[1,0]} ⚠️  (CRÍTICO)")
    print(f"   Verdaderos Positivos: {cm_val_opt[1,1]}")

    # Test con threshold óptimo
    y_test_pred_optimal = (y_test_pred_proba >= best_threshold).astype(int)
    test_recall_opt = recall_score(y_test, y_test_pred_optimal)
    test_precision_opt = precision_score(y_test, y_test_pred_optimal)
    test_f1_opt = f1_score(y_test, y_test_pred_optimal)
    test_accuracy_opt = accuracy_score(y_test, y_test_pred_optimal)
    
    # Registrar métricas de test con threshold óptimo en MLflow
    mlflow.log_metric("test_accuracy_optimal", test_accuracy_opt)
    mlflow.log_metric("test_f1_score_optimal", test_f1_opt)

    print(f"\n📊 TEST SET (Threshold = {best_threshold:.3f}):")
    print(f"   Accuracy:  {test_accuracy_opt:.4f}")
    print(f"   Precision: {test_precision_opt:.4f}")
    print(f"   Recall:    {test_recall_opt:.4f} ⭐ (MÉTRICA PRINCIPAL)")
    print(f"   F1-Score:  {test_f1_opt:.4f} ⭐")

    cm_test_opt = confusion_matrix(y_test, y_test_pred_optimal)
    print(f"\n📋 MATRIZ DE CONFUSIÓN:")
    print(cm_test_opt)
    print(f"\n   Verdaderos Negativos: {cm_test_opt[0,0]}")
    print(f"   Falsos Positivos:     {cm_test_opt[0,1]}")
    print(f"   Falsos Negativos:     {cm_test_opt[1,0]} ⚠️  (CRÍTICO)")
    print(f"   Verdaderos Positivos: {cm_test_opt[1,1]}")

    print(f"\n📄 REPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_test, y_test_pred_optimal, target_names=['No Stroke', 'Stroke']))

    # ============================================================================
    # VISUALIZACIONES
    # ============================================================================
    print("\n" + "="*80)
    print("📊 GENERANDO VISUALIZACIONES")
    print("="*80)

    # Curvas ROC y Precision-Recall
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. CURVA ROC
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba)

    axes[0].plot(fpr_val, tpr_val, label=f'Validation (AUC = {val_auc:.3f})', linewidth=2)
    axes[0].plot(fpr_test, tpr_test, label=f'Test (AUC = {test_auc:.3f})', linewidth=2)
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve - Random Forest', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 2. CURVA PRECISION-RECALL
    precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_pred_proba)
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred_proba)

    axes[1].plot(recall_val, precision_val, label=f'Validation (F1 = {val_f1:.3f})', linewidth=2)
    axes[1].plot(recall_test, precision_test, label=f'Test (F1 = {test_f1:.3f})', linewidth=2)
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve - Random Forest', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('../', exist_ok=True)
    plt.savefig('../random_forest_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Curvas guardadas: backend/random_forest_curves.png")

    # Importancia de Features
    feature_importance = pd.DataFrame({
        'feature': X_train_balanced.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n" + "="*80)
    print("🌳 TOP 15 FEATURES MÁS IMPORTANTES")
    print("="*80)
    print(feature_importance.head(15).to_string(index=False))

    # Visualización
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importancia', fontsize=12)
    plt.title('Top 15 Features más Importantes - Random Forest', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('../feature_importance_rf.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✅ Visualización guardada: backend/feature_importance_rf.png")

    # ============================================================================
    # GUARDAR MODELO
    # ============================================================================
    print("\n" + "="*80)
    print("💾 GUARDANDO MODELO Y RESULTADOS")
    print("="*80)

    import joblib

    # Crear carpeta models si no existe
    os.makedirs('../models', exist_ok=True)

    # Guardar modelo
    joblib.dump(rf_model, '../models/random_forest_model.pkl')
    print("✅ Modelo guardado: models/random_forest_model.pkl")

    # Guardar mejores parámetros
    with open('../models/rf_best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    print("✅ Mejores parámetros guardados: models/rf_best_params.pkl")

    # Guardar resultados de evaluación
    results = {
        'validation_threshold_0.5': {
            'accuracy': val_accuracy,
            'precision': val_precision,
            'recall': val_recall,
            'f1_score': val_f1,
            'auc_roc': val_auc
        },
        'test_threshold_0.5': {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'auc_roc': test_auc
        },
        'validation_threshold_optimal': {
            'threshold': float(best_threshold),
            'accuracy': val_accuracy_opt,
            'precision': val_precision_opt,
            'recall': val_recall_opt,
            'f1_score': val_f1_opt,
            'auc_roc': val_auc
        },
        'test_threshold_optimal': {
            'threshold': float(best_threshold),
            'accuracy': test_accuracy_opt,
            'precision': test_precision_opt,
            'recall': test_recall_opt,
            'f1_score': test_f1_opt,
            'auc_roc': test_auc
        },
        'best_params': best_params,
        'optimal_threshold': float(best_threshold),
        'feature_importance': feature_importance.to_dict('records')
    }

    with open('../models/rf_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("✅ Resultados guardados: models/rf_results.pkl")

    print("\n" + "="*80)
    print("🎉 ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print(f"\n📊 RESUMEN FINAL (Threshold = 0.5):")
    print(f"   Validation - Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
    print(f"   Test       - Recall: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

    print(f"\n📊 RESUMEN FINAL (Threshold Óptimo = {best_threshold:.3f}):")
    print(f"   Validation - Recall: {val_recall_opt:.4f}, F1: {val_f1_opt:.4f}, AUC: {val_auc:.4f}")
    print(f"   Test       - Recall: {test_recall_opt:.4f}, F1: {test_f1_opt:.4f}, AUC: {test_auc:.4f}")

    print(f"\n✅ Modelo listo para usar en producción")
    print(f"✅ Threshold óptimo recomendado: {best_threshold:.3f}")
    print("="*80)
    
    print("\n📊 MLFLOW: Run completado y registrado")
    print("="*80)

