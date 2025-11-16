"""
Script de verificación para MLflow
Verifica que todo esté listo para ejecutar train_random_forest.py
"""

import os
import sys

print("="*80)
print("🔍 VERIFICACIÓN DE SETUP PARA MLFLOW")
print("="*80)

# 1. Verificar librerías
print("\n1️⃣ VERIFICANDO LIBRERÍAS:")
print("-" * 80)

libraries = {
    'mlflow': 'MLflow',
    'sklearn': 'Scikit-learn',
    'optuna': 'Optuna',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'joblib': 'Joblib',
    'imblearn': 'Imbalanced-learn'
}

all_ok = True
for lib, name in libraries.items():
    try:
        module = __import__(lib)
        version = getattr(module, '__version__', 'N/A')
        print(f"   ✅ {name}: {version}")
    except ImportError:
        print(f"   ❌ {name}: NO INSTALADO")
        all_ok = False

# 2. Verificar datos preprocesados
print("\n2️⃣ VERIFICANDO DATOS PREPROCESADOS:")
print("-" * 80)

data_paths = ['../backend/data', '../data']
data_path = None

for path in data_paths:
    test_file = f'{path}/X_train_balanced.pkl'
    if os.path.exists(test_file):
        data_path = path
        break

if data_path:
    print(f"   ✅ Datos encontrados en: {data_path}")
    
    required_files = [
        'X_train_balanced.pkl',
        'y_train_balanced.pkl',
        'X_val_scaled.pkl',
        'y_val.pkl',
        'X_test_scaled.pkl',
        'y_test.pkl'
    ]
    
    all_files_ok = True
    for file in required_files:
        file_path = f'{data_path}/{file}'
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"   ✅ {file} ({size:.1f} KB)")
        else:
            print(f"   ❌ {file}: NO ENCONTRADO")
            all_files_ok = False
    
    if not all_files_ok:
        all_ok = False
else:
    print("   ❌ No se encontraron datos preprocesados")
    print("   📝 Necesitas ejecutar notebooks/stroke_preprocessing.ipynb primero")
    all_ok = False

# 3. Verificar estructura de carpetas
print("\n3️⃣ VERIFICANDO ESTRUCTURA:")
print("-" * 80)

folders = {
    '../models': 'Carpeta de modelos',
    '../src/data': 'Carpeta de datos originales'
}

for folder, name in folders.items():
    if os.path.exists(folder):
        print(f"   ✅ {name}: Existe")
    else:
        print(f"   ⚠️  {name}: No existe (puede ser normal)")

# 4. Verificar MLflow puede crear experimento
print("\n4️⃣ VERIFICANDO MLFLOW:")
print("-" * 80)

try:
    import mlflow
    print(f"   ✅ MLflow version: {mlflow.__version__}")
    
    # Verificar que puede crear experimento
    try:
        mlflow.set_experiment("test_experiment")
        print("   ✅ MLflow puede crear experimentos")
    except Exception as e:
        print(f"   ⚠️  MLflow tiene problemas: {e}")
        all_ok = False
except Exception as e:
    print(f"   ❌ Error con MLflow: {e}")
    all_ok = False

# 5. Verificar script de entrenamiento
print("\n5️⃣ VERIFICANDO SCRIPT DE ENTRENAMIENTO:")
print("-" * 80)

script_path = 'train_random_forest.py'
if os.path.exists(script_path):
    print(f"   ✅ {script_path}: Existe")
    
    # Verificar que tiene imports de MLflow
    with open(script_path, 'r') as f:
        content = f.read()
        if 'import mlflow' in content:
            print("   ✅ MLflow integrado en el script")
        else:
            print("   ❌ MLflow NO está integrado en el script")
            all_ok = False
else:
    print(f"   ❌ {script_path}: NO ENCONTRADO")
    all_ok = False

# Resumen final
print("\n" + "="*80)
if all_ok:
    print("✅ TODO LISTO PARA EJECUTAR MLFLOW")
    print("\n📋 PRÓXIMOS PASOS:")
    print("   1. Ejecuta: python train_random_forest.py")
    print("   2. En otra terminal: mlflow ui")
    print("   3. Abre: http://localhost:5000")
else:
    print("⚠️  HAY PROBLEMAS QUE RESOLVER")
    print("\n📋 ACCIONES NECESARIAS:")
    if not data_path:
        print("   • Ejecuta notebooks/stroke_preprocessing.ipynb para generar datos")
    print("   • Revisa los errores arriba")
print("="*80)

sys.exit(0 if all_ok else 1)

