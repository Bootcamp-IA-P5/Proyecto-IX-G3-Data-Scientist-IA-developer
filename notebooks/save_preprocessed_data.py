"""
Script para guardar los datos preprocesados desde el notebook
Ejecuta esto después de ejecutar stroke_preprocessing.ipynb
"""
import pickle
import os
import sys

# Añadir path para importar desde el notebook si es necesario
sys.path.append('..')

print("="*80)
print("💾 GUARDANDO DATOS PREPROCESADOS")
print("="*80)

# Verificar que las variables existen (deben estar en el namespace del notebook)
try:
    # Intentar cargar desde el namespace del notebook
    # Si ejecutas esto desde el notebook, las variables ya existen
    from IPython import get_ipython
    ipython = get_ipython()
    
    if ipython is None:
        print("❌ Error: Este script debe ejecutarse desde el notebook stroke_preprocessing.ipynb")
        print("   O ejecuta la celda de guardado directamente en el notebook")
        sys.exit(1)
    
    # Obtener variables del namespace
    X_train_balanced = ipython.user_ns.get('X_train_balanced')
    y_train_balanced = ipython.user_ns.get('y_train_balanced')
    X_val_scaled = ipython.user_ns.get('X_val_scaled')
    y_val = ipython.user_ns.get('y_val')
    X_test_scaled = ipython.user_ns.get('X_test_scaled')
    y_test = ipython.user_ns.get('y_test')
    scaler = ipython.user_ns.get('scaler')
    
    if X_train_balanced is None:
        print("❌ Error: Variables no encontradas. Ejecuta primero todas las celdas del notebook")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("   Ejecuta la celda de guardado directamente en el notebook stroke_preprocessing.ipynb")
    sys.exit(1)

# Crear carpetas si no existen
os.makedirs('../data', exist_ok=True)
os.makedirs('../backend/data', exist_ok=True)

# Guardar datasets
datasets = {
    'X_train_balanced': X_train_balanced,
    'y_train_balanced': y_train_balanced,
    'X_val_scaled': X_val_scaled,
    'y_val': y_val,
    'X_test_scaled': X_test_scaled,
    'y_test': y_test,
    'scaler': scaler
}

print("\n📂 Guardando en data/ (raíz):")
for name, data in datasets.items():
    with open(f'../data/{name}.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f"   ✅ {name}.pkl")

print("\n📂 Guardando en backend/data/ (para el script de entrenamiento):")
for name, data in datasets.items():
    with open(f'../backend/data/{name}.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f"   ✅ {name}.pkl")

print("\n" + "="*80)
print("💾 TODOS LOS DATASETS GUARDADOS CORRECTAMENTE")
print("="*80)

