"""
Script para verificar qué experimentos hay en MLflow
"""

import mlflow
from mlflow.tracking import MlflowClient

print("="*80)
print("🔍 VERIFICANDO EXPERIMENTOS EN MLFLOW")
print("="*80)

# Configurar tracking URI
mlflow.set_tracking_uri("file:./mlruns")

client = MlflowClient()

print("\n📊 EXPERIMENTOS ENCONTRADOS:")
print("-" * 80)

try:
    experiments = client.search_experiments()
    
    if len(experiments) == 0:
        print("❌ No se encontraron experimentos")
    else:
        for exp in experiments:
            print(f"\n✅ Experimento: {exp.name}")
            print(f"   ID: {exp.experiment_id}")
            print(f"   Ubicación: {exp.artifact_location}")
            
            # Buscar runs en este experimento
            runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=10)
            print(f"   Runs encontrados: {len(runs)}")
            
            if len(runs) > 0:
                print("   Runs:")
                for i, run in enumerate(runs[:5], 1):  # Mostrar primeros 5
                    run_name = run.info.run_name if run.info.run_name else f"Run {i}"
                    print(f"      {i}. {run_name} (ID: {run.info.run_id[:8]}...)")
                    
                    # Mostrar algunos parámetros
                    if run.data.params:
                        params_sample = dict(list(run.data.params.items())[:3])
                        print(f"         Parámetros: {params_sample}")
                    
                    # Mostrar algunas métricas
                    if run.data.metrics:
                        metrics_sample = dict(list(run.data.metrics.items())[:3])
                        print(f"         Métricas: {metrics_sample}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("💡 Si no ves experimentos, verifica que mlruns/ esté en el directorio correcto")
print("="*80)

