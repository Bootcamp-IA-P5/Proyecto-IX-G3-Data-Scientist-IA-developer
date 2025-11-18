# 🚀 Guía de Integración para Frontend

## 📋 Resumen Ejecutivo

Se han agregado **2 nuevos endpoints principales** para mejorar el dashboard y el centro de control:

1. **`GET /dashboard`** - Dashboard consolidado con estadísticas y métricas
2. **`GET /control-center`** - Centro de control con monitoreo detallado del sistema

---

## 🎯 Endpoints Nuevos

### 1. Dashboard: `GET /dashboard`

**URL:** `http://localhost:8000/dashboard`

**Descripción:** Endpoint consolidado que devuelve toda la información necesaria para el dashboard en una sola llamada.

**Respuesta incluye:**
- Estado del sistema (API, modelos)
- Estadísticas de predicciones (total, stroke/no-stroke, probabilidad promedio)
- Distribución de riesgo (Low/Medium/High)
- **Mejor modelo destacado** (Logistic Regression con 82% Recall)
- Comparación completa de modelos (3 modelos con todas las métricas)
- Resumen de rendimiento (promedios)

**Uso recomendado:** Reemplazar múltiples llamadas a `/status`, `/stats/overview`, `/stats/models/compare` por esta única llamada.

---

### 2. Control Center: `GET /control-center`

**URL:** `http://localhost:8000/control-center`

**Descripción:** Endpoint completo para el centro de control con monitoreo detallado del sistema.

**Respuesta incluye:**
- **Componentes del sistema:**
  - API REST (estado operativo)
  - Modelo ML (modelos cargados/disponibles)
  - Servicios (estado de servicios internos)
  - Almacenamiento (uso de disco con porcentajes)
- **Salud de modelos:** Estado detallado de cada modelo (cargado, tamaño, métricas)
- **Recursos del sistema:** Almacenamiento total y por tipo
- **Métricas de rendimiento:** Total de predicciones
- **Alertas y advertencias:** Sistema automático de alertas
- **Configuración:** Environment, debug, version, directorios

**Uso recomendado:** Panel de administración/monitoreo del sistema.

---

## 📊 Estructura de Respuestas

### Dashboard Response

```typescript
interface DashboardResponse {
  api_status: string;
  models_loaded: number;
  total_models: number;
  available_models: string[];
  total_predictions: number;
  stroke_predictions: number;
  no_stroke_predictions: number;
  average_probability: number;
  risk_distribution: {
    Low: number;
    Medium: number;
    High: number;
  };
  best_model: string | null;  // "logistic_regression_model.pkl"
  best_model_type: string | null;  // "LogisticRegression"
  best_model_metrics: {
    accuracy: number;
    precision: number;
    recall: number;  // 0.82 = 82% ⭐ MÁS IMPORTANTE
    f1_score: number;
    auc_roc: number;
  } | null;
  model_comparison: {
    [modelName: string]: {
      accuracy: number;
      precision: number;
      recall: number;
      f1_score: number;
      auc_roc: number;
    };
  };
  model_performance_summary: {
    total_models: number;
    models_with_metrics: number;
    average_accuracy: number;
    average_recall: number;
    average_auc_roc: number;
  };
}
```

### Control Center Response

```typescript
interface ControlCenterResponse {
  api_status: string;
  environment: string;
  version: string;
  components: Array<{
    name: string;  // "API REST", "Modelo ML", "Servicios", "Almacenamiento"
    status: "operational" | "warning" | "error";
    percentage: number;  // 0-100
    message: string;
    details?: Record<string, any>;
  }>;
  total_models: number;
  models_loaded: number;
  models_health: Array<{
    model_name: string;
    is_loaded: boolean;
    is_available: boolean;
    file_size_mb: number | null;
    status: "available" | "loaded" | "error";
    metrics_available: boolean;
  }>;
  total_storage_mb: number;
  models_storage_mb: number;
  total_predictions: number;
  average_response_time_ms: number | null;
  alerts: string[];
  warnings: string[];
  configuration: {
    environment: string;
    debug: boolean;
    host: string;
    port: number;
    api_version: string;
    models_directory: string;
    data_directory: string;
  };
}
```

---

## 🎨 Mejoras Visuales Sugeridas

### Para Dashboard (`/dashboard`):

1. **Card del Mejor Modelo:**
   - Destacar Logistic Regression
   - Mostrar Recall: 82.0% en grande
   - Badge "🏆 Mejor Modelo"

2. **Gráficos:**
   - Gráfico de pastel: Stroke vs No Stroke
   - Gráfico de barras: Distribución de riesgo (Low/Medium/High)
   - Tabla comparativa: 3 modelos con métricas

3. **Métricas clave:**
   - Total de predicciones
   - Probabilidad promedio
   - Resumen de rendimiento

### Para Control Center (`/control-center`):

1. **Panel de Componentes:**
   - Barras de progreso con colores:
     - 🟢 Verde: `operational` (0-79%)
     - 🟡 Amarillo: `warning` (80-94%)
     - 🔴 Rojo: `error` (95-100%)

2. **Salud de Modelos:**
   - ✅ Verde: Modelo cargado
   - ⚪ Gris: Modelo disponible
   - 📊 Badge: Métricas disponibles

3. **Alertas:**
   - ⚠️ Advertencias (fondo amarillo)
   - 🚨 Alertas (fondo rojo)

---

## 🔄 Actualización en Tiempo Real

**Recomendación:** Actualizar ambos endpoints cada 5-10 segundos usando:
- Polling con `setInterval`
- O WebSockets (si se implementa en el futuro)

```typescript
// Ejemplo de polling
useEffect(() => {
  const fetchData = async () => {
    const response = await fetch('/dashboard');
    const data = await response.json();
    setDashboardData(data);
  };

  fetchData();
  const interval = setInterval(fetchData, 10000); // 10 segundos

  return () => clearInterval(interval);
}, []);
```

---

## 📝 Endpoints Existentes (Siguen Funcionando)

Los siguientes endpoints **siguen disponibles** y funcionando:

- `GET /health` - Health check simple
- `GET /status` - Estado básico del sistema
- `GET /models` - Lista de modelos
- `GET /models/{model_name}` - Info de modelo específico
- `GET /stats/overview` - Estadísticas generales
- `GET /stats/risk-distribution` - Distribución de riesgo
- `GET /stats/models/compare` - Comparación de modelos
- `POST /predict` - Predicción individual
- `POST /predict/batch` - Predicciones en lote

**Nota:** Los nuevos endpoints `/dashboard` y `/control-center` **consolidan** información de varios endpoints, pero los originales siguen funcionando para compatibilidad.

---

## ✅ Checklist de Integración

### Dashboard:
- [ ] Integrar `GET /dashboard`
- [ ] Crear componente "Best Model Card" destacando Logistic Regression
- [ ] Implementar gráficos (pastel, barras, comparación)
- [ ] Mostrar estadísticas de predicciones
- [ ] Agregar actualización automática (polling)

### Control Center:
- [ ] Integrar `GET /control-center`
- [ ] Crear panel de componentes con barras de progreso
- [ ] Implementar panel de salud de modelos
- [ ] Mostrar recursos del sistema
- [ ] Implementar sistema de alertas/advertencias
- [ ] Agregar panel de configuración
- [ ] Agregar actualización automática (polling)

---

## 📚 Documentación Completa

Para más detalles, consulta:
- **Dashboard:** `docs/DASHBOARD_IMPROVEMENTS.md`
- **Control Center:** `docs/CONTROL_CENTER.md`

---

## 🚨 Notas Importantes

1. **Mejor Modelo:** El sistema identifica automáticamente Logistic Regression como el mejor modelo basado en **Recall (82%)**, que es la métrica más importante en contexto médico.

2. **Estados de Componentes:** Los porcentajes y estados se calculan automáticamente:
   - API REST: 100% si está corriendo
   - Modelo ML: % basado en modelos cargados/total
   - Servicios: Verificación de servicios internos
   - Almacenamiento: % basado en uso de disco

3. **Alertas Automáticas:** El sistema genera alertas y advertencias automáticamente basándose en el estado actual.

4. **Compatibilidad:** Los endpoints antiguos siguen funcionando, pero se recomienda migrar a los nuevos endpoints consolidados para mejor rendimiento.

---

## 💡 Ejemplo de Uso Rápido

```typescript
// Dashboard
const Dashboard = () => {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch('/dashboard')
      .then(res => res.json())
      .then(data => setData(data));
  }, []);

  return (
    <div>
      <BestModelCard model={data?.best_model} metrics={data?.best_model_metrics} />
      <PredictionStats stats={data} />
      <ModelComparison comparison={data?.model_comparison} />
    </div>
  );
};

// Control Center
const ControlCenter = () => {
  const [data, setData] = useState(null);

  useEffect(() => {
    const fetchData = () => {
      fetch('/control-center')
        .then(res => res.json())
        .then(data => setData(data));
    };
    
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <SystemComponents components={data?.components} />
      <ModelsHealth models={data?.models_health} />
      <AlertsAndWarnings alerts={data?.alerts} warnings={data?.warnings} />
    </div>
  );
};
```

---

## 📞 Soporte

Si tienes dudas sobre la integración, consulta:
- Los READMEs en `docs/`
- La documentación de Swagger: `http://localhost:8000/docs`
- Los ejemplos de código en los READMEs

