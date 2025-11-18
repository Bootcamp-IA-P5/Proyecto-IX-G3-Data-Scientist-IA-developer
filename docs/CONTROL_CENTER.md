# 🎛️ Centro de Control - Guía para Frontend

## 📊 Nuevo Endpoint: `GET /control-center`

Este endpoint proporciona información completa y detallada del sistema para el centro de control, incluyendo estado de componentes, salud de modelos, recursos del sistema, alertas y configuración.

---

## 📋 Estructura de la Respuesta

```json
{
  "api_status": "running",
  "environment": "development",
  "version": "1.0.0",
  "components": [
    {
      "name": "API REST",
      "status": "operational",
      "percentage": 100,
      "message": "API funcionando correctamente"
    },
    {
      "name": "Modelo ML",
      "status": "warning",
      "percentage": 0,
      "message": "0 de 3 modelos cargados",
      "details": {
        "models_loaded": 0,
        "total_models": 3
      }
    },
    {
      "name": "Servicios",
      "status": "operational",
      "percentage": 100,
      "message": "Todos los servicios operativos"
    },
    {
      "name": "Almacenamiento",
      "status": "operational",
      "percentage": 28,
      "message": "28.58 MB utilizados",
      "details": {
        "total_mb": 28.58,
        "models_mb": 13.64
      }
    }
  ],
  "total_models": 3,
  "models_loaded": 0,
  "models_health": [
    {
      "model_name": "logistic_regression_model.pkl",
      "is_loaded": false,
      "is_available": true,
      "file_size_mb": null,
      "status": "available",
      "metrics_available": true
    },
    {
      "model_name": "random_forest_model.pkl",
      "is_loaded": false,
      "is_available": true,
      "file_size_mb": 13.59,
      "status": "available",
      "metrics_available": true
    },
    {
      "model_name": "xgboost_model_no_smote.pkl",
      "is_loaded": false,
      "is_available": true,
      "file_size_mb": 0.05,
      "status": "available",
      "metrics_available": true
    }
  ],
  "total_storage_mb": 28.58,
  "models_storage_mb": 13.64,
  "total_predictions": 0,
  "average_response_time_ms": null,
  "alerts": [],
  "warnings": [
    "Ningún modelo está cargado en memoria",
    "No se han realizado predicciones aún"
  ],
  "configuration": {
    "environment": "development",
    "debug": false,
    "host": "0.0.0.0",
    "port": 8000,
    "api_version": "1.0.0",
    "models_directory": "/path/to/models",
    "data_directory": "/path/to/data"
  }
}
```

---

## 🎯 Mejoras Sugeridas para el Centro de Control

### 1. **Panel de Componentes del Sistema**

**Datos a mostrar:** `components`

**Diseño sugerido:**
```
┌─────────────────────────────────────────────────────────┐
│  Estado del Sistema                                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ✅ API REST                    [████████████] 100%     │
│     API funcionando correctamente                        │
│                                                          │
│  ⚠️ Modelo ML                  [          ] 0%          │
│     0 de 3 modelos cargados                             │
│                                                          │
│  ✅ Servicios                  [████████████] 100%      │
│     Todos los servicios operativos                      │
│                                                          │
│  ✅ Almacenamiento            [███        ] 28%         │
│     28.58 MB utilizados                                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Características:**
- Barra de progreso visual con colores:
  - 🟢 Verde: `operational` (0-79%)
  - 🟡 Amarillo: `warning` (80-94%)
  - 🔴 Rojo: `error` (95-100%)
- Iconos de estado (✅/⚠️/❌)
- Porcentaje y mensaje descriptivo
- Detalles adicionales al hacer hover/click

---

### 2. **Panel de Salud de Modelos**

**Datos a mostrar:** `models_health`

**Diseño sugerido:**
```
┌─────────────────────────────────────────────────────────┐
│  Modelos Disponibles                                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ⚪ 📊 logistic_regression_model.pkl                     │
│     Status: available | Métricas: ✅                     │
│                                                          │
│  ⚪ 📊 random_forest_model.pkl (13.59 MB)               │
│     Status: available | Métricas: ✅                     │
│                                                          │
│  ⚪ 📊 xgboost_model_no_smote.pkl (0.05 MB)              │
│     Status: available | Métricas: ✅                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Características:**
- **Iconos de estado:**
  - ✅ Verde: Modelo cargado (`is_loaded: true`)
  - ⚪ Blanco/Gris: Modelo disponible pero no cargado
  - ❌ Rojo: Modelo con error
- **Badge de métricas:** 📊 si `metrics_available: true`
- **Tamaño del archivo:** Mostrar en MB si está disponible
- **Click para ver detalles:** Abrir modal con información completa del modelo

---

### 3. **Panel de Recursos del Sistema**

**Datos a mostrar:** `total_storage_mb`, `models_storage_mb`

**Diseño sugerido:**
```
┌─────────────────────────────────────────────────────────┐
│  Recursos del Sistema                                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Almacenamiento Total                                    │
│  [████████████████████░░░░░░░░] 28.58 MB                │
│                                                          │
│  Almacenamiento de Modelos                               │
│  [████████████░░░░░░░░░░░░░░░░] 13.64 MB                │
│                                                          │
│  Almacenamiento de Datos                                │
│  [████████░░░░░░░░░░░░░░░░░░░░] 14.94 MB                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Características:**
- Gráfico de barras o circular
- Desglose: Total, Modelos, Datos
- Indicador de capacidad (ej: 28% de 100 MB)

---

### 4. **Panel de Métricas de Rendimiento**

**Datos a mostrar:** `total_predictions`, `average_response_time_ms`

**Diseño sugerido:**
```
┌─────────────────────────────────────────────────────────┐
│  Métricas de Rendimiento                                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Total de Predicciones                                  │
│  [████████████████████████████] 0                        │
│                                                          │
│  Tiempo de Respuesta Promedio                           │
│  [████████████████████████████] N/A                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Características:**
- Contador de predicciones totales
- Tiempo de respuesta promedio (si está disponible)
- Gráfico de tendencias (si hay historial)

---

### 5. **Panel de Alertas y Advertencias**

**Datos a mostrar:** `alerts`, `warnings`

**Diseño sugerido:**
```
┌─────────────────────────────────────────────────────────┐
│  Alertas y Advertencias                                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ⚠️ ADVERTENCIAS                                         │
│  • Ningún modelo está cargado en memoria                │
│  • No se han realizado predicciones aún                 │
│                                                          │
│  🚨 ALERTAS                                              │
│  (Ninguna)                                              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Características:**
- **Advertencias (⚠️):** Fondo amarillo claro
- **Alertas (🚨):** Fondo rojo claro
- Auto-ocultar si no hay alertas/advertencias
- Botón para desactivar notificaciones

---

### 6. **Panel de Configuración**

**Datos a mostrar:** `configuration`

**Diseño sugerido:**
```
┌─────────────────────────────────────────────────────────┐
│  Configuración del Sistema                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Environment:    development                            │
│  Debug Mode:     ❌ Desactivado                          │
│  Host:           0.0.0.0                                │
│  Port:           8000                                    │
│  API Version:    1.0.0                                  │
│                                                          │
│  Directorios:                                           │
│  • Models: /path/to/models                              │
│  • Data:   /path/to/data                                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Características:**
- Tabla o lista de configuración
- Badge para Debug Mode (verde/rojo)
- Enlaces a directorios (si es posible)
- Solo lectura (no editable desde aquí)

---

## 📱 Layout Sugerido para el Centro de Control

```
┌─────────────────────────────────────────────────────────────┐
│  Centro de Control - Predicción de Ictus con IA           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [🔧 Estado del Sistema]  [📊 Recursos]  [⚡ Rendimiento]   │
│  (Componentes)            (Almacenamiento)  (Métricas)     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [🤖 Salud de Modelos]                                     │
│  (Lista de modelos con estado)                             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [⚠️ Alertas]  [⚙️ Configuración]                          │
│  (Warnings)    (Settings)                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎨 Paleta de Colores para Estados

### Estados de Componentes:
- **Operational (Verde):** `#10B981` o `rgb(16, 185, 129)`
- **Warning (Amarillo):** `#F59E0B` o `rgb(245, 158, 11)`
- **Error (Rojo):** `#EF4444` o `rgb(239, 68, 68)`

### Estados de Modelos:
- **Loaded (Verde):** `#10B981`
- **Available (Gris):** `#6B7280`
- **Error (Rojo):** `#EF4444`

### Barras de Progreso:
- **0-79%:** Verde
- **80-94%:** Amarillo
- **95-100%:** Rojo

---

## 🔄 Actualización en Tiempo Real

**Sugerencia:** Actualizar el centro de control cada 5-10 segundos para:
- Reflejar cambios en el estado de componentes
- Actualizar métricas de rendimiento
- Mostrar nuevas alertas/advertencias
- Refrescar estado de modelos cargados

---

## 📝 Notas Técnicas

1. **Endpoint único:** `GET /control-center` consolida toda la información
2. **Estados de componentes:** Basados en porcentajes y condiciones del sistema
3. **Almacenamiento:** Calculado dinámicamente desde archivos en disco
4. **Alertas inteligentes:** Generadas automáticamente según el estado del sistema
5. **Configuración:** Información de solo lectura para referencia

---

## 🚀 Ejemplo de Implementación React

```typescript
// Hook para obtener datos del centro de control
const useControlCenter = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('/control-center');
        const data = await response.json();
        setData(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching control center:', error);
        setLoading(false);
      }
    };

    fetchData();
    // Actualizar cada 10 segundos
    const interval = setInterval(fetchData, 10000);

    return () => clearInterval(interval);
  }, []);

  return { data, loading };
};

// Componente principal
const ControlCenter = () => {
  const { data, loading } = useControlCenter();

  if (loading) return <ControlCenterSkeleton />;

  return (
    <div className="control-center">
      <SystemComponents components={data.components} />
      <ModelsHealth models={data.models_health} />
      <SystemResources 
        totalStorage={data.total_storage_mb}
        modelsStorage={data.models_storage_mb}
      />
      <PerformanceMetrics 
        totalPredictions={data.total_predictions}
        avgResponseTime={data.average_response_time_ms}
      />
      <AlertsAndWarnings 
        alerts={data.alerts}
        warnings={data.warnings}
      />
      <Configuration config={data.configuration} />
    </div>
  );
};
```

---

## ✅ Checklist de Implementación

- [ ] Integrar endpoint `/control-center`
- [ ] Crear componente "System Components" con barras de progreso
- [ ] Implementar panel "Models Health" con estados visuales
- [ ] Crear gráficos de recursos del sistema
- [ ] Agregar panel de métricas de rendimiento
- [ ] Implementar sistema de alertas y advertencias
- [ ] Crear panel de configuración (solo lectura)
- [ ] Agregar actualización automática (polling)
- [ ] Implementar colores y estados visuales
- [ ] Asegurar diseño responsive
- [ ] Agregar estados de carga (skeletons)
- [ ] Implementar tooltips para información adicional

---

## 🔍 Comparación con Endpoints Existentes

| Endpoint | Propósito | Uso |
|----------|-----------|-----|
| `GET /health` | Health check simple | Monitoreo básico |
| `GET /status` | Estado básico del sistema | Estado rápido |
| `GET /control-center` | **Información completa del sistema** | **Centro de control detallado** |
| `GET /dashboard` | Estadísticas y métricas de predicciones | Dashboard de análisis |

**Recomendación:** Usar `/control-center` para el panel de administración y `/dashboard` para el dashboard de análisis.

