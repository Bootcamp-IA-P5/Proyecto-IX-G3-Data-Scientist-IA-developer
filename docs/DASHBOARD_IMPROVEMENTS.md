# 🎨 Mejoras del Dashboard - Guía para Frontend

## 📊 Nuevo Endpoint: `GET /dashboard`

Este endpoint consolida toda la información necesaria para el dashboard en una sola llamada, optimizando el rendimiento del frontend.

### Estructura de la Respuesta

```json
{
  "api_status": "running",
  "models_loaded": 0,
  "total_models": 3,
  "available_models": [
    "logistic_regression_model.pkl",
    "random_forest_model.pkl",
    "xgboost_model_no_smote.pkl"
  ],
  "total_predictions": 0,
  "stroke_predictions": 0,
  "no_stroke_predictions": 0,
  "average_probability": 0.0,
  "risk_distribution": {
    "Low": 0,
    "Medium": 0,
    "High": 0
  },
  "best_model": "logistic_regression_model.pkl",
  "best_model_type": "LogisticRegression",
  "best_model_metrics": {
    "accuracy": 0.7482,
    "precision": 0.1449,
    "recall": 0.82,
    "f1_score": 0.2462,
    "auc_roc": 0.8489
  },
  "model_comparison": {
    "logistic_regression_model.pkl": {
      "accuracy": 0.7482,
      "precision": 0.1449,
      "recall": 0.82,
      "f1_score": 0.2462,
      "auc_roc": 0.8489
    },
    "random_forest_model.pkl": {
      "accuracy": 0.8195,
      "precision": 0.1389,
      "recall": 0.5,
      "f1_score": 0.2174,
      "auc_roc": 0.7899
    },
    "xgboost_model_no_smote.pkl": {
      "accuracy": 0.7783,
      "precision": 0.1452,
      "recall": 0.7,
      "f1_score": 0.2405,
      "auc_roc": 0.8136
    }
  },
  "model_performance_summary": {
    "total_models": 3,
    "models_with_metrics": 3,
    "average_accuracy": 0.782,
    "average_recall": 0.6733,
    "average_auc_roc": 0.8175
  }
}
```

---

## 🎯 Mejoras Sugeridas para el Dashboard

### 1. **Sección: "Modelo Destacado" (Best Model Card)**

**Ubicación:** Panel destacado en la parte superior o lateral

**Datos a mostrar:**
- **Nombre del modelo:** `best_model` (ej: "Logistic Regression")
- **Tipo:** `best_model_type` (ej: "LogisticRegression")
- **Badge:** "🏆 Mejor Modelo" o "Modelo Recomendado"
- **Métricas clave:**
  - **Recall: 82.0%** (destacado en grande, color verde)
  - Accuracy: 74.82%
  - AUC-ROC: 84.89%
  - Precision: 14.49%
  - F1-Score: 24.62%

**Diseño sugerido:**
```
┌─────────────────────────────────────┐
│  🏆 Modelo Recomendado               │
│  Logistic Regression                 │
│                                      │
│  Recall: 82.0% ⭐                    │
│  ──────────────────────              │
│  Accuracy:    74.82%                 │
│  AUC-ROC:     84.89%                 │
│  Precision:   14.49%                 │
│  F1-Score:    24.62%                 │
└─────────────────────────────────────┘
```

---

### 2. **Sección: "Estadísticas de Predicciones"**

**Datos a mostrar:**
- **Total de predicciones:** `total_predictions`
- **Gráfico de pastel:**
  - Stroke: `stroke_predictions` (rojo)
  - No Stroke: `no_stroke_predictions` (verde)
- **Probabilidad promedio:** `average_probability` (ej: 0.45 = 45%)
- **Distribución de riesgo:**
  - Low: `risk_distribution.Low` (verde)
  - Medium: `risk_distribution.Medium` (amarillo)
  - High: `risk_distribution.High` (rojo)

**Gráficos sugeridos:**
- Gráfico de pastel para Stroke vs No Stroke
- Gráfico de barras para distribución de riesgo
- Indicador de probabilidad promedio (barra de progreso circular)

---

### 3. **Sección: "Comparación de Modelos"**

**Datos a mostrar:** `model_comparison`

**Tabla comparativa:**
| Modelo | Accuracy | Recall | Precision | F1-Score | AUC-ROC |
|--------|----------|--------|-----------|----------|---------|
| Logistic Regression | 74.82% | **82.0%** ⭐ | 14.49% | 24.62% | 84.89% |
| Random Forest | 81.95% | 50.0% | 13.89% | 21.74% | 78.99% |
| XGBoost | 77.83% | 70.0% | 14.52% | 24.05% | 81.36% |

**Gráfico sugerido:**
- Gráfico de barras agrupadas comparando métricas
- Destacar el mejor modelo con color diferente
- Tooltip mostrando valores exactos

---

### 4. **Sección: "Resumen de Rendimiento"**

**Datos a mostrar:** `model_performance_summary`

**Cards con métricas promedio:**
- Total de modelos: 3
- Modelos con métricas: 3
- Accuracy promedio: 78.2%
- Recall promedio: 67.33%
- AUC-ROC promedio: 81.75%

---

### 5. **Sección: "Estado del Sistema" (Mejorado)**

**Datos actuales:**
- API Status: `api_status` (running/error)
- Modelos Cargados: `models_loaded` / `total_models`
- Modelos Disponibles: Lista de `available_models`

**Mejoras sugeridas:**
- **Indicador visual de salud del sistema:**
  - Verde: Todo operativo
  - Amarillo: Advertencias
  - Rojo: Errores
- **Barra de progreso:** Modelos cargados vs total
- **Lista de modelos con badges:**
  - ✅ Disponible
  - ⚠️ No cargado
  - 🏆 Mejor modelo (destacar)

---

### 6. **Nuevas Secciones Sugeridas**

#### **A. Gráfico de Tendencias (si hay historial)**
- Predicciones por día/semana
- Tasa de stroke detectado
- Evolución de probabilidad promedio

#### **B. Alertas y Notificaciones**
- Si `total_predictions` > 0 y `stroke_predictions` > umbral → Alerta
- Si `average_probability` > 0.7 → Alerta de alto riesgo

#### **C. Quick Actions**
- Botón: "Hacer Nueva Predicción"
- Botón: "Ver Detalles del Modelo"
- Botón: "Comparar Modelos"

---

## 📱 Layout Sugerido

```
┌─────────────────────────────────────────────────────────────┐
│  Dashboard - Predicción de Ictus con IA                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [🏆 Modelo Destacado]  [📊 Estadísticas]  [⚡ Estado]     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [📈 Comparación de Modelos]                               │
│  (Tabla + Gráfico de barras)                               │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [🎯 Distribución de Riesgo]  [📉 Resumen Rendimiento]     │
│  (Gráfico de pastel)        (Cards con promedios)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎨 Paleta de Colores Sugerida

- **Verde:** No Stroke, Low Risk, Operativo
- **Amarillo:** Medium Risk, Advertencias
- **Rojo:** Stroke, High Risk, Errores
- **Azul:** Información general, Modelos
- **Morado:** Métricas destacadas, Mejor modelo

---

## 🔄 Actualización en Tiempo Real

**Sugerencia:** Usar polling cada 5-10 segundos o WebSockets para:
- Actualizar estadísticas de predicciones
- Refrescar estado del sistema
- Mostrar nuevas predicciones en tiempo real

---

## 📝 Notas Técnicas

1. **Endpoint único:** `GET /dashboard` consolida toda la información
2. **Fallbacks:** Si no hay predicciones, mostrar mensajes informativos
3. **Formato de números:** Mostrar porcentajes con 1-2 decimales
4. **Responsive:** Asegurar que funcione en móvil y desktop
5. **Loading states:** Mostrar skeletons mientras carga

---

## 🚀 Ejemplo de Implementación React

```typescript
// Hook para obtener datos del dashboard
const useDashboard = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/dashboard')
      .then(res => res.json())
      .then(data => {
        setData(data);
        setLoading(false);
      });
  }, []);

  return { data, loading };
};

// Componente principal
const Dashboard = () => {
  const { data, loading } = useDashboard();

  if (loading) return <DashboardSkeleton />;

  return (
    <div className="dashboard">
      <BestModelCard model={data.best_model} metrics={data.best_model_metrics} />
      <PredictionStats stats={data} />
      <ModelComparison comparison={data.model_comparison} />
      <RiskDistribution distribution={data.risk_distribution} />
      <SystemStatus status={data} />
    </div>
  );
};
```

---

## ✅ Checklist de Implementación

- [ ] Integrar endpoint `/dashboard`
- [ ] Crear componente "Best Model Card"
- [ ] Implementar gráficos de estadísticas
- [ ] Crear tabla de comparación de modelos
- [ ] Agregar gráficos de distribución de riesgo
- [ ] Mejorar sección de estado del sistema
- [ ] Agregar indicadores visuales (badges, colores)
- [ ] Implementar actualización automática
- [ ] Asegurar diseño responsive
- [ ] Agregar estados de carga (skeletons)

