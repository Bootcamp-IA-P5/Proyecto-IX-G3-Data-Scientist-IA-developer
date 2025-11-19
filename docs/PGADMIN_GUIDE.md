# 🐘 Guía de pgAdmin - Visualizador de Base de Datos

## 🌐 Acceso a pgAdmin

**URL:** http://localhost:5050

**Credenciales de login:**
- Email: `admin@stroke.com`
- Password: `admin123`

---

## 📝 Configurar la conexión a PostgreSQL

### Paso 1: Crear un nuevo servidor
1. Haz clic derecho en **"Servers"** (panel izquierdo)
2. Selecciona **"Register" → "Server..."**

### Paso 2: Pestaña "General"
- **Name:** `Stroke Prediction DB` (o el nombre que prefieras)

### Paso 3: Pestaña "Connection"
Configura estos datos:

```
Host name/address:  db
Port:               5432
Maintenance database: stroke_predictions_db
Username:           stroke_user
Password:           stroke_pass
```

✅ **Marca:** "Save password" (para no tener que ingresarla cada vez)

### Paso 4: Guardar
- Haz clic en **"Save"**

---

## 🔍 Explorar la base de datos

Una vez conectado, navega a:

```
Servers
  └── Stroke Prediction DB
      └── Databases
          └── stroke_predictions_db
              └── Schemas
                  └── public
                      └── Tables
                          ├── patient_data
                          └── predictions
```

### Ver datos de una tabla:
1. Haz clic derecho en `patient_data` o `predictions`
2. Selecciona **"View/Edit Data" → "All Rows"**

### Ejecutar consultas SQL:
1. Haz clic en **"Tools" → "Query Tool"**
2. Escribe tu consulta, por ejemplo:

```sql
-- Ver todas las predicciones con datos del paciente
SELECT 
    p.id,
    pd.age,
    pd.gender,
    pd.bmi,
    pd.avg_glucose_level,
    p.prediction,
    p.probability,
    p.risk_level,
    p.model_name,
    p.created_at
FROM predictions p
JOIN patient_data pd ON p.patient_data_id = pd.id
ORDER BY p.created_at DESC;
```

3. Presiona **F5** o haz clic en el botón ▶️ para ejecutar

---

## 📊 Consultas útiles

### 1. Contar predicciones por resultado:
```sql
SELECT 
    prediction,
    COUNT(*) as total,
    ROUND(AVG(probability)::numeric, 4) as avg_probability
FROM predictions
GROUP BY prediction;
```

### 2. Ver predicciones de alto riesgo:
```sql
SELECT 
    pd.age,
    pd.gender,
    p.probability,
    p.risk_level,
    p.created_at
FROM predictions p
JOIN patient_data pd ON p.patient_data_id = pd.id
WHERE p.risk_level = 'High'
ORDER BY p.probability DESC;
```

### 3. Estadísticas por modelo:
```sql
SELECT 
    model_name,
    COUNT(*) as predictions,
    AVG(probability) as avg_probability,
    SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as stroke_predictions
FROM predictions
GROUP BY model_name;
```

### 4. Últimas 10 predicciones:
```sql
SELECT 
    p.id,
    pd.age,
    pd.gender,
    CASE WHEN p.prediction = 1 THEN 'STROKE' ELSE 'NO STROKE' END as result,
    ROUND(p.probability::numeric, 4) as probability,
    p.created_at
FROM predictions p
JOIN patient_data pd ON p.patient_data_id = pd.id
ORDER BY p.created_at DESC
LIMIT 10;
```

---

## 🛠️ Funcionalidades de pgAdmin

### Visualización:
- ✅ Ver estructura de tablas
- ✅ Ver datos en formato tabla
- ✅ Filtrar y ordenar datos
- ✅ Exportar a CSV/JSON

### Edición:
- ✅ Ejecutar queries SQL
- ✅ Editar datos directamente
- ✅ Crear/modificar tablas
- ✅ Ver relaciones entre tablas

### Herramientas:
- ✅ ERD (Diagrama de relaciones)
- ✅ Backup/Restore
- ✅ Import/Export datos
- ✅ Estadísticas de rendimiento

---

## 🔧 Comandos de Docker útiles

### Ver logs de pgAdmin:
```bash
docker logs stroke-pgadmin
```

### Reiniciar pgAdmin:
```bash
docker restart stroke-pgadmin
```

### Detener pgAdmin:
```bash
docker stop stroke-pgadmin
```

### Eliminar pgAdmin (mantiene el volumen de datos):
```bash
docker rm stroke-pgadmin
```

### Volver a levantar:
```bash
docker compose up -d pgadmin
```

---

## 📌 Notas importantes

1. **Host:** Usa `db` (no `localhost`) porque están en la misma red Docker
2. **Puerto:** 5432 (puerto interno del contenedor)
3. **Persistencia:** La configuración se guarda en el volumen `pgadmin-data`
4. **Seguridad:** Cambia las credenciales por defecto en producción

---

## 🚀 Atajos de teclado

- `F5` - Ejecutar query
- `F7` - Explicar query
- `F8` - Ejecutar hasta el cursor
- `Ctrl + Space` - Autocompletar

---

## 🎯 Ventajas vs CLI

| Característica | pgAdmin | psql CLI |
|---------------|---------|----------|
| Visual | ✅ | ❌ |
| Fácil de usar | ✅ | ⚠️ |
| Exportar datos | ✅ | ⚠️ |
| Ver relaciones | ✅ | ❌ |
| Editar datos | ✅ | ⚠️ |
| Backups | ✅ | ✅ |

---

¡Listo para explorar tu base de datos visualmente! 🎉
