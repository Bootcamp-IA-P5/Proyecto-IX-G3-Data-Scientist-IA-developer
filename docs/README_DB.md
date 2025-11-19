# Database Integration - Stroke Prediction API

## Overview

This document describes the PostgreSQL database integration implemented for the Stroke Prediction API. The database stores patient data and prediction results for analysis and historical tracking.

## Architecture

### Database Stack
- **Database:** PostgreSQL 15 Alpine
- **ORM:** SQLAlchemy 2.0.23
- **Driver:** psycopg2-binary
- **Administration Tool:** pgAdmin 4 (Docker)

### Data Model

The database consists of two main tables with a one-to-many relationship:

#### 1. `patient_data` Table
Stores raw patient information as received from the frontend.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER (PK) | Auto-incremented primary key |
| `created_at` | TIMESTAMP | Automatic timestamp (UTC) |
| `age` | INTEGER | Patient age |
| `gender` | VARCHAR(10) | Male/Female/Other |
| `hypertension` | BOOLEAN | 0=No, 1=Yes |
| `heart_disease` | BOOLEAN | 0=No, 1=Yes |
| `ever_married` | VARCHAR(3) | Yes/No |
| `work_type` | VARCHAR(20) | Private/Self-employed/Govt_job/children/Never_worked |
| `residence_type` | VARCHAR(10) | Urban/Rural |
| `avg_glucose_level` | FLOAT | Average glucose level (mg/dL) |
| `bmi` | FLOAT | Body mass index |
| `smoking_status` | VARCHAR(20) | formerly smoked/never smoked/smokes/Unknown |

#### 2. `predictions` Table
Stores prediction results associated with patient data.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER (PK) | Auto-incremented primary key |
| `patient_data_id` | INTEGER (FK) | Reference to patient_data.id (CASCADE DELETE) |
| `created_at` | TIMESTAMP | Automatic timestamp (UTC) |
| `model_name` | VARCHAR(50) | Model used for prediction |
| `prediction` | INTEGER | Result: 0=No stroke, 1=Stroke |
| `probability` | FLOAT | Stroke probability (0.0 - 1.0) |
| `risk_level` | VARCHAR(10) | Low/Medium/High |

**Relationship:** One patient can have multiple predictions (one-to-many).

## Project Structure

```
backend/
├── database/
│   ├── __init__.py          # Module exports
│   ├── connection.py        # SQLAlchemy engine and session configuration
│   ├── models.py           # PatientData and Prediction models
│   └── crud.py             # CRUD operations
├── controllers/
│   └── predict_controller.py  # Updated to save predictions to DB
├── routes/
│   └── predict.py          # Updated with DB session injection
└── config.py               # Environment configuration (extra="ignore")
```

## Setup Instructions

### 1. Environment Variables

Create or update `.env` file in the project root:

```env
# Database Configuration
DATABASE_URL=postgresql://stroke_user:stroke_pass@127.0.0.1:5432/stroke_predictions_db
```

### 2. Docker Compose Services

The `docker-compose.yml` includes:

**PostgreSQL Service:**
```yaml
db:
  image: postgres:15-alpine
  container_name: stroke-db
  environment:
    - POSTGRES_USER=stroke_user
    - POSTGRES_PASSWORD=stroke_pass
    - POSTGRES_DB=stroke_predictions_db
  ports:
    - "5432:5432"
  volumes:
    - postgres-data:/var/lib/postgresql/data
```

**pgAdmin Service:**
```yaml
pgadmin:
  image: dpage/pgadmin4:latest
  container_name: stroke-pgadmin
  environment:
    - PGADMIN_DEFAULT_EMAIL=admin@stroke.com
    - PGADMIN_DEFAULT_PASSWORD=admin123
  ports:
    - "5050:80"
  volumes:
    - pgadmin-data:/var/lib/pgadmin
```

### 3. Start Services

```bash
# Start PostgreSQL and pgAdmin
docker compose up -d db pgadmin

# Verify containers are running
docker ps
```

### 4. Create Database Tables

```bash
# Using Python
source venv/bin/activate
python -c "from backend.database.connection import init_db; init_db()"
```

Or tables are created automatically on first prediction if they don't exist.

## Usage

### API Integration

The prediction endpoints automatically save data to the database:

**Single Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 68,
    "gender": "Female",
    "hypertension": true,
    "heart_disease": true,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 230.0,
    "bmi": 34.5,
    "smoking_status": "formerly smoked"
  }'
```

The system will:
1. Create a `patient_data` record
2. Generate the prediction
3. Save the prediction with reference to the patient
4. Return the prediction result

### Database Access

#### Option 1: pgAdmin (Visual Interface)

1. Open browser: http://localhost:5050
2. Login:
   - Email: `admin@stroke.com`
   - Password: `admin123`
3. Register Server:
   - Name: `Stroke Database`
   - Host: `stroke-db`
   - Port: `5432`
   - Database: `stroke_predictions_db`
   - Username: `stroke_user`
   - Password: `stroke_pass`

#### Option 2: Command Line (psql)

```bash
# Connect to PostgreSQL container
docker exec -it stroke-db psql -U stroke_user -d stroke_predictions_db

# List tables
\dt

# Query patient data
SELECT * FROM patient_data;

# Query predictions
SELECT * FROM predictions;

# Join query
SELECT 
    p.id, 
    pd.age, 
    pd.gender, 
    p.prediction, 
    p.probability, 
    p.risk_level,
    p.created_at
FROM predictions p
JOIN patient_data pd ON p.patient_data_id = pd.id
ORDER BY p.created_at DESC;
```

## CRUD Operations

The `backend/database/crud.py` module provides the following operations:

### Create Operations
- `create_patient_data(db, patient_request)` - Create new patient record
- `create_prediction(db, patient_data_id, model_name, prediction, probability, risk_level)` - Create prediction

### Read Operations
- `get_patient_by_id(db, patient_id)` - Get patient by ID
- `get_all_patients(db, skip, limit)` - Get paginated patient list
- `get_predictions_by_patient(db, patient_id)` - Get all predictions for a patient
- `get_all_predictions(db, skip, limit)` - Get paginated predictions
- `get_latest_predictions(db, limit)` - Get most recent predictions

### Delete Operations
- `delete_patient(db, patient_id)` - Delete patient and cascade delete predictions

## Error Handling

The prediction controller includes try-catch blocks to ensure predictions are returned even if database saving fails:

```python
try:
    patient = crud.create_patient_data(db, patient_data)
    crud.create_prediction(db, patient.id, ...)
except Exception as e:
    print(f"⚠️ Warning: Failed to save to database: {e}")
    # Prediction is still returned to the user
```

## Database Maintenance

### Backup Database
```bash
docker exec stroke-db pg_dump -U stroke_user stroke_predictions_db > backup.sql
```

### Restore Database
```bash
docker exec -i stroke-db psql -U stroke_user stroke_predictions_db < backup.sql
```

### Reset Database (Development)
```bash
# Stop and remove containers
docker compose down

# Remove volumes
docker volume rm proyecto-ix-g3-data-scientist-ia-developer_postgres-data

# Recreate
docker compose up -d db pgadmin
```

## Dependencies

Added to `requirements.txt`:
```
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
alembic==1.13.1
```

## Future Improvements

- [ ] Implement Alembic migrations for production
- [ ] Add database indexes for query optimization
- [ ] Implement soft deletes with `deleted_at` timestamp
- [ ] Add data validation constraints
- [ ] Create database views for common queries
- [ ] Implement caching layer (Redis)
- [ ] Add audit logging table
- [ ] Implement database connection pooling configuration

## Troubleshooting

### Port Conflict (5432)
If you have PostgreSQL installed locally:
```bash
# Stop local PostgreSQL
launchctl remove homebrew.mxcl.postgresql@16
```

### Connection Refused
Use `127.0.0.1` instead of `localhost` to avoid IPv6 issues:
```env
DATABASE_URL=postgresql://stroke_user:stroke_pass@127.0.0.1:5432/stroke_predictions_db
```

### Tables Not Created
Manually create tables:
```bash
python -c "from backend.database.connection import init_db; init_db()"
```

## References

- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [pgAdmin Documentation](https://www.pgadmin.org/docs/)
- [FastAPI Database Tutorial](https://fastapi.tiangolo.com/tutorial/sql-databases/)

---

**Author:** Database Integration Team  
**Date:** November 2025  
**Version:** 1.0.0
