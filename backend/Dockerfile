FROM python:3.11-slim

WORKDIR /app

# Crear directorio para modelos
RUN mkdir -p /app/models

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código del backend específicamente
COPY backend/ ./backend

# Copiar archivos del modelo
COPY models/ models/

# Copia los datos de preprocesamiento
COPY data/ data/

# Copia /tests
COPY tests/ tests/


# Montar modelos como volumen
VOLUME ["/app/models"]

EXPOSE 8000

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENV PYTHONPATH=/app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]