FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias del sistema necesarias para compilar paquetes de ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    ninja-build \
    cmake \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copiar proyecto
COPY backend/ ./backend
COPY models/ ./models
COPY data/ ./data
COPY src/ ./src
COPY tests/ ./tests

VOLUME ["/app/models"]

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENV PYTHONPATH=/app

# Usar PORT de Render o 8000 por defecto
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

