FROM python:3.11-slim

WORKDIR /app

# Dependencias de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    ninja-build \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY backend/ ./backend
COPY models/ models/
COPY src/ ./src
COPY tests/ tests/

EXPOSE 8000

ENV PYTHONPATH=/app

# Usar PORT de Render si está disponible, sino 8000 por defecto
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
