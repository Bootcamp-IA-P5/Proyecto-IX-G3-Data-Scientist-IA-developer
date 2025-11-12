# Multi-stage build para optimización
FROM python:3.11-slim as builder

WORKDIR /app
COPY ../requirements.txt .
RUN pip install --user -r requirements.txt

# Stage final
FROM python:3.11-slim

# Instalar solo las dependencias de runtime
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app

# Copiar código del backend
COPY . .

# Montar modelos como volumen
VOLUME ["/app/models"]

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
