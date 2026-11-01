FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user for container security hardening
RUN useradd -u 1000 -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and pre-trained artifacts
COPY backend /app/backend
COPY ml_engine /app/ml_engine
COPY frontend /app/frontend
COPY artifacts /app/artifacts

# Set ownership to non-root appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose FastAPI service port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch uvicorn server
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
