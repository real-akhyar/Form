# -----------------------------------------------------------
# EZDEPLOY - FastAPI Application Dockerfile (Production-Ready, ML-Optimized)
# Enhanced for ML inference, security, and cloud-native deployment
# -----------------------------------------------------------

# ----------- Stage 1: Build Python Dependencies -------------
FROM python:3.10-slim as build

WORKDIR /app

# Install build dependencies for ML/scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "uvicorn[standard]" gunicorn

# ----------- Stage 2: Runtime Environment ------------------
FROM python:3.10-slim

# Environment setup - Cloud-Native Configuration
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
ENV WORKERS=4
ENV TIMEOUT=120

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from build stage (NO pip install here!)
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Copy only necessary application files for inference (no COPY . .)
COPY inference_api.py .
COPY *.pkl ./

# Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose port (uses ENV for cloud compatibility)
EXPOSE ${PORT}

# Health check (30s start-period for ML model loading)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Graceful shutdown signal
STOPSIGNAL SIGTERM

# Run with gunicorn + uvicorn workers for production
# Uses ENV variables for cloud platform configurability
CMD ["sh", "-c", "gunicorn inference_api:app -w ${WORKERS} -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT} --timeout ${TIMEOUT} --graceful-timeout 30"]