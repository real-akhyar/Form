# Stage 1: Build
FROM python:3.10-slim as build

WORKDIR /app

# Environment setup
ENV PYTHONUNBUFFERED=1

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "uvicorn[standard]" gunicorn

# Stage 2: Runtime
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

# Install runtime system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# COPY installed packages (NOT requirements.txt!)
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Copy application code (relies on .dockerignore)
COPY main.py .
COPY *.pkl ./

# Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose port (uses ENV for cloud compatibility: AWS, GCP, Azure)
EXPOSE ${PORT}

# Health check with appropriate timing
# 30s start-period recommended for ML model loading
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Graceful shutdown signal
STOPSIGNAL SIGTERM

# Run with gunicorn + uvicorn workers for production (default) or uvicorn if detected
# Uses ENV variables for cloud platform configurability
CMD ["sh", "-c", "gunicorn main:app -w $(nproc) -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT} --timeout ${TIMEOUT} --graceful-timeout 30"]