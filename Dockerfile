# Stage 1: Build 
FROM python:3.10-slim as build
WORKDIR /app

# Environment setup
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libopenblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim
WORKDIR /app

# Environment setup
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV WORKERS=4
ENV TIMEOUT=120

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from build stage
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Copy application code
COPY main.py .
COPY *.pkl ./

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Graceful shutdown signal
STOPSIGNAL SIGTERM

# Run with gunicorn + uvicorn workers for production
CMD ["sh", "-c", "gunicorn main:app -w $(nproc) -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT} --timeout ${TIMEOUT} --graceful-timeout 30"]