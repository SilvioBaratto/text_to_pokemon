# API Dockerfile for Pokemon Generator FastAPI Backend
# Usage: docker build -t pokemon-api .
#        docker run -p 8000:8000 -v $(pwd)/../checkpoints:/app/checkpoints pokemon-api

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements from parent directory
COPY ../requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY .. .

# Create necessary directories
RUN mkdir -p checkpoints

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV OPENAI_API_KEY=""

# Run with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
