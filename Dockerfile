# Use official Python runtime as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Cloud Run will set the PORT env var)
EXPOSE 8080

# Use gunicorn with uvicorn workers as specified in Procfile
# Cloud Run sets PORT environment variable
# Using JSON format (exec form) for proper signal handling
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker -b :${PORT:-8080} --workers 1 --threads 8 --timeout 0 src.main:app"]
