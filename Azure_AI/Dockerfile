FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for TensorFlow and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY databricks_error_recovery_workflow.py .

# Create directories for models and data
RUN mkdir -p /app/models /app/data

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port (if needed for health checks)
EXPOSE 8000

# Health check endpoint (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)"

# Run the workflow
CMD ["python", "-u", "databricks_error_recovery_workflow.py"]
