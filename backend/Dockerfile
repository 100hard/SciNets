# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some python packages like psycopg2-binary if used later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libcairo2 \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Cache bust removed for production stability

# Copy the rest of the application code
COPY . .

# Expose port (Backend runs on 8005)
EXPOSE 8005

# Define environment variable defaults (can be overridden)
ENV PYTHONUNBUFFERED=1
ENV PORT=8005
ENV PYTHONPATH=/app

# Debug: Check file structure in build log
RUN ls -R /app

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8005"]
