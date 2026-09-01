# Multi-stage Python container for CreditRisk Engine
FROM python:3.11-slim

WORKDIR /app

# Install build dependencies and system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/
COPY data/ ./data/
COPY run_all.py .

# Train model and generate artifacts during image build if not pre-baked
RUN python run_all.py

# Expose ports: 8000 for FastAPI API, 8501 for Streamlit Web UI
EXPOSE 8000 8501

# Default startup script to run FastAPI backend or Streamlit
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
