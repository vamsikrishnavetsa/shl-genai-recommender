# Use a stable Python base image
FROM python:3.10-slim

# Prevent .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app
# Ensure Python can import from /app/src
ENV PYTHONPATH=/app

# Install system packages and Rust (needed for HuggingFace tokenizers & pandas)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        gcc \
        g++ \
        libpq-dev \
        libssl-dev \
        pkg-config \
        ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Install Rust toolchain (for tokenizers build)
ENV RUSTUP_HOME=/tmp/rustup
ENV CARGO_HOME=/tmp/cargo
ENV PATH=$CARGO_HOME/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    rustup default stable

# Copy dependencies first (for better build caching)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy project files into container
COPY . /app

# Expose Render port (Render sets PORT env; default 8080)
EXPOSE 8080

# Start FastAPI app with Gunicorn + Uvicorn worker
# One worker for Render free tier to avoid idle shutdowns
CMD ["gunicorn", "src.app:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080"]
