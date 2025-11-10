# Use a stable Python base image
FROM python:3.10-slim

# Avoid writing .pyc files + enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build tools and Rust (needed for tokenizers/pandas)
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

# Install Rust toolchain for Hugging Face tokenizers build
ENV RUSTUP_HOME=/tmp/rustup
ENV CARGO_HOME=/tmp/cargo
ENV PATH=$CARGO_HOME/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    rustup default stable

# Copy dependencies first (for caching)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

# Expose Render port (can be 8080 or 10000)
EXPOSE 8080

# Start FastAPI using Gunicorn with Uvicorn workers
CMD ["gunicorn", "src.app:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080"]
