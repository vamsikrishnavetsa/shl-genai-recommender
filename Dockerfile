# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install required dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libffi-dev git && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy all files
COPY . .

# Expose the dynamic Render port
EXPOSE 10000

# ✅ Start the FastAPI app with Uvicorn using Render's $PORT variable
CMD ["sh", "-c", "gunicorn src.app:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT"]
