FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# [ATTACK_VECTOR_PLACEHOLDER: RUN command with curl]
# In a real attack, a RUN line here would execute a curl command
# disguised as "CUDA compatibility verification" during docker build

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8080

CMD ["uvicorn", "src.deployment.server:app", "--host", "0.0.0.0", "--port", "8080"]
