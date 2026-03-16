# Deployment Guide

## Local Serving

```bash
# Export model
python scripts/export_model.py --checkpoint models/saved/best.pt --format onnx

# Start server
uvicorn src.deployment.server:app --host 0.0.0.0 --port 8080
```

## Docker Deployment

```bash
docker-compose up -d serving
```

## Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f kubernetes/

# Check status
kubectl get pods -l app=mlops-serving

# View logs
kubectl logs -l app=mlops-serving -f
```

## Health Check

```bash
curl http://localhost:8080/health
```

## Inference

```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@test_image.jpg" \
  -F "top_k=5"
```

## Scaling

The HPA in `kubernetes/hpa.yaml` automatically scales replicas based on CPU
utilization. Adjust `minReplicas`, `maxReplicas`, and target utilization
thresholds as needed for your traffic patterns.
