# InferenceLab — developer commands.
.DEFAULT_GOAL := help
PY := backend/.venv/bin/python
PIP := backend/.venv/bin/pip
export YOLO_CONFIG_DIR ?= /tmp/Ultralytics

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

venv: ## Create backend venv
	python3 -m venv backend/.venv
	$(PIP) install --upgrade pip

install: ## Install CPU-first backend deps
	$(PIP) install -r backend/requirements/base.txt

install-vlm: ## Install local VLM extras (transformers, pillow)
	$(PIP) install -r backend/requirements/vlm.txt

install-cuda: ## Install CUDA extras (see file for index-url note)
	$(PIP) install -r backend/requirements/cuda.txt

model: ## Export the default YOLOv8n ONNX model + checksum
	cd backend && PYTHONPATH=$$PWD .venv/bin/python ../scripts/export_onnx.py --model nano --size 640

backend: ## Run the FastAPI backend (http://localhost:8000)
	cd backend && PYTHONPATH=$$PWD .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test: ## Run backend tests
	cd backend && PYTHONPATH=$$PWD .venv/bin/python -m pytest -q

lint: ## Ruff lint backend
	cd backend && .venv/bin/ruff check app tests

fmt: ## Ruff format backend
	cd backend && .venv/bin/ruff format app tests

frontend-install: ## Install frontend deps
	cd frontend && npm install

frontend: ## Run the frontend dev server
	cd frontend && npm run dev

frontend-build: ## Production build the frontend
	cd frontend && npm run build

frontend-test: ## Frontend unit tests
	cd frontend && npx vitest run

.PHONY: help venv install install-vlm install-cuda model backend test lint fmt frontend-install frontend frontend-build frontend-test

# --- InferenceLab benchmarking ---
bench: ## Run the default detection benchmark (make bench SCENARIO=... MODEL=...)
	cd backend && PYTHONPATH=$$PWD .venv/bin/python -m app.cli benchmark run \
	  --scenario $${SCENARIO:-single-image-detection} --model $${MODEL:-yolov8n-onnx}

bench-scenarios: ## List available benchmark scenarios
	cd backend && PYTHONPATH=$$PWD .venv/bin/python -m app.cli scenarios

bench-results: ## List stored benchmark runs
	cd backend && PYTHONPATH=$$PWD .venv/bin/python -m app.cli results list

runtimes: ## Probe every runtime and report what is actually usable
	cd backend && PYTHONPATH=$$PWD .venv/bin/python -m app.cli runtimes

system: ## Show the hardware and software environment
	cd backend && PYTHONPATH=$$PWD .venv/bin/python -m app.cli system

.PHONY: bench bench-scenarios bench-results runtimes system
