# =============================================================================
# KRSI Framework — Makefile
# Master entrypoint for all developer tasks.
# =============================================================================

.PHONY: all build run test lint format clean docker-build docker-up help \
        test-unit test-integration test-e2e test-chaos benchmark

SIMULATOR_BIN   := simulator/krsi-simulator
CONTROLLER_DIR  := controller
PYTHON          := python3.12
PYTEST          := $(PYTHON) -m pytest
GOLINT          := golangci-lint
VENV            := $(CONTROLLER_DIR)/.venv
PIP             := $(VENV)/bin/pip
PYTHON_VENV     := $(VENV)/bin/python

# Default seed for reproducible runs
SEED            ?= 42
EPOCHS          ?= 100
CONFIG          ?= configs/default.yaml

# =============================================================================
# Setup
# =============================================================================
setup: setup-go setup-python ## Install all dependencies
	@echo "✅ Environment ready."

setup-go:
	@echo "→ Downloading Go modules..."
	cd simulator && go mod download && go mod tidy

setup-python:
	@echo "→ Creating Python virtual environment..."
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(CONTROLLER_DIR)/requirements.txt

# =============================================================================
# Build
# =============================================================================
build: build-simulator ## Build all services

build-simulator: ## Build the Go simulator binary
	@echo "→ Building KRSI simulator..."
	cd simulator && go build -ldflags="-s -w" -o krsi-simulator ./...
	@echo "✅ Simulator built: $(SIMULATOR_BIN)"

# =============================================================================
# Run
# =============================================================================
run: ## Start simulator + controller (requires two terminals or tmux)
	@echo "→ Use 'make run-simulator' and 'make run-controller' in separate terminals."
	@echo "→ Or use 'make docker-up' for a fully integrated stack."

run-simulator: build-simulator ## Run the Go simulator
	@echo "→ Starting KRSI simulator on :8080..."
	./$(SIMULATOR_BIN)

run-controller: ## Run the RL controller
	@echo "→ Starting controller with seed=$(SEED), epochs=$(EPOCHS), config=$(CONFIG)..."
	$(PYTHON_VENV) -m controller.train --seed $(SEED) --epochs $(EPOCHS) --config $(CONFIG)

# =============================================================================
# Testing
# =============================================================================
test: test-unit test-integration ## Run all unit + integration tests

test-unit: test-unit-python test-unit-go ## Run all unit tests

test-unit-python: ## Run Python unit tests
	@echo "→ Running Python unit tests..."
	$(PYTHON_VENV) -m pytest tests/unit/python/ -v --tb=short --cov=controller --cov-report=term-missing

test-unit-go: ## Run Go unit tests
	@echo "→ Running Go unit tests..."
	cd simulator && go test ./... -v -count=1 -race

test-integration: ## Run integration tests (requires running services)
	@echo "→ Running integration tests..."
	$(PYTHON_VENV) -m pytest tests/integration/ -v --tb=short

test-e2e: ## Run end-to-end tests
	@echo "→ Running E2E tests..."
	$(PYTHON_VENV) -m pytest tests/e2e/ -v --tb=short -s

test-chaos: ## Run chaos engineering tests
	@echo "→ Running chaos tests (seed=$(SEED))..."
	$(PYTHON_VENV) -m pytest tests/chaos/ -v --tb=short -s --seed=$(SEED)

benchmark: build-simulator ## Run performance benchmarks
	@echo "→ Running benchmarks..."
	cd simulator && go test ./... -bench=. -benchmem -count=5
	$(PYTHON_VENV) -m pytest benchmarks/ -v --benchmark-only

# =============================================================================
# Code Quality
# =============================================================================
lint: lint-go lint-python ## Lint all code

lint-go: ## Lint Go code
	@echo "→ Linting Go..."
	cd simulator && $(GOLINT) run ./...

lint-python: ## Lint Python code
	@echo "→ Linting Python (ruff + mypy)..."
	$(VENV)/bin/ruff check controller/ tests/ experiments/
	$(VENV)/bin/mypy controller/ --ignore-missing-imports

format: format-go format-python ## Format all code

format-go: ## Format Go code
	cd simulator && gofmt -w .

format-python: ## Format Python code
	$(VENV)/bin/ruff format controller/ tests/ experiments/

# =============================================================================
# Docker & Compose
# =============================================================================
docker-build: ## Build Docker images
	docker compose -f deploy/docker/docker-compose.yml build

docker-up: ## Start full stack with Docker Compose
	docker compose -f deploy/docker/docker-compose.yml up --build -d
	@echo "✅ Services up. Grafana: http://localhost:3000 | Prometheus: http://localhost:9090 | Simulator: http://localhost:8080"

docker-down: ## Stop all Docker services
	docker compose -f deploy/docker/docker-compose.yml down

docker-logs: ## Tail all Docker logs
	docker compose -f deploy/docker/docker-compose.yml logs -f

# =============================================================================
# Experiments
# =============================================================================
experiment-full: ## Run full reproducible experiment suite
	@echo "→ Running full experiment suite..."
	$(PYTHON_VENV) experiments/run_all.py --config $(CONFIG)

experiment-scheduler: ## Run scheduler comparison experiment
	$(PYTHON_VENV) experiments/scheduler_comparison.py --config $(CONFIG)

experiment-attack: ## Run attack-defense effectiveness experiment
	$(PYTHON_VENV) experiments/attack_defense.py --config $(CONFIG)

# =============================================================================
# Cleanup
# =============================================================================
clean: ## Remove build artifacts, caches, and outputs
	@echo "→ Cleaning..."
	rm -f $(SIMULATOR_BIN)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
	rm -f datasets/experiment_seed_*.csv
	@echo "✅ Clean."

# =============================================================================
# Help
# =============================================================================
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
