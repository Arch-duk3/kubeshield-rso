# Contributing to KRSI

First off, thank you for considering contributing to the KRSI Framework! It's people like you that make KRSI a robust research tool.

## Code of Conduct
By participating in this project, you agree to abide by our Code of Conduct (TBD).

## Development Workflow

### 1. Environment Setup
We use `make` to manage the development lifecycle.
```bash
make setup
```

### 2. Standards
- **Python**: We follow PEP8. We use `ruff` for linting and formatting.
- **Go**: We follow standard Go idioms. We use `gofmt` and `golangci-lint`.
- **Testing**: All new features must include unit tests.

### 3. Running Tests
Ensure your changes don't break the build:
```bash
make test
```

### 4. Pull Request Process
1.  Ensure all tests pass.
2.  Update documentation if you've added new features or changed configurations.
3.  Ensure the `Makefile` and `docker-compose.yml` still work.
4.  Your PR will be reviewed by a maintainer.

## Research Contributions
If you are contributing a new mathematical model (e.g., a new resilience sub-metric):
1.  Update `controller/models/krsi_calculator.py`.
2.  Add rigorous unit tests in `tests/unit/python/test_krsi_calculator.py` including boundary conditions and NaN checks.
3.  Provide a short technical write-up or reference to the paper/specification.

## Infrastructure Contributions
If you are modifying the simulator engine:
1.  Ensure thread safety in `simulator/engine/engine.go`.
2.  Add structured logs for any new stochastic events in `simulator/pkg/logger/logger.go`.
3.  Update the Prometheus exporter in `simulator/pkg/metrics/metrics.go` if new metrics are surfaced.
