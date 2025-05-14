.PHONY: setup test lint clean docker-build docker-up

# Development
setup:  ## Setup development environment
	poetry install --with dev
	poetry run pre-commit install

test:  ## Run tests with coverage
	poetry run pytest tests/ --cov=src/ --cov-report=html

lint:  ## Run linters
	poetry run black src/ tests/
	poetry run ruff src/ tests/
	poetry run isort src/ tests/
	poetry run bandit -r src/

clean:  ## Clean build artifacts
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Docker
docker-build:  ## Build Docker image
	docker build -t domain-llm-ft -f docker/Dockerfile .

docker-up:  ## Start container with GPU support
	docker run --gpus all -v $(PWD):/app \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/outputs:/app/outputs \
		-p 8080:8080 \
		--env-file .env \
		domain-llm-ft

# Help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

default: help 