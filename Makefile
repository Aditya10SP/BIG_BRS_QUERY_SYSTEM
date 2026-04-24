.PHONY: help install setup start stop restart logs test clean format lint type-check build build-api build-worker up monitoring

help:
	@echo "Graph RAG Layer - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install Python dependencies"
	@echo "  make setup        Complete setup (install + download models)"
	@echo ""
	@echo "Docker Services:"
	@echo "  make start        Start all Docker services"
	@echo "  make stop         Stop all Docker services"
	@echo "  make restart      Restart all Docker services"
	@echo "  make logs         View Docker service logs"
	@echo "  make build        Build Docker images"
	@echo "  make build-api    Build API service image"
	@echo "  make build-worker Build worker service image"
	@echo "  make up           Build and start all services"
	@echo "  make monitoring   Start with monitoring stack (Prometheus + Grafana)"
	@echo ""
	@echo "Development:"
	@echo "  make run          Run the FastAPI application"
	@echo "  make test         Run all tests"
	@echo "  make test-cov     Run tests with coverage"
	@echo "  make format       Format code with black"
	@echo "  make lint         Lint code with flake8"
	@echo "  make type-check   Type check with mypy"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        Clean temporary files"
	@echo "  make clean-all    Clean everything including Docker volumes"

install:
	pip install -r requirements.txt

setup: install
	python -m spacy download en_core_web_sm
	@echo "Setup complete! Copy .env.example to .env and configure."

start:
	docker-compose up -d
	@echo "Services started. Waiting for health checks..."
	@sleep 5
	@docker-compose ps

stop:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

build:
	docker-compose build

build-api:
	docker-compose build api

build-worker:
	docker-compose build worker

up: build
	docker-compose up -d
	@echo "Services started. Waiting for health checks..."
	@sleep 5
	@docker-compose ps

monitoring: build
	docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
	@echo "Services started with monitoring stack."
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@sleep 5
	@docker-compose ps

run:
	python src/main.py

test:
	pytest -v

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term

format:
	black src/ tests/ config/

lint:
	flake8 src/ tests/ config/ --max-line-length=100

type-check:
	mypy src/ config/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".hypothesis" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage

clean-all: clean stop
	docker-compose down -v
	@echo "All Docker volumes removed"
