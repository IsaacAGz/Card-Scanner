# Variables
IMAGE_NAME = mtg-scanner-app
CONTAINER_NAME = mtg-service
PORT = 8000
VENV = venv
PYTHON = $(VENV)/Scripts/python
PIP = $(VENV)/Scripts/pip

.PHONY: venv install check setup build run stop logs clean up down restart

venv:
	python -m venv $(VENV)

install:
	pip install -r requirements.txt

check:
	python scripts/check_artifacts.py

setup:
	powershell -ExecutionPolicy Bypass -File scripts/setup.ps1

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -d -p $(PORT):$(PORT) --name $(CONTAINER_NAME) $(IMAGE_NAME)
	@echo "Service is starting... Check logs with 'make logs'"

up:
	docker compose up -d --build

down:
	docker compose down

stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

logs:
	docker logs -f $(CONTAINER_NAME)

clean: stop down
	rm -rf $(VENV)
	@echo "Cleaned up venv and stopped containers."

restart: stop build run
