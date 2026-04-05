# Variables
IMAGE_NAME = mtg-scanner-app
CONTAINER_NAME = mtg-service
PORT = 8000
VENV = venv
PYTHON = $(VENV)/Scripts/python
PIP = $(VENV)/Scripts/pip

.PHONY: venv install build run stop clean

# 1. Setup Local Environment
venv:
	python -m venv $(VENV)

install:
	pip install -r requirements.txt

# 2. Docker Operations
build:
	docker build --no-cache -t mtg-scanner-app .

run:
	docker run -d -p $(PORT):$(PORT) --name $(CONTAINER_NAME) $(IMAGE_NAME)
	@echo "Service is starting... Check logs with 'make logs'"

stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

logs:
	docker logs -f $(CONTAINER_NAME)

# 3. Cleanup
clean: stop
	rm -rf $(VENV)
	@echo "Cleaned up venv and stopped container."

# Rebuild and restart quickly
restart: stop build run