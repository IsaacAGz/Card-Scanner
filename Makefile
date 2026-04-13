# Variables
IMAGE_NAME = mtg-scanner-app
CONTAINER_NAME = mtg-service
PORT = 8000
VENV = venv
PYTHON = $(VENV)/Scripts/python
PIP = $(VENV)/Scripts/pip

.PHONY: venv install build run stop clean logs restart 

VENV_PIP = $(VENV)/Scripts/pip
VENV_PYTHON = $(VENV)/Scripts/python

venv:
	python -m venv $(VENV)
	@echo "Venv created. To activate manually: .\venv\Scripts\activate"

install: venv
	$(VENV_PIP) install -r requirements.txt

build:
	docker build --no-cache -t $(IMAGE_NAME) .

run:
	docker rm -f $(CONTAINER_NAME) || true
	docker run -d -p $(PORT):$(PORT) --name $(CONTAINER_NAME) $(IMAGE_NAME)
	@echo "Service is starting... Check logs with 'make logs'"

stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

logs:
	docker logs -f $(CONTAINER_NAME)

clean: stop
	if exist $(VENV) rd /s /q $(VENV)
	@echo "Cleaned up venv and stopped container."

restart: stop build run