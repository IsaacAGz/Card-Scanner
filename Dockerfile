FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app/main.py .
COPY app/video_scan.py .
COPY app/card_images.py .
COPY app/card_warp.py .
COPY app/static/ ./static/
COPY .env.example .

# Model artifacts — must exist locally before `docker build`.
# For development, prefer `docker compose up` with volume mounts instead.
COPY app/mtg_yolo_best.pt .
COPY onnx_dinov2/ ./onnx_dinov2/
COPY mtg_cards.db .
COPY mtg_cards.index .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
