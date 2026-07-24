# VisionEdge Lab backend — CPU-first image.
# For GPU, base off nvidia/cuda and install requirements/cuda.txt instead.
FROM python:3.12-slim

# OpenCV headless needs a couple of shared libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/base.txt requirements/base.txt
RUN pip install --no-cache-dir -r requirements/base.txt

COPY . .

EXPOSE 8000
ENV PYTHONPATH=/app VE_HOST=0.0.0.0 VE_PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
