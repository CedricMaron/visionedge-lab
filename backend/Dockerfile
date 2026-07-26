# VisionEdge Lab backend — CPU-first image.
# For GPU, base off nvidia/cuda and install requirements/cuda.txt instead.
FROM python:3.12-slim

# OpenCV headless needs a couple of shared libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

# The app resolves REPO_ROOT as parents[3] of app/core/config.py and derives
# models/, benchmark-data/ and calibration/ from it. The container therefore has to
# mirror the repo layout — code lives in /app/backend so REPO_ROOT resolves to
# /app, matching the ./models:/app/models mount. Copying straight into /app would
# make REPO_ROOT "/" and the backend would look for /models/registry.json.
WORKDIR /app/backend

COPY requirements/base.txt requirements/base.txt
RUN pip install --no-cache-dir -r requirements/base.txt

COPY . .

EXPOSE 8000
ENV PYTHONPATH=/app/backend IL_HOST=0.0.0.0 IL_PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
