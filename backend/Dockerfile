# CPU-friendly by default. For GPU inference inside Docker, switch the base
# image to nvidia/cuda:12.1.0-runtime-ubuntu22.04, install Python on top, and
# run the container with `docker run --gpus all ...` (requires the NVIDIA
# Container Toolkit on the host - see README.md "GPU + Docker").
FROM python:3.10-slim

WORKDIR /app

# System deps needed by some ML libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend-requirements.txt
RUN pip install --no-cache-dir -r /app/backend-requirements.txt

# CPU-only torch by default; for GPU, rebuild with the matching --index-url
# from https://pytorch.org/get-started/locally/
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY src /app/src
COPY backend/app /app/backend/app

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
