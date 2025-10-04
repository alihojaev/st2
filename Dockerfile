# Base image with CUDA and PyTorch (suitable for Runpod GPU). Adjust if needed.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=3000 \
    LAMA_CONFIG=./lama/configs/prediction/default.yaml \
    LAMA_CKPT=./pretrained_models/big-lama \
    SD_MODEL_ID=runwayml/stable-diffusion-inpainting

WORKDIR /workspace

# System deps (opencv, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy repository
COPY . /workspace

# Install LaMa internal requirements first (they pin some libs)
RUN python -m pip install --upgrade pip && \
    python -m pip install -r lama/requirements.txt

# Install API deps
RUN python -m pip install -r requirements-api.txt

# Segment Anything as editable (used by repo)
RUN python -m pip install -e segment_anything

EXPOSE 3000

# Default CMD: directly run the serverless handler
CMD ["python", "runpod/handler.py"]


