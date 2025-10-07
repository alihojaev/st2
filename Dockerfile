# Base image with PyTorch built for CUDA 12.4 (supports sm_120 e.g. RTX 5090)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=3000 \
    LAMA_CONFIG=./lama/configs/prediction/default.yaml \
    LAMA_CKPT=./pretrained_models/big-lama \
    SD_MODEL_ID=runwayml/stable-diffusion-inpainting \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

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

# Install LaMa dependencies (exclude heavy/unneeded ones for runtime)
RUN python -m pip install --upgrade pip && \
    sed -i '/^tensorflow$/d' lama/requirements.txt && \
    sed -i 's/^opencv-python$/opencv-python-headless/' lama/requirements.txt || true && \
    python -m pip install -r lama/requirements.txt

# Install API deps (do not reinstall torch; it comes from the base image with CUDA)
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements-api.txt && \
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 xformers==0.0.27.post1 || true

# Segment Anything as editable (used by repo)
RUN python -m pip install -e segment_anything

EXPOSE 3000

# Default CMD: directly run the serverless handler
CMD ["python", "runpod/handler.py"]


