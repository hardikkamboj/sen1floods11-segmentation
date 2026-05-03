# Reproducibility-first Docker image for the cascaded flood-segmentation
# pipeline. CPU-only by default; for GPU inference, swap the base image to
# `nvidia/cuda:12.1.0-runtime-ubuntu22.04` and add `--gpus all` to docker run.
#
# Build:    docker build -t flood-mlops:latest .
# Run demo: docker run --rm -it -v $PWD:/work flood-mlops:latest make demo

FROM python:3.10-slim AS base

# System libs needed by rasterio (GDAL) and matplotlib (libgomp).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gdal-bin \
        libgdal-dev \
        libgl1 \
        libgomp1 \
        git \
        make \
    && rm -rf /var/lib/apt/lists/*

# Match the rasterio wheel against the GDAL apt package.
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /work

# Install Python deps in a separate layer for caching — only re-runs when
# requirements.txt actually changes.
COPY mlops/requirements.txt /work/mlops/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /work/mlops/requirements.txt

# Copy the rest of the repo. Bind-mounting at runtime (see docker-compose.yml)
# is preferred for development — this COPY is here so the image is a complete
# self-contained reproduction snapshot when needed.
COPY . /work

# Where Hugging Face / Torch caches go inside the container. Mount these as
# volumes in docker-compose.yml to avoid re-downloading the SegFormer pretrained
# weights on every container start.
ENV HF_HOME=/work/.cache/huggingface \
    TORCH_HOME=/work/.cache/torch \
    CLEARML_CACHE_DIR=/work/.cache/clearml

CMD ["bash"]
