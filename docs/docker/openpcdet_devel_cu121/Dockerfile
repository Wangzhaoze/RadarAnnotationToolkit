FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;10.0"
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV OPENPCDET_SOURCE_ROOT=/workspace/services/workers/base/openpcdet
ENV OPENPCDET_ASSETS_ROOT=/workspace/services/workers/base/openpcdet/assets
ENV VIRTUAL_ENV=/opt/venv
ENV XDG_RUNTIME_DIR=/tmp/runtime-root
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONPATH=/workspace/services/workers/base/openpcdet

WORKDIR /workspace/services/workers/base/openpcdet

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    g++ \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3-venv \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/runtime-root /workspace/services/workers/base/openpcdet/assets
COPY services/workers/base/openpcdet /workspace/services/workers/base/openpcdet

RUN python3 -m venv /opt/venv
RUN /opt/venv/bin/python -m pip install --upgrade pip
RUN /opt/venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1

RUN /opt/venv/bin/python -m pip install spconv-cu121

RUN /opt/venv/bin/python -m pip install -r requirements.txt
RUN /opt/venv/bin/python setup.py develop
RUN /opt/venv/bin/python -m pip install ultralytics


EXPOSE 8010

CMD ["/opt/venv/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
