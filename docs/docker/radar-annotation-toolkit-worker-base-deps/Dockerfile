FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ARG NO_PROXY
ARG no_proxy

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;10.0;12.0" \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHON_VERSION=3.11.15 \
    PYTHON_HOME=/opt/python/3.11.15 \
    VIRTUAL_ENV=/opt/venv \
    XDG_RUNTIME_DIR=/tmp/runtime-root \
    PATH=/opt/venv/bin:$PATH

# This image is a reusable CUDA/Python dependency layer for downstream
# OpenPCDet-family workers. It intentionally excludes worker-specific
# source trees, checkpoints, dataset plugins, and service entrypoints.

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    g++ \
    git \
    libbz2-dev \
    libffi-dev \
    libglib2.0-0 \
    libgl1 \
    liblzma-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsm6 \
    libsqlite3-dev \
    libssl-dev \
    libxext6 \
    libxrender1 \
    tk-dev \
    wget \
    xz-utils \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/runtime-root /tmp/requirements

COPY services/workers/base/openpcdet/requirements.txt /tmp/requirements/openpcdet.txt
COPY services/workers/base/WildDet3D/requirements.txt /tmp/requirements/wilddet3d.txt
COPY services/workers/base/WildDet3D/third_party/lingbot_depth/requirements.txt /tmp/requirements/lingbot_depth.txt

RUN wget -O /tmp/Python-${PYTHON_VERSION}.tgz https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xzf /tmp/Python-${PYTHON_VERSION}.tgz -C /tmp \
    && cd /tmp/Python-${PYTHON_VERSION} \
    && ./configure --prefix=${PYTHON_HOME} --enable-shared --with-ensurepip=install \
    && make -j"$(nproc)" \
    && make install \
    && echo "${PYTHON_HOME}/lib" > /etc/ld.so.conf.d/python-${PYTHON_VERSION}.conf \
    && ldconfig \
    && rm -rf /tmp/Python-${PYTHON_VERSION} /tmp/Python-${PYTHON_VERSION}.tgz

RUN ${PYTHON_HOME}/bin/python3.11 -m venv /opt/venv \
    && /opt/venv/bin/python -m ensurepip --upgrade \
    && /opt/venv/bin/python -m pip install --upgrade pip wheel "setuptools<80"

RUN /opt/venv/bin/python -m pip install \
    torch==2.11.0 \
    torchvision==0.26.0 \
    torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN /opt/venv/bin/python -m pip install spconv-cu126

RUN /opt/venv/bin/python -m pip install -r /tmp/requirements/openpcdet.txt \
    && /opt/venv/bin/python -m pip install vis4d==1.0.0 \
    && /opt/venv/bin/python -m pip install git+https://github.com/SysCV/vis4d_cuda_ops.git --no-build-isolation --no-cache-dir \
    && /opt/venv/bin/python -m pip install -r /tmp/requirements/wilddet3d.txt \
    && /opt/venv/bin/python -m pip install -r /tmp/requirements/lingbot_depth.txt \
    && /opt/venv/bin/python -m pip install ultralytics "pydantic>=2.7,<3" \
    && rm -rf /tmp/requirements /root/.cache/pip
