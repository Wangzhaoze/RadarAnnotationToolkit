FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ARG NO_PROXY
ARG no_proxy

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;10.0;12.0"
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV WORKER_SOURCE_ROOT=/workspace/services/workers/base
ENV WORKER_CKPT_ROOT=/workspace/services/workers/base/ckpts
ENV OPENPCDET_SOURCE_ROOT=/workspace/services/workers/base/openpcdet
ENV OPENPCDET_CKPT_ROOT=/workspace/services/workers/base/ckpts/openpcdet
ENV WILDDET3D_CKPT_ROOT=/workspace/services/workers/base/ckpts/wilddet3d
ENV YOLO_SOURCE_ROOT=/workspace/services/workers/base/yolo
ENV YOLO_CKPT_ROOT=/workspace/services/workers/base/ckpts/yolo
ENV DATASET_PLUGIN_ROOT=/workspace/services/datasets
ENV PYTHON_VERSION=3.11.15
ENV PYTHON_HOME=/opt/python/3.11.15
ENV VIRTUAL_ENV=/opt/venv
ENV XDG_RUNTIME_DIR=/tmp/runtime-root
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONPATH=/workspace/services:/workspace/services/workers/base:/workspace/services/workers/base/openpcdet

WORKDIR /workspace/services/workers/base/openpcdet

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    g++ \
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

RUN mkdir -p \
    /tmp/runtime-root \
    /workspace/services/workers/base/ckpts/openpcdet \
    /workspace/services/workers/base/ckpts/wilddet3d \
    /workspace/services/workers/base/ckpts/yolo
COPY services/workers/base/openpcdet/requirements.txt /tmp/openpcdet-requirements.txt

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
    && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel
RUN /opt/venv/bin/python -m pip install \
    torch==2.11.0 \
    torchvision==0.26.0 \
    torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN /opt/venv/bin/python -m pip install spconv-cu126

RUN /opt/venv/bin/python -m pip install -r /tmp/openpcdet-requirements.txt
RUN /opt/venv/bin/python -m pip install ultralytics

RUN /opt/venv/bin/python -m pip install vis4d==1.0.0

RUN /opt/venv/bin/python -m pip install git+https://github.com/SysCV/vis4d_cuda_ops.git --no-build-isolation --no-cache-dir

RUN /opt/venv/bin/python -m pip install --no-cache-dir \
    numpy \
    Pillow \
    triton \
    \
    einops \
    timm \
    transformers \
    huggingface_hub \
    \
    ftfy==6.1.1 \
    regex \
    "iopath>=0.1.10" \
    typing_extensions \
    "setuptools<80" \
    \
    opencv-python \
    matplotlib \
    \
    pycocotools \
    pyquaternion \
    scipy \
    terminaltables \
    ml_collections \
    tqdm \
    \
    pyarrow \
    \
    utils3d@git+https://github.com/EasternJournalist/utils3d.git

RUN /opt/venv/bin/python -m pip install "pydantic>=2.7,<3" fastapi==0.115.0

COPY services/workers/base /workspace/services/workers/base
COPY services/datasets /workspace/services/datasets

RUN /opt/venv/bin/python setup.py develop

EXPOSE 8010

CMD ["/opt/venv/bin/python", "-m", "uvicorn", "worker_base.main:app", "--host", "0.0.0.0", "--port", "8010"]
