FROM wangzhaoze/radar-annotation-toolkit-worker-base-deps:cu12.8-cudnn-devel-ubuntu22.04-py3.11

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ARG NO_PROXY
ARG no_proxy

ENV WORKER_SOURCE_ROOT=/workspace/services/workers/base \
    WORKER_CKPT_ROOT=/workspace/services/workers/base/ckpts \
    OPENPCDET_SOURCE_ROOT=/workspace/services/workers/base/openpcdet \
    OPENPCDET_CKPT_ROOT=/workspace/services/workers/base/ckpts/openpcdet \
    WILDDET3D_SOURCE_ROOT=/workspace/services/workers/base/WildDet3D \
    WILDDET3D_CKPT_ROOT=/workspace/services/workers/base/ckpts/wilddet3d \
    YOLO_SOURCE_ROOT=/workspace/services/workers/base/yolo \
    YOLO_CKPT_ROOT=/workspace/services/workers/base/ckpts/yolo \
    DATASET_PLUGIN_ROOT=/workspace/services/datasets \
    PYTHONPATH=/workspace/services:/workspace/services/workers/base:/workspace/services/workers/base/openpcdet

WORKDIR /workspace/services/workers/base/openpcdet

RUN mkdir -p \
    /tmp/runtime-root \
    /workspace/services/workers/base/ckpts/openpcdet \
    /workspace/services/workers/base/ckpts/wilddet3d \
    /workspace/services/workers/base/ckpts/yolo

COPY services/workers/base /workspace/services/workers/base
COPY services/datasets /workspace/services/datasets

RUN /opt/venv/bin/python setup.py develop

EXPOSE 8010

CMD ["/opt/venv/bin/python", "-m", "uvicorn", "worker_base.main:app", "--host", "0.0.0.0", "--port", "8010"]
