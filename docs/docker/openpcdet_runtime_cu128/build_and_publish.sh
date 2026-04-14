#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

DOCKERHUB_NAMESPACE="${DOCKERHUB_NAMESPACE:?Set DOCKERHUB_NAMESPACE before running this script.}"
IMAGE_REPOSITORY="${IMAGE_REPOSITORY:-radarannotationtoolkit-openpcdet-runtime}"
IMAGE_TAG="${IMAGE_TAG:-cu128-v1}"
IMAGE_REF="${DOCKERHUB_NAMESPACE}/${IMAGE_REPOSITORY}:${IMAGE_TAG}"
PUSH="${PUSH:-0}"

export DOCKER_BUILDKIT=1

if [[ -z "${HTTP_PROXY:-}" && -n "${http_proxy:-}" ]]; then
  export HTTP_PROXY="${http_proxy}"
fi
if [[ -z "${HTTPS_PROXY:-}" && -n "${https_proxy:-}" ]]; then
  export HTTPS_PROXY="${https_proxy}"
fi
if [[ -z "${NO_PROXY:-}" && -n "${no_proxy:-}" ]]; then
  export NO_PROXY="${no_proxy}"
fi

BUILD_ARGS=()
if [[ -n "${HTTP_PROXY:-}" ]]; then
  BUILD_ARGS+=(--build-arg "HTTP_PROXY=${HTTP_PROXY}")
  BUILD_ARGS+=(--build-arg "http_proxy=${HTTP_PROXY}")
fi
if [[ -n "${HTTPS_PROXY:-}" ]]; then
  BUILD_ARGS+=(--build-arg "HTTPS_PROXY=${HTTPS_PROXY}")
  BUILD_ARGS+=(--build-arg "https_proxy=${HTTPS_PROXY}")
fi
if [[ -n "${NO_PROXY:-}" ]]; then
  BUILD_ARGS+=(--build-arg "NO_PROXY=${NO_PROXY}")
  BUILD_ARGS+=(--build-arg "no_proxy=${NO_PROXY}")
fi

docker build \
  --progress=plain \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_REF}" \
  "${BUILD_ARGS[@]}" \
  "${REPO_ROOT}"

if [[ "${PUSH}" == "1" ]]; then
  docker push "${IMAGE_REF}"
fi
