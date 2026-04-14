BUILD_ARGS=()

if [[ -z "${HTTP_PROXY:-}" && -n "${http_proxy:-}" ]]; then
  export HTTP_PROXY="${http_proxy}"
fi
if [[ -z "${HTTPS_PROXY:-}" && -n "${https_proxy:-}" ]]; then
  export HTTPS_PROXY="${https_proxy}"
fi

if [[ -n "${HTTP_PROXY:-}" ]]; then
  BUILD_ARGS+=(--build-arg "HTTP_PROXY=${HTTP_PROXY}")
  BUILD_ARGS+=(--build-arg "http_proxy=${HTTP_PROXY}")
fi
if [[ -n "${HTTPS_PROXY:-}" ]]; then
  BUILD_ARGS+=(--build-arg "HTTPS_PROXY=${HTTPS_PROXY}")
  BUILD_ARGS+=(--build-arg "https_proxy=${HTTPS_PROXY}")
fi

docker build \
  "${BUILD_ARGS[@]}" \
  -t worker-base .
