#!/bin/bash
set -exo pipefail

CUDA_ARGS=""
if [[ "${gpu_variant}" == "cuda" ]]; then
    CUDA_ARGS="-Ccmake.define.USE_CUDA=ON"
fi

${PYTHON} -m pip install . --no-deps --no-build-isolation -vv \
    -Ccmake.define.CMAKE_GENERATOR="Ninja" \
    ${CUDA_ARGS}
