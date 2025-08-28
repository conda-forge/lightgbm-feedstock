# For CPU builds, we include OpenCL too. USE_GPU means OpenCL and not CUDA builds
if [[ "$cuda_compiler_version" == "None" ]]; then
  export CMAKE_ARGS="${CMAKE_ARGS} -DUSE_GPU=1"
else
  export CMAKE_ARGS="${CMAKE_ARGS} -DUSE_CUDA=1"
fi

mkdir build
cd build

cmake ${CMAKE_ARGS} -DBUILD_CLI=OFF -DUSE_HOMEBREW_FALLBACK=OFF ..
make -j${CPU_COUNT}
make install

