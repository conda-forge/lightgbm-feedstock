if "%cuda_compiler_version%" == "None" (
  set "CMAKE_ARGS=%CMAKE_ARGS% -DUSE_GPU=1"
) else (
  set "CMAKE_ARGS=%CMAKE_ARGS% -DUSE_CUDA=1"
)

mkdir build
cd build

set "CMAKE_GENERATOR_PLATFORM="
set "CMAKE_GENERATOR_TOOLSET="
set "CMAKE_GENERATOR=Ninja"

cmake -G Ninja %CMAKE_ARGS% -DBUILD_CLI=OFF ..
ninja -j%CPU_COUNT%
ninja install
