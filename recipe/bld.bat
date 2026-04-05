@echo off
setlocal enabledelayedexpansion

set CUDA_ARGS=
if "%gpu_variant%"=="cuda" (
    set "CUDA_ARGS=-Ccmake.define.USE_CUDA=ON"
    set CUDA_HOME=%BUILD_PREFIX%\Library
    set CUDA_PATH=%BUILD_PREFIX%\Library
)

%PYTHON% -m pip install . --no-deps --no-build-isolation -vv ^
    -Ccmake.define.CMAKE_GENERATOR="Visual Studio 17 2022" ^
    %CUDA_ARGS%
if errorlevel 1 exit 1
