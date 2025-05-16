#!/usr/bin/env bash
set -e

if [ -z "${PYTHON}" ]; then
    echo "Cannot install ilab when the Python version is unset."
    exit 1
fi

# Validate that the CUDA libs + drivers can be found after configuring CUDA paths
export CUDA_HOME="/usr/local/cuda"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64"
export PATH="$PATH:$CUDA_HOME/bin"
nvidia-smi

$PYTHON -m venv --upgrade-deps venv

# shellcheck disable=SC1091
. venv/bin/activate
$PYTHON -m pip cache remove llama_cpp_python

pip_install="$PYTHON -m pip install -v -c constraints-dev.txt"
CMAKE_ARGS="-DGGML_CUDA=on" $pip_install .

# https://github.com/instructlab/instructlab/issues/1821
# install with Torch and build dependencies installed
$pip_install packaging wheel setuptools-scm
$pip_install .[cuda] -r requirements-vllm-cuda.txt