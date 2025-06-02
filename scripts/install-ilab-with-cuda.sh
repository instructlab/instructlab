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

# pre-install some build dependencies
$pip_install packaging wheel setuptools-scm

# flash-attn has a bug in the setup.py that causes pip to attempt installing it
# before torch is installed. This is a bug because their setup.py depends on
# importing the module, so it should have been listed in build_requires. Alas!
#
# See: https://github.com/Dao-AILab/flash-attention/pull/958
# Also: https://github.com/instructlab/instructlab/issues/1821
#
# first, pre-install flash-attn build dependencies
$pip_install torch packaging setuptools wheel psutil ninja

# now build flash-attn using the pre-installed build dependencies; this will
# guarantee that the build version of torch will match the runtime version of
# torch; otherwise, all kinds of problems may occur, like missing symbols when
# accessing C extensions and such
$pip_install flash-attn --no-build-isolation

CMAKE_ARGS="-DGGML_CUDA=on" $pip_install .
$pip_install .[cuda] -r requirements-vllm-cuda.txt
