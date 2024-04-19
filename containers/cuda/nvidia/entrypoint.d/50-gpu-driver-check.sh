#!/bin/bash
# Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Check if libcuda.so.1 -- the CUDA driver -- is present in the ld.so cache or in LD_LIBRARY_PATH
_LIBCUDA_FROM_LD_CACHE=$(ldconfig -p | grep libcuda.so.1)
_LIBCUDA_FROM_LD_LIBRARY_PATH=$( ( IFS=: ; for i in ${LD_LIBRARY_PATH}; do ls $i/libcuda.so.1 2>/dev/null | grep -v compat; done) )
_LIBCUDA_FOUND="${_LIBCUDA_FROM_LD_CACHE}${_LIBCUDA_FROM_LD_LIBRARY_PATH}"

# Check if /dev/nvidiactl (like on Linux) or /dev/dxg (like on WSL2) or /dev/nvgpu (like on Tegra) is present
_DRIVER_FOUND=$(ls /dev/nvidiactl /dev/dxg /dev/nvgpu 2>/dev/null)

# If either is not true, then GPU functionality won't be usable.
if [[ -z "${_LIBCUDA_FOUND}" || -z "${_DRIVER_FOUND}" ]]; then
  echo
  echo "WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available."
  echo "   Use the NVIDIA Container Toolkit to start this container with GPU support; see"
  echo "   https://docs.nvidia.com/datacenter/cloud-native/ ."
  export NVIDIA_CPU_ONLY=1
fi
