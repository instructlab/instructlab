#!/bin/bash
# Copyright (c) 2018-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

if [ "${NVIDIA_CPU_ONLY:-0}" -eq 0 ]; then
  ###################################
  # Check current driver version
  ###################################

  #_KMD_VERSION=$(nvidia-smi -q -d COMPUTE | grep "^Driver Version" | sed 's/^.*: //'        2>/dev/null)
  _KMD_VERSION=$(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader           2>/dev/null)

  if [[ "${_KMD_VERSION}" == "[N/A]" ]]; then
    _KMD_VERSION=$(sed -n 's/^NVRM.*Kernel Module\( for [a-z0-9_]*\| \) *\([^() ]*\).*$/\2/p' /proc/driver/nvidia/version 2>/dev/null)
  fi

  # Note: _RUNNING_CUDA_VERSION might have already been set by e.g.
  # 51-gpu-sm-version-check.sh, in which case we don't need to find it again
  _RUNNING_CUDA_VERSION=${_RUNNING_CUDA_VERSION:-$(nvidia-smi -q -d COMPUTE 2>/dev/null | grep "^CUDA Version" | sed 's/^.*: //')}

  _KMD_VERSION_MAJOR=$(echo "${_KMD_VERSION}" | cut -d. -f1)
  _COMPAT_UMD_VERSION_MAJOR=$(echo "${CUDA_DRIVER_VERSION}" | cut -d. -f1)
  _CONTAINER_CUDA_VERSION_MAJOR=$(echo "${CUDA_VERSION}" | cut -d. -f1)
  _CONTAINER_CUDA_VERSION_MINOR=$(echo "${CUDA_VERSION}" | cut -d. -f2)
  _RUNNING_CUDA_VERSION_MAJOR=$(echo "${_RUNNING_CUDA_VERSION}" | cut -d. -f1)
  _RUNNING_CUDA_VERSION_MINOR=$(echo "${_RUNNING_CUDA_VERSION}" | cut -d. -f2)

  if [[ -z "${_KMD_VERSION}" && -e /dev/nvgpu ]]; then
    echo
    echo "NVIDIA Tegra driver detected."

  elif [[ ! "${_KMD_VERSION}" =~ ^[0-9]+(\.[0-9]+)*$ ]]; then
    echo
    echo "Failed to detect NVIDIA driver version."

  elif [[ -n "${_DEVICEQUERY_ERRNUM:-}" ]]; then
    # If 51-gpu-sm-version-check.sh previously noticed that deviceQuery
    # can't even run, then there's basically no way that any of the compat
    # modes will work, and any of the messages from this script will simply
    # add confusion. Therefore we bypass them.
    :

  elif [[ "${_KMD_VERSION_MAJOR}" -lt "${_COMPAT_UMD_VERSION_MAJOR}" ]]; then

    # If the KMD major version number is lower than the container's UMD
    # major version, then either we're in forward compat mode or we're in
    # enhanced compat mode or else things will likely fail.
    if [[ "${_CUDA_COMPAT_STATUS}" == "CUDA Driver OK" ]]; then
      echo
      echo "NOTE: CUDA Forward Compatibility mode ENABLED."
      echo "  Using CUDA ${_RUNNING_CUDA_VERSION} driver version ${CUDA_DRIVER_VERSION} with kernel driver version ${_KMD_VERSION}."
      echo "  See https://docs.nvidia.com/deploy/cuda-compatibility/ for details."

    elif (( test "${_RUNNING_CUDA_VERSION_MAJOR}" -eq "${_CONTAINER_CUDA_VERSION_MAJOR}" ) && \
          ( test "${_RUNNING_CUDA_VERSION_MINOR}" -lt "${_CONTAINER_CUDA_VERSION_MINOR}" )) && \
         (( test -e /dev/nvidiactl && dpkg --compare-versions "${_KMD_VERSION}" ge "450.80.02" ) || \
          ( test -e /dev/dxg       && dpkg --compare-versions "${_KMD_VERSION}" ge "470.31" )); then
      echo
      echo "WARNING: CUDA Minor Version Compatibility mode ENABLED."
      echo "  Using driver version ${_KMD_VERSION} which has support for CUDA ${_RUNNING_CUDA_VERSION}.  This container"
      echo "  was built with CUDA ${_CONTAINER_CUDA_VERSION_MAJOR}.${_CONTAINER_CUDA_VERSION_MINOR} and will be run in Minor Version Compatibility mode."
      echo "  CUDA Forward Compatibility is preferred over Minor Version Compatibility for use"
      echo "  with this container but was unavailable:"
      echo "  [[$_CUDA_COMPAT_STATUS]]"
      echo "  See https://docs.nvidia.com/deploy/cuda-compatibility/ for details."

    else
      echo
      echo "ERROR: This container was built for NVIDIA Driver Release ${CUDA_DRIVER_VERSION%.*} or later, but"
      echo "       version ${_KMD_VERSION} was detected and compatibility mode is UNAVAILABLE."
      echo
      echo "       [[${_CUDA_COMPAT_STATUS}]]"
      sleep 2

    fi
  fi
fi
