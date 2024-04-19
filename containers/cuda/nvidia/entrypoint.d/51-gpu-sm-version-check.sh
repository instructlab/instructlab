#!/bin/bash
# Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

if [ "${NVIDIA_CPU_ONLY:-0}" -eq 0 ]; then
  ###################################
  # Query the available devices to see if they are within the range
  # of supported CUDA compute capability ("SM") versions
  ###################################

  # Specify default min and max SM versions to check for
  MINSMVER=${MINSMVER:-52}
  MAXSMVER=${MAXSMVER:-90}

  _HAS_SUPPORTED_SMVER=0
  _DEVICEQUERY_ERRNUM=""
  _MAJMIN_REGEX_DEC="([[:digit:]]+\.[[:digit:]]+)"    # match e.g. 8.6, 9.0, 10.0, ...
  _MAJMIN_REGEX_NO_DEC="([[:digit:]]+)([[:digit:]])"  # match e.g. 86, 90, 100, ...
  _DEVICENAME_REGEX="Device [[:digit:]]*: \"([[:print:]]*)\""
  _SM_VERSION_REGEX="CUDA Capability Major/Minor version number: *${_MAJMIN_REGEX_DEC}"
  _RUNNING_CUDA_VERSION_REGEX=" CUDA Driver.*Version *${_MAJMIN_REGEX_DEC} /"
  _DEVICEQUERY_ERROR_REGEX="CUDA Driver API error ([[:digit:]?]+) \"(.*)\""

  [[ "${MINSMVER}" =~ ${_MAJMIN_REGEX_NO_DEC} ]]
  _MINSMVER_DEC="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
  [[ "${MAXSMVER}" =~ ${_MAJMIN_REGEX_NO_DEC} ]]
  _MAXSMVER_BUILT_DEC="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
  _MAXSMVER_SUPPORT_DEC="${BASH_REMATCH[1]}.999"  # even if we're built for up to, say, SM 8.6, we can still work on SM 8.9

  # Detect SM versions present in the current system
  while read i
  do
    if [[ "$i" =~ ${_DEVICENAME_REGEX} ]]; then
      _DEVICE_NAME=${BASH_REMATCH[1]}
    elif [[ "$i" =~ ${_RUNNING_CUDA_VERSION_REGEX} ]]; then
      _RUNNING_CUDA_VERSION="${BASH_REMATCH[1]}"
    elif [[ "$i" =~ ${_DEVICEQUERY_ERROR_REGEX} ]]; then
      _DEVICEQUERY_ERRNUM="${BASH_REMATCH[1]}"
      _DEVICEQUERY_ERRSTR="${BASH_REMATCH[2]}"
    elif [[ "$i" =~ ${_SM_VERSION_REGEX} ]]; then
      _DEVICE_SM_VER_DEC="${BASH_REMATCH[1]}"
      if dpkg --compare-versions "${_DEVICE_SM_VER_DEC}" "ge" "${_MINSMVER_DEC}"; then
        if dpkg --compare-versions "${_DEVICE_SM_VER_DEC}" "le" "${_MAXSMVER_BUILT_DEC}"; then
          ((_HAS_SUPPORTED_SMVER++))
        elif dpkg --compare-versions "${_DEVICE_SM_VER_DEC}" "le" "${_MAXSMVER_SUPPORT_DEC}"; then
          echo "WARNING: Detected ${_DEVICE_NAME} GPU, which may not yet be supported in this version of the container"
          ((_HAS_SUPPORTED_SMVER++))
        else
          echo "WARNING: Detected ${_DEVICE_NAME} GPU, which is not yet supported in this version of the container"
        fi
      else
        echo "ERROR: Detected ${_DEVICE_NAME} GPU, which is not supported by this container"
      fi
    fi
  done < <(deviceQuery | grep -i '\(^Device\|version\|^CUDA Driver API error\)')
  if [ -n "${_DEVICEQUERY_ERRNUM}" ]; then
    echo
    echo "ERROR: The NVIDIA Driver is present, but CUDA failed to initialize.  GPU functionality will not be available."
    echo "   [[ ${_DEVICEQUERY_ERRSTR^} (error ${_DEVICEQUERY_ERRNUM}) ]]"
    # Special case: attempt to better detect when MIG mode is enabled but no MIG instances are created.
    # At least up to and including R530, all we get back from the CUDA Driver in this case is error 3,
    # CUDA_ERROR_NOT_INITIALIZED, which isn't very specific
    if [[ "${_DEVICEQUERY_ERRNUM}" == "3" ]] && nvidia-smi mig -lgi |& grep -q "Failed to display GPU instances"; then
      echo
      echo "   [[ Possible MIG misconfiguration detected ]]"
    fi
  elif [ ${_HAS_SUPPORTED_SMVER} -eq 0 ]; then
    echo "ERROR: No supported GPU(s) detected to run this container"
  fi
fi
