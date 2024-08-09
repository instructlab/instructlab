#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

if [[ -z "${DISABLE_MOFED_VERSION_WARNING}" ]]; then

  _DETECTED_MOFED=$(cat /sys/module/mlx5_core/version 2>/dev/null || true)
  if [[ -n "${_DETECTED_MOFED}" ]] && \
     [[ "${_DETECTED_MOFED}" != "$(uname -r)" ]] && \
     dpkg --compare-versions "${_DETECTED_MOFED:-0}" lt 4.9
  then
    # If the Mellanox driver is detected, appears to be from MOFED rather than
    # inbox, and is from MOFED < 4.9, then our container won't be compatible with
    # that old driver, and we issue a warning.

    echo
    echo "ERROR: Detected Mellanox network driver ${_DETECTED_MOFED}, but version 4.9 or higher"
    echo "       is required for multi-node operation support with this container."
    sleep 2
  fi

  _DETECTED_NVPEERMEM=$(cat /sys/kernel/mm/memory_peers/nv_mem/version 2>/dev/null ||
                        cat /sys/kernel/mm/memory_peers/nvidia-peermem/version 2>/dev/null || true)
  if [[ -n "${_DETECTED_MOFED}" && -z "${_DETECTED_NVPEERMEM}" ]]; then
    echo
    echo "NOTE: Mellanox network driver detected, but NVIDIA peer memory driver not"
    echo "      detected.  Multi-node communication performance may be reduced."
  fi

fi
