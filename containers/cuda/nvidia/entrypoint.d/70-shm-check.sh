#!/bin/bash
# Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

if [[ "$(df -k /dev/shm |grep ^shm |awk '{print $2}') " == "65536 " ]]; then
  echo
  echo "NOTE: The SHMEM allocation limit is set to the default of 64MB.  This may be"
  echo "   insufficient for ${NVIDIA_PRODUCT_NAME:-}.  NVIDIA recommends the use of the following flags:"
  echo "   docker run --gpus all ${NVIDIA_SHM_FLAGS:---ipc=host --ulimit memlock=-1 --ulimit stack=67108864} ..."
fi
