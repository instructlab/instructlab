#!/bin/bash
# Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

MIN_CPU_INST_SET=${MIN_CPU_INST_SET:-AVX}

if [[ "$(uname -m)" == "x86_64" ]] && ! cat /proc/cpuinfo | grep flags | sort -u | grep -i ${MIN_CPU_INST_SET} >& /dev/null; then
  echo
  echo "ERROR: This container was built for CPUs supporting at least the ${MIN_CPU_INST_SET} instruction set, but"
  echo "       the CPU detected was $(cat /proc/cpuinfo |grep "model name" | sed 's/^.*: //' | sort -u), which does not report"
  echo "       support for ${MIN_CPU_INST_SET}.  An Illegal Instrution exception at runtime is likely to result."
  if [[ "${MIN_CPU_INST_SET}" == "AVX" ]]; then
    echo "       See https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX ."
  fi
  sleep 2
fi
