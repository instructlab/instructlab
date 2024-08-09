#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

function print_repeats() {
  local -r char="$1" count="$2"
  local i
  for ((i=1; i<=$count; i++)); do echo -n "$char"; done
  echo
}
function print_banner_text() {
  local -r name="$1"
  print_repeats "=" $((${#name} + 6))
  echo "== ${name} =="
  print_repeats "=" $((${#name} + 6))
}

_banner_file="${BASH_SOURCE[0]/%.sh/.txt}"

# 10-banner.sh allows itself to be skipped if there exists a
# 10-banner.txt, which will be cat'd next alphabetically
if [[ ! -f "${_banner_file}" && -n "${NVIDIA_PRODUCT_NAME}" ]]; then
  echo
  print_banner_text "${NVIDIA_PRODUCT_NAME}"
fi
