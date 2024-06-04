#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -e
# Remove GPU support files that are not necessary for current GPU arch

TORCH="${VIRTUAL_ENV:-/opt/rocm-venv}/lib/${PYTHON:-python3.1?}/site-packages/torch"

case "$AMDGPU_ARCH" in
	gfx9*)
		rm -rf /usr/lib*/rocm/gfx8
		rm -rf /usr/lib*/rocm/gfx10
		rm -rf /usr/lib*/rocm/gfx11
		rm -f /usr/lib*/rocblas/library/*gfx8*
		rm -f /usr/lib*/rocblas/library/*gfx10*
		rm -f /usr/lib*/rocblas/library/*gfx11*
		rm -f "${TORCH}"/lib/rocblas/library/*gfx8*
		rm -f "${TORCH}"/lib/rocblas/library/*gfx10*
		rm -f "${TORCH}"/lib/rocblas/library/*gfx11*
		;;
	gfx10*)
		rm -rf /usr/lib*/rocm/gfx8
		rm -rf /usr/lib*/rocm/gfx9
		rm -rf /usr/lib*/rocm/gfx11
		rm -f /usr/lib*/rocblas/library/*gfx8*
		rm -f /usr/lib*/rocblas/library/*gfx9*
		rm -f /usr/lib*/rocblas/library/*gfx11*
		rm -f "${TORCH}"/lib/rocblas/library/*gfx8*
		rm -f "${TORCH}"/lib/rocblas/library/*gfx9*
		rm -f "${TORCH}"/lib/rocblas/library/*gfx11*
		;;
	gfx11*)
		rm -rf /usr/lib*/rocm/gfx8
		rm -rf /usr/lib*/rocm/gfx9
		rm -rf /usr/lib*/rocm/gfx10
		rm -f /usr/lib*/rocblas/library/*gfx8*
		rm -f /usr/lib*/rocblas/library/*gfx9*
		rm -f /usr/lib*/rocblas/library/*gfx10*
		rm -f "${TORCH}"/lib/rocblas/library/*gfx8*
		rm -f "${TORCH}"/lib/rocblas/library/*gfx9*
		rm -f "${TORCH}"/lib/rocblas/library/*gfx10*
		;;
	*)
		echo "ERROR: $0 unknown AMDGPU_ARCH=$AMDGPU_ARCH"
		exit 2
esac

# find /usr/lib* /opt/ -path '*/*gfx[189][0-9][0-9a-z]*' | grep -v $AMDGPU_ARCH | xargs rm -f
