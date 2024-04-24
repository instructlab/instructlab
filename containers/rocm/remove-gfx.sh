#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -e
# Remove GPU support files that are not necessary for current GPU arch

TORCH="${VIRTUAL_ENV:-/opt/rocm-venv}/lib/${PYTHON:-python3.1?}/site-packages/torch"

rmgfx() {
	for GFX in "$@"; do
		echo "Removing ${GFX} files"
		rm -rfv "/usr/lib*/rocm/${GFX}"
		rm -fv "/usr/lib*/rocblas/library/*${GFX}*"
		rm -fv "/opt/rocm/lib/rocblas/library/*${GFX}*"
		rm -fv "${TORCH}/lib/rocblas/library/*${GFX}*"
		rm -fv "${TORCH}/lib/hipblaslt/library/*${GFX}*"
	done
}

case "$AMDGPU_ARCH" in
	gfx9*)
		rmgfx gfx8 gfx10 gfx11
		;;
	gfx10*)
		rmgfx gfx8 gfx9 gfx11
		;;
	gfx11*)
		rmgfx gfx8 gfx9 gfx10
		;;
	*)
		echo "ERROR: $0 unknown AMDGPU_ARCH=$AMDGPU_ARCH"
		exit 2
esac

# find /usr/lib* /opt/ -path '*/*gfx[189][0-9][0-9a-z]*' | grep -v $AMDGPU_ARCH | xargs rm -f
