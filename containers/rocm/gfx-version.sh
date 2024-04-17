#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -e

NAME=$1

if test -z "$NAME"; then
    NAME=$(amdgpu-arch)
fi

case "$NAME" in
	gfx803 | gfx900 | gfx906 | gfx908 | gfx90a | gfx940 | gfx941 | gfx942 | gfx1010 | gfx1012)
        echo "ERROR: No mapping for '$NAME', yet" >&2
        exit 3
        ;;
	gfx1030 | gfx1031 | gfx1032)
        echo "AMDGPU_ARCH=gfx1030"
        echo "HSA_OVERRIDE_GFX_VERSION=10.3.0"
		;;
	gfx1100 | gfx1101 | gfx1102)
        echo "AMDGPU_ARCH=gfx1100"
        echo "HSA_OVERRIDE_GFX_VERSION=11.0.0"
		;;
	*)
		echo "ERROR: unknown or unsupported GFX name '$NAME'" >&2
		exit 2
esac
