# Containers

This directory contains Containerfiles which can be used to easily bring up a development environment for use with this repository.

`Containerfile.toolbox-rocm` is suitable for use with hosts equipped with AMD ROCm GPUs. This Containerfile is intended for use with [Toolbox](https://containertoolbx.org/install/).

## Building

With this repository cloned and in the root of this repository:

1. Identify which AMD GPU you have on your machine. See: [GPU
   acceleration](../docs/gpu-acceleration.md) for what this is and how to
   figure it out.
2. Build the image: `podman build --build-arg=AMDGPU_TARGETS=<your GPU identifier> -t instructlab:rocm --file=./containers/Containerfile.toolbox-rocm .`

## Running

1. Create a new toolbox container using this image: `toolbox create --image localhost/instructlab:rocm instructlab`
2. Enter your toolbox container: `toolbox enter instructlab`
3. Create your work directory: `mkdir instructlab`
