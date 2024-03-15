# Putting `lab` in a Container AND making it go fast 

Containerization of `lab` allows for portability and ease of setup. With this, users can now run lab on OpenShift to test the speed of `lab train` and `generate` using dedicated GPUs. This guide shows you how to put the `lab`CLI, all of its dependencies,
and your GPU into a container for an isolated and easily reproducible experience.

Currently we have two processes for containerization: ROCM and CUDA Containerfiles. Check out [/containers](/containers) in this repository for more information about your specific architecture.