# Putting `ilab` in a Container AND making it go fast

Containerization of `ilab` allows for portability and ease of setup. With this,
users can now run lab on OpenShift to test the speed of `ilab model train` and `generate`
using dedicated GPUs. This guide shows you how to put the `ilab` CLI, all of its
dependencies, and your GPU into a container for an isolated and easily reproducible
experience.

## Steps to build an image then run a container

The [`Containerfile`](../containers/cuda/Containerfile)
is based on Nvidia CUDA image, which plugs
directly into Podman via their `nvidia-container-toolkit`! The `ubi9` base
image does not have most packages installed. The bulk of the `Containerfile` is
spent configuring your system so `ilab` can be installed and run properly.
`ubi9` as compared to `ubuntu` cannot install the entire `nvidia-12-4` toolkit.
This did not impact performance during testing.

The CUDA Containerfile is located [here](../containers/cuda/Containerfile).

Voila! You now have a container with CUDA and GPUs enabled!

### Sources

[Nvidia Container Toolkit Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

[Podman Support for Container Device Interface](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
