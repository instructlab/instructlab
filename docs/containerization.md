# Putting InstructLab in a Container AND making it go fast

Containerization of `ilab` allows for portability and ease of setup. With this,
users can now run lab on OpenShift to test the speed of `ilab train` and `generate`
using dedicated GPUs. This guide shows you how to put the `ilab` CLI, all of its
dependencies, and your GPU into a container for an isolated and easily reproducible
experience.

The rest of the document and the `Makefile` assume that you are running a
Fedora-like Linux distribution with Podman and SELinux. If you are running
another Linux distribution with Docker, then run make with `CENGINE=docker`,
substitute `podman` with `docker`, and remove the `:z` suffix from volume
mounts.

## Build the InstructLab container image for CUDA

We encourage you to read the [Containerfile](../containers/cuda/Containerfile).

To reduce the size of the final image, we do a multi-stage build with the _devel_
CUDA image for the build stage and a virtualenv. The base image for the final
stage is the CUDA _runtime_ image, with no build tools and we copy the virtual
environment from the first stage. The layout of the container mimics that of
an `ubi9/python-311` container with virtual environment in `/opt/app-root` and
content in `/opt/app-root/src`.

You will also notice that the entrypoint of the container image is `/opt/app-root/bin/ilab`.
This allows to call the `ilab` command with opening a new shell.

For convenience, we have created a `cuda` target in the Makefile, so building is
as simple as `make cuda`.

The default image name is `localhost/instructlab`, but you can override it
with the `CONTAINER_PREFIX` environment variable. The CUDA version can be change
with `CUDA_VERSION`.

### Configure Podman with NVIDIA container runtime

To configure your machine, you can follow NVIDIA's documentation to
[install NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-yum-or-dnf)
and to [configure Container Device Interface (CDI)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
to expose the GPUs to Podman.

Here is a quick procedure if you haven't.

```shell
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf config-manager --enable nvidia-container-toolkit-experimental
sudo dnf install -y nvidia-container-toolkit
```

Then, you can verify that NVIDIA container toolkit can see your GPUs.

```shell
nvidia-ctk cdi list
```

Example output:

```shell
INFO[0000] Found 2 CDI devices
nvidia.com/gpu=0
nvidia.com/gpu=all
```

Finally, you can generate the CDI configuration for the NVIDIA devices:

```shell
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

### Run the CUDA-accelerated InstructLab container

When running our model, we want to create some paths that will be mounted in
the container to provide data persistence. As an unprivileged user, you will
run a rootless container and map your UID to UID `1001` inside the container
with `--userns keep-id:uid=1001`. The volume mount will contain InstructLab's
configuration, models, taxonomy, and Hugging Face download cache.

```shell
mkdir -p $HOME/.config/instructlab
```

Then, we can run the container, mounting the above folder and using the first
NVIDIA GPU.

```shell
podman run --rm -it --userns keep-id:uid=1001 --device nvidia.com/gpu=0 --volume $HOME/.config/instructlab:/opt/app-root/src:z localhost/instructlab:cuda
```

The above command will give you an interactive shell inside the container.
Let's initialize our configuration, download the model and start the chatbot.

```shell
(app-root) ilab init
(app-root) ilab download
(app-root) ilab chat
```


## Containerfile design

The container files use a multi-stage builder approach to keep the final image small. There are typically several stages:

- The `runtime` stage contains packages and Python virtual environment needed to run InstructLab.
- The `builder` stage has build tools and development files to build and compile additional packages.
- One or several stages fill the virtual environment with PyTorch stack.
- The `final` stage assembles `runtime` and the virtual environment.

`pip install` and `dnf install` use a cache mount to cache downloads. This speeds up rebuilds and shares common packages between builds of multiple containers.

Python byte code `__pycache__` is removed and not created by pip (`PIP_NO_COMPILE`) to reduce the size of the image.

The virtual env is owned by the default user (uid `1001`).
