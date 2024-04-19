# Putting `ilab` in a Container AND making it go fast

Containerization of `ilab` allows for portability and ease of setup. With this,
users can now run lab on OpenShift to test the speed of `ilab train` and `generate`
using dedicated GPUs. This guide shows you how to put the `ilab` CLI, all of its
dependencies, and your GPU into a container for an isolated and easily reproducible
experience.

## Build the `ilab` container image

We encourage you to read the [Containerfile](../containers/cuda/Containerfile).
understand the following explanation.

To reduce the size of the final image, we do a multi-stage build with the _devel_
CUDA image for the build stage and a virtualenv. The base image for the final
stage is the CUDA _runtime_ image, with no build tools and we copy the `site-packages`
folder from the first stage.

You will also notice that the entrypoint of the container image is `/opt/app-root/bin/ilab`.
This allows to call the `ilab` command with opening a new shell. See below how it
is beneficial when combined with an alias.

For convenience, we have created a `cuda` target in the Makefile, so building is
as simple as `make cuda`.

The default image name is `localhost/instructlab`, but you can override it with
the `CONTAINER_PREFIX` environment variable.

## Configure Podman with NVIDIA container runtime

To configure your machine running RHEL 9.4+, you can follow NVIDIA's documentation to
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

## Run the GPU-accelerated `ilab` container

When running our model, we want to create some paths that will be mounted in
the container to provide data persistence. As an unprivileged user, you will
run a rootless container, so you need to let the internal user (UID 1001)
access the files in the host. For that, we use `podman unshare chown`.

```shell
mkdir -p ${HOME}/.ilab
podman unshare chown 1001:1001 -R ${HOME}/.ilab
```

Then, we can run the container, mounting the above folder and using the first
NVIDIA GPU.

```shell
podman run --rm -it --user 1001 --device nvidia.com/gpu=0 --volume ${HOME}/.ilab:/opt/app-root/ilab:Z localhost/instructlab:cuda
```

The above command will print the help, as we didn't pass any argument.

Let's initialize our configuration, download the model and start the chatbot.

```
podman run --rm -it --device nvidia.com/gpu=0 --volume ${HOME}/.ilab:/opt/appr-root/ilab:Z localhost/instructlab:cuda init
podman run --rm -it --device nvidia.com/gpu=0 --volume ${HOME}/.ilab:/opt/app-root/ilab:Z localhost/instructlab:cuda download
podman run --rm -it --device nvidia.com/gpu=0 --volume ${HOME}/.ilab:/opt/app-root/ilab:Z localhost/instructlab:cuda chat
```

## Creating an alias

Now that you know how to run the container, you probably find it cumbersome
to type the long `podman run` command, so we provide an alias definition in
the [containers/cuda/instructlab-cuda.alias](../containers/cuda/instructlab-cuda.alias)
file.

You simply need to put it in your `${HOME}/.bashrc.d` folder and restart your
bash shell to be able to call only `instructlab` or `ilab`.
