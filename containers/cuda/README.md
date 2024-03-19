# Running `lab` in a Container with NVIDIA GPU Acceleration

## Steps to build an image then run a container:

**Containerfile:**

```dockerfile
FROM nvcr.io/nvidia/cuda:12.3.2-devel-ubi9
RUN dnf install -y python3.11 && dnf install -y openssh && dnf install -y git && dnf install -y python3-pip && dnf install -y make automake gcc gcc-c++
RUN ssh-keyscan github.com > ~/.ssh/known_hosts
WORKDIR /instruct-lab
RUN python3.11 -m ensurepip
RUN dnf install -y gcc
RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm
RUN dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo && dnf repolist && dnf config-manager --set-enabled cuda-rhel9-x86_64 && dnf config-manager --set-enabled cuda && dnf config-manager --set-enabled epel && dnf update -y
RUN --mount=type=ssh,id=default python3.11 -m pip install --force-reinstall nvidia-cuda-nvcc-cu12 
RUN export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64" \
    && export CUDA_HOME=/usr/local/cuda \
    && export PATH="/usr/local/cuda/bin:$PATH" \
    && export XLA_TARGET=cuda120 \
    && export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
RUN --mount=type=ssh,id=default CMAKE_ARGS="-DLLAMA_CUBLAS=on" python3.11 -m pip install --force-reinstall --no-cache-dir llama-cpp-python 
RUN --mount=type=ssh,id=default python3.11 -m pip install git+ssh://git@github.com/instruct-lab/cli.git@stable
CMD ["/bin/bash"]
```

Or image: TBD (am I allowed to have a public image with references to lab in it?)

This containerfile is based on Nvidia's CUDA image, which plugs directly into Podman via their `nvidia-container-toolkit`! The ubi9 base image does not have most packages installed. The bulk of the `containerfile` is spent configuring your system so `lab` can install and run properly. The ubi9 base image as compared to ubuntu cannot install the entire nvidia-12-4 toolkit. This did not impact performance during testing. You can run the following steps to build an image then run a container:

1. Podman build –ssh=default -f <Containerfile_Path>
2. curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo |   sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
3. sudo yum-config-manager --enable nvidia-container-toolkit-experimental
4. sudo dnf install -y nvidia-container-toolkit
5. sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
6. nvidia-ctk cdi list
    Example output: 
    INFO[0000] Found 2 CDI devices                     	 
    nvidia.com/gpu=0
    nvidia.com/gpu=all
7. podman run --device nvidia.com/gpu=0  --security-opt=label=disable -it <IMAGE_ID>

Voila! You now have a container with CUDA and GPUs enabled!

#### Sources:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html – nvidia container toolkit
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html podman

#### Notes:
Thanks to Taj Salawu for figuring out how to pass the git ssh keys properly!
