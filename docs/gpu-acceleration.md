# 🏎️ Making `ilab` go fast

By default, `ilab` will attempt to use your GPU for inference and synthesis. This
works on a wide variety of common systems, but less-common configurations may
require some additional tinkering to get it enabled. This document aims to
describe how you can GPU-accelerate `ilab` on a variety of different
environments.

`ilab` relies on two Python packages that can be GPU accelerated: `torch`
and `llama-cpp-python`. In short, you'll need to replace the default versions of
these packages with versions that have been compiled for GPU-specific support,
recompile `ilab`, then run it.

## Python 3.11 (Linux only)

> **NOTE:** This section may be outdated. At least AMD ROCm works fine with
> Python 3.12 and Torch 2.2.1+rocm5.7 binaries.

Unfortunately, at the time of writing, `torch` does not have GPU-specific
support for the latest Python (3.12), so if you're on Linux, it's recommended
to set up a Python 3.11-specific `venv` and install `ilab` to that to minimize
issues. (MacOS ships Python 3.9, so this step shouldn't be necessary.) Here's
how to do that on Fedora with `dnf`:

  ```shell
  # Install Python 3.11
  sudo dnf install python3.11 python3.11-devel

  # Remove old venv from instructlab/ directory (if it exists)
  rm -r venv

  # Create and activate new Python 3.11 venv
  python3.11 -m venv venv
  source venv/bin/activate

  # Install lab (assumes a locally-cloned repo)
  # You can clone the repo if you haven't already done so (either one)
  # gh repo clone instructlab/instructlab
  # git clone https://github.com/instructlab/instructlab.git
  pip install ./instructlab[cuda]
  ```

With Python 3.11 installed, it's time to replace some packages!

### llama-cpp-python backends

Go to the project's GitHub to see
the [supported backends](https://github.com/abetlen/llama-cpp-python/tree/v0.2.79?tab=readme-ov-file#supported-backends).

Whichever backend you choose, you'll see a `pip install` command. First
you have to purge pip's wheel cache to force a rebuild of llama-cpp-python:

 ```shell
 pip cache remove llama_cpp_python
 ```

You'll want to add a few options to ensure it gets installed over the
existing package, has the desired backend, and the correct version.

```shell
pip install --force-reinstall --no-deps llama_cpp_python==0.2.79 -C cmake.args="-DLLAMA_$BACKEND=on"
```

where `$BACKEND` is one of `HIPBLAS` (ROCm), `CUDA`, `METAL`
(Apple Silicon MPS), `CLBLAST` (OpenCL), or another backend listed in
llama-cpp-python's documentation.

### Nvidia/CUDA

`torch` should already ship with CUDA support, so you only have to replace
`llama-cpp-python`.

Ensure you have the latest proprietary Nvidia drivers installed.  You can
easily validate whether you are using `nouveau` or `nvidia` kernel drivers with
the following command.  If your output shows `Kernel driver in use: nouveau`,
you are **not running** with the proprietary Nvidia drivers.

```shell
# Check video driver
sudo dnf install pciutils
lspci -n -n -k | grep -A 2 -e VGA -e 3D
```

If needed, install the proprietary NVidia drivers

```shell
# Enable RPM Fusion Repos
sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm

# Install Nvidia Drivers

# There may be extra steps for enabling secure boot.  View the following blog for further details: https://blog.monosoul.dev/2022/05/17/automatically-sign-nvidia-kernel-module-in-fedora-36/

sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda

# Reboot to load new kernel drivers
sudo reboot

# Check video driver
lspci -n -n -k | grep -A 2 -e VGA -e 3D
```

You should now see `Kernel driver in use: nvidia`. The next step is to ensure
CUDA 12.4 is installed.

```shell
# Install CUDA 12.4 and nvtop to monitor GPU usage
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/cuda-fedora39.repo

sudo dnf clean all
sudo dnf -y install cuda-toolkit-12-4 nvtop
```

Go to the project's GitHub to see the
[supported backends](https://github.com/abetlen/llama-cpp-python/tree/v0.2.79?tab=readme-ov-file#supported-backends).
Find the `CUDA` backend. You'll see a `pip install` command.
You'll want to add a few options to ensure it gets installed over the
existing package: `--force-reinstall`. Your final
command should look like this:

```shell
# Verify CUDA can be found in your PATH variable
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin

# Recompile llama-cpp-python using CUDA
pip cache remove llama_cpp_python
pip install --force-reinstall llama_cpp_python==0.2.79 -C cmake.args="-DLLAMA_CUDA=on"

# Re-install InstructLab
pip install ./instructlab[cuda]
```

If you are running Fedora 40, you need to replace the `Recompile llama-cpp-python using CUDA` section above with the
following until [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#host-compiler-support-policy)
supports GCC v14.1+.

```shell
# Recompile llama-cpp-python using CUDA
sudo dnf install clang17
CUDAHOSTCXX=$(which clang++-17) pip install --force-reinstall llama_cpp_python==0.2.79 -C cmake.args="-DLLAMA_CUDA=on"
```

Proceed to the `Initialize` section of
the [CLI README](https://github.com/instructlab/instructlab?tab=readme-ov-file#%EF%B8%8F-initialize-ilab),
and use the `nvtop` utility to validate GPU utilization when interacting
with `ilab model chat` or `ilab data generate`

### AMD/ROCm

Your user account must be in the `video` and `render` group to have permission
to access the GPU hardware. If the `id` command does not show both groups, then
run the following command. You have to log out log and log in again to refresh
your current user session.

```shell
sudo usermod -a -G render,video $LOGNAME
```

#### ROCm container

The most convenient approach is the [ROCm toolbox container](amd-rocm.md). The container comes with PyTorch, llama-cpp, and other dependencies pre-installed and ready-to-use.

#### Manual installation

`torch` does not yet ship with AMD ROCm support, so you'll need to install a version compiled with support.

Visit [PyTorch "Get Started Locally" page](https://pytorch.org/get-started/locally/)
and use the matrix installer tool to find the ROCm package. `Stable, Linux, Pip,
Python, ROCm 5.7` in the matrix installer spits out the following command:

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

You don't need `torchvision` or `torchaudio`, so get rid of those. You also want
to make _very_ sure you're installing the right package, and not the old one
that doesn't have GPU support, so you should add these options:
`--force-reinstall` and `--no-cache-dir`. Your command should look like below.
Run it to install the new version of `torch`.

```shell
pip install torch --force-reinstall --no-cache-dir --index-url https://download.pytorch.org/whl/rocm6.0
```

With that done, it's time to move on to `llama-cpp-python`.

#### hipBLAS

If using hipBLAS you may need to install additional ROCm and hipBLAS
Dependencies:

```shell
# Optionally enable repo.radeon.com repository, available through AMD documentation or Radeon Software for Linux for RHEL 9.3 at https://www.amd.com/en/support/linux-drivers
# The above will get you the latest 6.x drivers, and will not work with rocm5.7 pytorch
# to grab rocm 5.7 drivers: https://repo.radeon.com/amdgpu-install/23.30.3/rhel/9.2/
# ROCm Dependencies
sudo dnf install rocm-dev rocm-utils rocm-llvm rocminfo

# hipBLAS dependencies
sudo dnf install hipblas-devel hipblas rocblas-devel
```

With those dependencies installed, you should be able to install (and build)
`llama-cpp-python`!

You can use `rocminfo | grep gfx` from `rocminfo` package or `amdgpu-arch` from
`clang-tools-extra` package to find our GPU model to include in the build
command - this may not be necessary in Fedora 40+ or ROCm 6.0+.  You should see
something like the following if you have an AMD Integrated and Dedicated GPU:

```shell
$ rocminfo | grep gfx
  Name:                    gfx1100
      Name:                    amdgcn-amd-amdhsa--gfx1100
  Name:                    gfx1036
      Name:                    amdgcn-amd-amdhsa--gfx103
```

In this case, `gfx1100` is the model we're looking for (our dedicated GPU) so
we'll include that in our build command as follows:

```shell
export PATH=/opt/rocm/llvm/bin:$PATH
pip cache remove llama_cpp_python
CMAKE_ARGS="-DLLAMA_HIPBLAS=on -DCMAKE_C_COMPILER='/opt/rocm/llvm/bin/clang' -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm -DAMDGPU_TARGETS=gfx1100" FORCE_CMAKE=1 pip install --force-reinstall llama_cpp_python==0.2.79
```

> **Note:** This is explicitly forcing the build to use the ROCm compilers and
> prefix path for dependency resolution in the CMake build.  This works around
> an issue in the CMake and ROCm version in Fedora 39 and below and is fixed in
> Fedora 40.  With Fedora 40's ROCm packages, use
> `CMAKE_ARGS="-DLLAMA_HIPBLAS=on -DCMAKE_C_COMPILER=/usr/bin/clang
> -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DAMDGPU_TARGETS=gfx1100"` instead.

Once that package is installed, recompile `ilab` with `pip install .[rocm]`.  You also
need to tell `HIP` which GPU to use - you can find this out via `rocminfo`
although it is typically GPU 0.  To set which device is visible to HIP, we'll
set `export HIP_VISIBLE_DEVICES=0` for GPU 0.   You may also have to set
`HSA_OVERRIDE_GFX_VERSION` to override ROCm GFX version detection, for example
`export HSA_OVERRIDE_GFX_VERSION=10.3.0` to force an unsupported `gfx1032` card
to use use supported `gfx1030` version.  The environment variable
`AMD_LOG_LEVEL` enables debug logging of ROCm libraries, for example
`AMD_LOG_LEVEL=3` to print API calls to `stderr`.

Now you can skip to the `Testing` section.

#### CLBlast (OpenCL)

Your final command should look like so (this uses `CLBlast`):

```shell
pip cache remove llama_cpp_python
pip install --force-reinstall llama_cpp_python==0.2.79 -C cmake.args="-DLLAMA_CLBLAST=on"
```

Once that package is installed, recompile `ilab` with `pip install .` and skip
to the `Testing` section.

### Metal/Apple Silicon

The `ilab` default installation should have Metal support by default. If that
isn't the case, these steps might help to enable it.

`torch` should already ship with Metal support, so you only have to
replace `llama-cpp-python`. Go to the project's GitHub to see the
[supported backends](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends).
Find the `Metal` backend. You'll see a `pip install` command. You'll want to
add a few options to ensure it gets installed over the existing package:
`--force-reinstall` and `--no-cache-dir`. Your final command should look like so:

```shell
pip cache remove llama_cpp_python
pip install --force-reinstall llama_cpp_python==0.2.79 -C cmake.args="-DLLAMA_METAL=on"
```

Once that package is installed, recompile `ilab` with `pip install .[mps]` and skip
to the `Testing` section.

### Testing

Test your changes by chatting to the LLM. Run `ilab model serve` and `ilab model chat` and
chat to the LLM. If you notice significantly faster inference, congratulations!
You've enabled GPU acceleration. You should also notice that the `ilab data generate`
step will take significantly less time.  You can use tools like `nvtop` and
`radeontop` to monitor GPU usage.

Use the scripts `containers/bin/debug-pytorch` and `containers/bin/debug-llama` to verify that PyTorch and llama-cpp are able to use your GPU.

The `torch` and `llama_cpp` packages provide functions to debug GPU support.  Here is an example from an AMD ROCm system with a single GPU, ROCm build of PyTorch and llama-cpp with HIPBLAS.  Don't be confused by the fact that PyTorch uses `torch.cuda` API for ROCm or llama-cpp reports hipBLAS as cuBLAS.  The packages treat ROCm like a variant of CUDA.

```python
>>> import torch
>>> torch.__version__
'2.2.1+rocm5.7'
>>> torch.version.cuda or 'n/a'
'n/a'
>>> torch.version.hip or 'n/a'
'5.7.31921-d1770ee1b'
>>> torch.cuda.is_available()
True
>>> torch.cuda.device_count()
1
>>> torch.cuda.get_device_name(torch.cuda.current_device())
'AMD Radeon RX 7900 XT'
```

```python
>>> import llama
>>> llama_cpp.__version__
'0.2.56'
>>> llama_cpp.llama_supports_gpu_offload()
True
>>> llama_cpp.llama_backend_init()
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 ROCm devices:
  Device 0: AMD Radeon RX 7900 XT, compute capability 11.0, VMM: no
```

## Training

`ilab model train`  also experimentally supports GPU acceleration on Linux. Details
of a working set up is included above. Training is memory-intensive and requires
a modern GPU to work. The GPU must support `bfloat16` or `fp16` and have at
least 17 GiB of free GPU memory. Nvidia CUDA on WSL2 is able to use shared host
memory (USM) if GPU memory is not sufficient, but that comes with a performance
penalty. Training on Linux Kernel requires all data to fit in GPU memory. We are
working on improvements like 4-bit quantization.

It has been successfully tested on:

- Nvidia GeForce RTX 3090 (24 GiB), Fedora 39, PyTorch 2.2.1 CUDA 12.1
- Nvidia GeForce RTX 3060 Ti (8 GiB + 9 GiB shared), Fedora 39 on WSL2, CUDA 12.1
- Nvidia Tesla V100 (16 GB) on AWS `p3.2xlarge`, Fedora 39, PyTorch 2.2.1, 4-bit quantization
- AMD Radeon RX 7900 XT (20 GiB), Fedora 39, PyTorch 2.2.1+rocm5.7
- AMD Radeon RX 7900 XTX (24 GiB), Fedora 39, PyTorch 2.2.1+rocm5.7
- AMD Radeon RX 6700 XT (12 GiB), Fedora 39, PyTorch 2.2.1+rocm5.7, 4-bit
quantization

Incompatible devices:

- NVidia cards with Turing architecture (GeForce RTX 20 series) or older. They
  lack support for `bfloat16` and `fp16`.

> **Note:** PyTorch implements AMD ROCm support on top of its `torch.cuda` API
> and treats AMD GPUs as CUDA devices. In a ROCm build of PyTorch, `cuda:0` is
> actually the first ROCm device.
<!-- -->
> **Note:** Training does not use a local lab server. You can stop `ilab model serve`
> to free up GPU memory.

```shell
ilab model train --device cuda
```

```shell
LINUX_TRAIN.PY: PyTorch device is 'cuda:0'
  NVidia CUDA version: n/a
  AMD ROCm HIP version: 5.7.31921-d1770ee1b
  Device 'cuda:0' is 'AMD Radeon RX 7900 XT'
  Free GPU memory: 19.9 GiB of 20.0 GiB
LINUX_TRAIN.PY: NUM EPOCHS IS:  1
...
```
