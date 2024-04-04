# instruct-lab toolbox container for AMD ROCm GPUs

The ROCm container file is designed for AMD GPUs with RDNA3 architecture (`gfx1100`). The container can be build for RDNA2 (`gfx1030`) and older GPUs, too. Please refer to [AMD's system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/develop/reference/system-requirements.html) for a list of officially supported cards. ROCm is known to work on more consumer GPUs.

The container file creates a [toolbox](https://github.com/containers/toolbox) container for [`toolbox(1)`](https://www.mankier.com/1/toolbox) command line tool. A toolbox containers has seamless access to the entire system including user's home directory, networking, hardware, SSH agent, and more.

The container has all Python dependencies installed in a virtual env. The virtual env is already activated when you enter the container. However the ilab `cli` is **not** installed. 

## Quick start

1. git clone the `cli` and `taxonomy` project into a common folder in your
   home directory (e.g. `~/path/to/instruct-lab`)
2. add your account to `render` and `video` group: `sudo usermod -a -G render,video $LOGNAME`
3. install build dependency for this container: `sudo dnf install toolbox podman make rocminfo`
4. build the container: `make rocm`
5. create a toolbox `make rocm-toolbox`
6. enter toolbox `toolbox enter instructlab`. The container has your
   home directory mounted.
7. install ilab cli with `pip install -e ~/path/to/instruct-lab/cli/`

`ilab generate` and `ilab chat` use the GPU automatically. `ilab train` needs
more powerful and recent GPU and therefore does not use GPU by default. To
train on a GPU, run `ilab train --device cuda`.


## Building for other GPU architectures

Use the `amdgpu-arch` or `rocminfo` tool to get the short name

```shell
dnf install clang-tools-extra rocminfo
amdgpu-arch
rocminfo | grep gfx
```

Map the name to a LLVM GPU target and an override GFX version. PyTorch 2.2.1+rocm5.7 provides a limited set of rocBLAS Kernels. Fedora 40's ROCm packages have more Kernels. For now we are limited to what PyTorch binaries provide until Fedora ships `python-torch` with ROCm support.

| Name      | xnack/USM | Version  | PyTorch | Fedora |
|-----------|-----------|----------|:-------:|:------:|
| `gfx900`  |           | `9.0.0`  | ✅      | ✅     |
| `gfx906`  | `xnack-`  | `9.0.6`  | ✅      | ✅     |
| `gfx908`  | `xnack-`  | `9.0.8`  | ✅      | ✅     |
| `gfx90a`  | `xnack-`  | `9.0.10` | ✅      | ✅     |
| `gfx90a`  | `xnack+`  | `9.0.10` | ✅      | ✅     |
| `gfx940`  |           |          | ❌      | ✅     |
| `gfx941`  |           |          | ❌      | ✅     |
| `gfx942`  |           |          | ❌      | ✅     |
| `gfx1010` |           |          | ❌      | ✅     |
| `gfx1012` |           |          | ❌      | ✅     |
| `gfx1030` |           | `10.3.0` | ✅      | ✅     |
| `gfx1100` |           | `11.0.0` | ✅      | ✅     |
| `gfx1101` |           |          | ❌      | ✅     |
| `gfx1102` |           |          | ❌      | ✅     |

If your card is not listed or unsupported, try the closest smaller value, e.g. for `gfx1031` use target `gfx1030` and override `10.3.0`. See [ROCm/ROCR-Runtime `isa.cpp`](https://github.com/ROCm/ROCR-Runtime/blob/rocm-6.0.2/src/core/runtime/isa.cpp#L245) and [LLVM User Guide for `AMDGPU`](https://llvm.org/docs/AMDGPUUsage.html#processors) for more information.

| Marketing Name         | Name      | Arch  | Target    | GFX version | Memory | Chat | Train |
|------------------------|-----------|-------|-----------|-------------|--------|:----:|:-----:|
| AMD Radeon RX 7900 XT  | `gfx1100` | RDNA3 | `gfx1100` | `11.0.0`    | 20 GiB | ✅   | ✅    |
| AMD Radeon RX 7900 XTX |           | RDNA3 |           |             | 24 GiB | ✅   | ✅    |
| AMD Radeon RX 6700     | `gfx1031` | RDNA2 | `gfx1030` | `10.3.0`    | 10 GiB | ✅   | ❌    |

Build the container with additional build arguments:

```shell
podman build \
    --build-arg AMDGPU_ARCH="gfx1030" \
    --build-arg HSA_OVERRIDE_GFX_VERSION="10.3.0" \
    -f container/rocm/Containerfile \
    -t localhost/instructlab:rocm-gf1030
```
