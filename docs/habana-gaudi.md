# Intel Gaudi / Habana Labs HPU with SynapseAI

> **WARNING** Intel Gaudi support is currently under development and not ready for production.
>
> **NOTE** These instructions install `llama-cpp-python` for CPU. Inference in `ilab model chat`, `ilab model serve`, and `ilab data generate` is not using hardware acceleration.

## System requirements

- RHEL 9 on `x86_64` (tested with RHEL 9.3 and patched installer)
- Intel Gaudi 2 device
- [Habana Labs](https://docs.habana.ai/en/latest/index.html) software stack (tested with 1.16.2)
- software from Habana Vault for [RHEL](https://vault.habana.ai/ui/native/rhel) and [PyTorch](https://vault.habana.ai/ui/native/gaudi-pt-modules)
- software [HabanaAI GitHub](https://github.com/HabanaAI/) org like [optimum-habana](https://github.com/HabanaAI/optimum-habana-fork) fork

## System preparation

### Kernel modules, firmware, firmware tools

1. Enable CRB and EPEL repositories

```shell
sudo subscription-manager repos --enable codeready-builder-for-rhel-9-$(arch)-rpms
sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm
```

2. Add Habana Vault repository `/etc/yum.repos.d/Habana-Vault.repo`

```ini
[vault]
name=Habana Vault
baseurl=https://vault.habana.ai/artifactory/rhel/9/9.2
enabled=1
repo_gpgcheck=0
```

3. Install firmware and tools

```shell
dnf install habanalabs-firmware habanalabs-firmware-tools
```

4. Install Kernel drivers. This will build and install several Kernel modules with DKMS

```shell
dnf install habanalabs
```

5. Load Kernel drivers

```shell
modprobe habanalabs_en habanalabs_cn habanalabs
```

6. Check journald for device

```shell
journalctl -o cat | grep habanalabs
habanalabs hl0: Loading secured firmware to device, may take some time...
habanalabs hl0: preboot full version: 'Preboot version hl-gaudi2-1.14.0-fw-48.0.1-sec-7 (Jan 07 2024 - 20:03:16)'
habanalabs hl0: boot-fit version 49.0.0-sec-9
habanalabs hl0: Successfully loaded firmware to device
habanalabs hl0: Linux version 49.0.0-sec-9
habanalabs hl0: Found GAUDI2 device with 96GB DRAM
habanalabs hl0: hwmon1: add sensors information
habanalabs hl0: Successfully added device 0000:19:00.0 to habanalabs driver
```

7. Check `hl-smi`

````shell
hl-smi
+-----------------------------------------------------------------------------+
| HL-SMI Version:                              hl-1.15.1-fw-49.0.0.0          |
| Driver Version:                                     1.15.1-62f612b          |
|-------------------------------+----------------------+----------------------+
| AIP  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | AIP-Util  Compute M. |
|===============================+======================+======================|
|   0  HL-225              N/A  | 0000:19:00.0     N/A |                   0  |
| N/A   29C   N/A    93W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
| Compute Processes:                                               AIP Memory |
|  AIP       PID   Type   Process name                             Usage      |
|=============================================================================|
|   0        N/A   N/A    N/A                                      N/A        |
+=============================================================================+
````

See [Intel Gaudi SW Stack for RHEL 9.2](https://docs.habana.ai/en/latest/shared/Install_Driver_and_Firmware.html)
for detailed documentation.

## Other tools

The Habana Vault repository provides several other tools, e.g. OCI container runtime hooks.

```shell
dnf install habanalabs-graph habanatool habanalabs-thunk habanalabs-container-runtime
```

## Install Python, Intel oneMKL, and PyTorch stack

Retrieve installer script

```shell
curl -O https://vault.habana.ai/artifactory/gaudi-installer/1.15.1/habanalabs-installer.sh
chmod +x habanalabs-installer.sh
```

> **NOTE**
>
> Habana Labs Installer 1.15.1 only supports RHEL 9.2 and will fail on 9.3+. You can hack around the limitation by patching the installer:
>
> ```shell
> sed -i 's/OS_VERSION=\$VERSION_ID/OS_VERSION=9.2/' habanalabs-installer.sh
> ```

Install dependencies (use `--verbose` for verbose logging). This will install several RPM packages, download Intel compilers + libraries, download + compile Python 3.10, and more.

```shell
export MAKEFLAGS="-j$(nproc)"
./habanalabs-installer.sh install --type dependencies --skip-install-firmware
```

Install PyTorch with Habana Labs framework in a virtual environment:

```shell
export HABANALABS_VIRTUAL_DIR=$HOME/habanalabs-venv
./habanalabs-installer.sh install --type pytorch --venv
```

Validate installation:

```shell
./habanalabs-installer.sh validate
```

## Habana Lab's PyTorch stack

Habana Labs comes with a modified fork of PyTorch that is build with Intel's oneAPI Math Kernel Library (oneMKL). The actual HPU bindings and helpers are provided by the `habana_framework` package. Imports of `habana_framework` sub-packages register `hpu` device support, `torch.hpu` module, and `dynamo` backends.

The [`SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer) from `trl` does not work with Habana stack. Instead the `GaudiSFTTrainer` from [optimum-habana](https://huggingface.co/docs/optimum/habana/index) is needed. The version on PyPI is currently broken, but the HabanaAI [optimum-habana-fork](https://github.com/HabanaAI/optimum-habana-fork) works.

## Install and run InstructLab with Intel Gaudi

Install `InstructLab` from checkout with additional dependencies:

```shell
. $HABANALABS_VIRTUAL_DIR/bin/activate
pip install ./instructlab[hpu]
```

> **TIP** If `llama-cpp-python` fails to build with error ``unsupported instruction `vpdpbusd'``, then install with `CFLAGS="-mno-avx" pip install ...`.

Train environment (see [Habana runtime environment variables](https://docs.habana.ai/en/latest/PyTorch/Reference/Runtime_Flags.html)

```shell
# environment variables for training
export TSAN_OPTIONS='ignore_noninstrumented_modules=1'
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=7516192768
export LD_PRELOAD=/lib64/libtcmalloc.so

# work around race condition on systems with lots of cores
export OMP_NUM_THREADS=16

# Gaudi configuration
export PT_HPU_LAZY_MODE=0
export PT_HPU_ENABLE_EAGER_CACHE=TRUE
export PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE=TRUE
export PT_ENABLE_INT64_SUPPORT=1

# additional environment variables for debugging
#export ENABLE_CONSOLE=true
#export LOG_LEVEL_ALL=5
#export LOG_LEVEL_PT_FALLBACK=1
```

Train on HPU

```shell
ilab model train --device=hpu
```

Output:

```shell-session
LINUX_TRAIN.PY: Using device 'hpu'
============================= HABANA PT BRIDGE CONFIGURATION ===========================
 PT_HPU_LAZY_MODE = 0
 PT_RECIPE_CACHE_PATH =
 PT_CACHE_FOLDER_DELETE = 0
 PT_HPU_RECIPE_CACHE_CONFIG =
 PT_HPU_MAX_COMPOUND_OP_SIZE = 9223372036854775807
 PT_HPU_LAZY_ACC_PAR_MODE = 1
 PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES = 0
---------------------------: System Configuration :---------------------------
Num CPU Cores : 48
CPU RAM       : 263943121 KB
------------------------------------------------------------------------------
Device count: 1
  hpu:0 is 'GAUDI2', cap: 1.15.1.b3dea3b61 (sramBaseAddress=1153202979533225984, dramBaseAddress=1153203082662772736, sramSize=50331648, dramSize=102106132480, tpcEnabledMask=16777215, dramEnabled=1, fd=21, device_id=0, device_type=4)
PT and Habana Environment variables
  HABANALABS_HLTHUNK_TESTS_BIN_PATH="/opt/habanalabs/src/hl-thunk/tests/arc"
  HABANA_LOGS="/var/log/habana_logs/"
  HABANA_PLUGINS_LIB_PATH="/usr/lib/habanatools/habana_plugins"
  HABANA_PROFILE="profile_api_light"
  HABANA_SCAL_BIN_PATH="/opt/habanalabs/engines_fw"
  PT_ENABLE_INT64_SUPPORT="1"
  PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE="TRUE"
  PT_HPU_ENABLE_EAGER_CACHE="TRUE"
  PT_HPU_LAZY_MODE="0"
```

## Container

```shell
dnf install habanalabs-container-runtime podman
make hpu
podman run -ti --privileged -v ./data:/opt/app-root/src:z localhost/instructlab:hpu
```

## Known issues and limitations

- Training is limited to a single device, no DistributedDataParallel, yet.
- On systems with lots of CPU cores, training sometimes crashes with a segfault right after "loading the base model". The back trace suggests a race condition in `libgomp` or oneMKL. Use the environment variable `OMP_NUM_THREADS` to reduce `OMP`'s threads, e.g. `OMP_NUM_THREADS=1`.
- `habana-container-hook` can cause `podman build` to fail.
- Training parameters are not optimized and verified for best results.
- `llama-cpp` has no hardware acceleration backend for HPUs. Inference (`ilab data generate` and `ilab model chat`) is slow and CPU bound.
- The container requires `--privileged`. A non-privileged container is missing `/dev/hl*` and other device files for HPUs.
