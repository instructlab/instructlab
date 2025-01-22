# Intel Gaudi / Habana Labs HPU with SynapseAI

> **WARNING** Intel Gaudi support is currently under development and not ready for production.
>
> **NOTE** These instructions install `llama-cpp-python` for CPU. Inference in `ilab model chat`, `ilab model serve`, and `ilab data generate` is not using hardware acceleration.

## System requirements

- RHEL 9 on `x86_64` (tested with RHEL 9.4)
- Intel Gaudi 2 device
- [Habana Labs](https://docs.habana.ai/en/latest/index.html) software stack (tested with 1.18.0)
- software from Habana Vault for [RHEL](https://vault.habana.ai/ui/native/rhel) and [PyTorch](https://vault.habana.ai/ui/native/gaudi-pt-modules)

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
baseurl=https://vault.habana.ai/artifactory/rhel/9/9.4
enabled=1
gpgcheck=1
repo_gpgcheck=0
gpgkey=https://vault.habana.ai/artifactory/api/v2/repositories/rhel/keyPairs/primary/public
```

3. Install firmware and tools

```shell
dnf install habanalabs-firmware habanalabs-firmware-odm habanalabs-firmware-tools habanalabs-rdma-core habanalabs-graph habanalabs-thunk
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
habanalabs_en: loading driver, version: 1.18.0-95323a5
habanalabs_ib: loading driver, version: 1.18.0-95323a5
habanalabs_cn: loading driver, version: 1.18.0-95323a5
habanalabs: loading driver, version: 1.18.0-ee698fb
habanalabs 0000:21:00.0: habanalabs device found [1da3:1060] (rev 1)
...
habanalabs 0000:21:00.0 hbl_0: IB device registered
accel accel0: Successfully added device 0000:21:00.0 to habanalabs driver
```

7. Check `hl-smi`

````shell
hl-smi
+-----------------------------------------------------------------------------+
| HL-SMI Version:                              hl-1.18.0-fw-53.1.1.1          |
| Driver Version:                                     1.18.0-ee698fb          |
|-------------------------------+----------------------+----------------------+
| AIP  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | AIP-Util  Compute M. |
|===============================+======================+======================|
|   0  HL-325              N/A  | 0000:21:00.0     N/A |                   0  |
| N/A   43C   N/A   264W / 900W |    672MiB / 131072MiB |     0%           N/A|
|-------------------------------+----------------------+----------------------+
````

See [Intel Gaudi SW Stack for RHEL 9.4](https://docs.habana.ai/en/latest/shared/Install_Driver_and_Firmware.html)
for detailed documentation.

## Other tools

The Habana Vault repository provides several other tools, e.g. OCI container runtime hooks.

```shell
dnf install habanalabs-container-runtime
```

## Install Python, Intel oneMKL, and PyTorch stack

Retrieve installer script

```shell
curl -O https://vault.habana.ai/artifactory/gaudi-installer/1.18.0/habanalabs-installer.sh
chmod +x habanalabs-installer.sh
```

Install dependencies (use `--verbose` for verbose logging). This will install several RPM packages, download Intel compilers + libraries, and more.

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
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=7516192768
export LD_PRELOAD=/lib64/libtcmalloc.so

# work around race condition on systems with lots of cores
export OMP_NUM_THREADS=16

# Gaudi configuration
export PT_HPU_LAZY_MODE=0

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
 PT_HPU_EAGER_PIPELINE_ENABLE = 1
 PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE = 1
---------------------------: System Configuration :---------------------------
Num CPU Cores : 224
CPU RAM       : 1055745600 KB
------------------------------------------------------------------------------
Device count: 8
  hpu:0 is 'GAUDI3', cap: 1.18.0.1b7f293 (sramBaseAddress=144396662951903232, dramBaseAddress=144396800491520000, sramSize=0, dramSize=136465870848, tpcEnabledMask=18446744073709551615, dramEnabled=1, fd=17, device_id=0, device_type=5)
PT and Habana Environment variables
  HABANALABS_HLTHUNK_TESTS_BIN_PATH="/opt/habanalabs/src/hl-thunk/tests/arc"
  HABANA_LOGS="/var/log/habana_logs/"
  HABANA_PLUGINS_LIB_PATH="/usr/lib/habanatools/habana_plugins"
  HABANA_PROFILE="profile_api_light"
  HABANA_SCAL_BIN_PATH="/opt/habanalabs/engines_fw"
  PT_HPU_LAZY_MODE="0"
```

## Container

```shell
sudo dnf install podman
sudo setsebool -P container_use_devices=true
make hpu
podman run -ti --device=/dev/accel/ --device=/dev/infiniband/ -v ./data:/opt/app-root/src:z localhost/instructlab:hpu
```

## Known issues and limitations

- Training is limited to a single device, no DistributedDataParallel, yet.
- On systems with lots of CPU cores, training sometimes crashes with a segfault right after "loading the base model". The back trace suggests a race condition in `libgomp` or oneMKL. Use the environment variable `OMP_NUM_THREADS` to reduce `OMP`'s threads, e.g. `OMP_NUM_THREADS=1`.
- `habana-container-hook` can cause `podman build` to fail.
- Training parameters are not optimized and verified for best results.
- `llama-cpp` has no hardware acceleration backend for HPUs. Inference (`ilab data generate` and `ilab model chat`) is slow and CPU bound.
