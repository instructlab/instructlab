# SPDX-License-Identifier: Apache-2.0
# Standard
import importlib.metadata
import os
import platform
import sys
import typing


def _platform_info() -> typing.Dict[str, typing.Any]:
    """Platform and machine information"""
    info = {
        "sys.version": sys.version,
        "sys.platform": sys.platform,
        "os.name": os.name,
        "platform.release": platform.release(),
        "platform.machine": platform.machine(),
    }
    if sys.version_info >= (3, 10) and sys.platform == "linux":
        os_release = platform.freedesktop_os_release()
        for key in ["ID", "VERSION_ID", "PRETTY_NAME"]:
            value = os_release.get(key)
            if value:
                info[f"os-release.{key}"] = value
    return info


def _torch_info() -> typing.Dict[str, typing.Any]:
    """Torch capabilities and devices"""
    # pylint: disable=import-outside-toplevel
    # Third Party
    import torch

    return {
        "torch.version": torch.__version__,
        "torch.backends.cpu.capability": torch.backends.cpu.get_cpu_capability(),
        "torch.version.cuda": torch.version.cuda,
        "torch.version.hip": torch.version.hip,
        "torch.cuda.available": torch.cuda.is_available(),
        "torch.backends.cuda.is_built": torch.backends.cuda.is_built(),
        "torch.backends.mps.is_built": torch.backends.mps.is_built(),
        "torch.backends.mps.is_available": torch.backends.mps.is_available(),
    }


def _torch_cuda_info() -> typing.Dict[str, typing.Any]:
    """Torch Nvidia CUDA / AMD ROCm devices"""
    # pylint: disable=import-outside-toplevel
    # Third Party
    import torch

    if not torch.cuda.is_available():
        return {}

    info = {
        "torch.cuda.bf16": torch.cuda.is_bf16_supported(),
        "torch.cuda.current": torch.cuda.current_device(),
    }

    for idx in range(torch.cuda.device_count()):
        device = torch.device("cuda", idx)
        free, total = torch.cuda.mem_get_info(device)
        capmax, capmin = torch.cuda.get_device_capability(device)
        info[f"torch.cuda.{idx}.name"] = torch.cuda.get_device_name(device)
        info[f"torch.cuda.{idx}.free"] = f"{(free / 1024**3):.1f}"
        info[f"torch.cuda.{idx}.total"] = f"{(total / 1024**3):.1f}"
        info[f"torch.cuda.{idx}.capability"] = f"{capmax}.{capmin}"

    return info


def _torch_hpu_info() -> typing.Dict[str, typing.Any]:
    """Intel Gaudi (HPU) devices"""
    # pylint: disable=import-outside-toplevel
    # Third Party
    import torch

    try:
        # Third Party
        from habana_frameworks.torch import hpu
    except ImportError:
        return {}

    info = {
        # 'habana_frameworks' has package name 'habana_torch_plugin'
        "habana_torch_plugin.version": importlib.metadata.version(
            "habana_torch_plugin"
        ),
        "torch.hpu.is_available": hpu.is_available(),
    }

    if not info["torch.hpu.is_available"]:
        return info

    info["torch.hpu.device_count"] = hpu.device_count()
    for idx in range(hpu.device_count()):
        device = torch.device("hpu", idx)
        info[f"torch.hpu.{idx}.name"] = hpu.get_device_name(device)
        info[f"torch.hpu.{idx}.capability"] = hpu.get_device_capability(device)
        prop: str = hpu.get_device_properties(device)
        info[f"torch.hpu.{idx}.properties"] = prop.strip("()")
    for key, value in sorted(os.environ.items()):
        if key.startswith(("PT_", "HABANA", "LOG_LEVEL_", "ENABLE_CONSOLE")):
            info[f"env.{key}"] = value
    return info


def _llama_cpp_info() -> typing.Dict[str, typing.Any]:
    """llama-cpp-python capabilities"""
    # pylint: disable=import-outside-toplevel
    # Third Party
    import llama_cpp

    return {
        "llama_cpp_python.version": importlib.metadata.version("llama_cpp_python"),
        "llama_cpp_python.supports_gpu_offload": llama_cpp.llama_supports_gpu_offload(),
    }


def get_sysinfo() -> typing.Dict[str, typing.Any]:
    """Get system information"""
    info = {
        "instructlab.version": importlib.metadata.version("instructlab"),
    }
    info.update(_platform_info())
    info.update(_torch_info())
    info.update(_torch_cuda_info())
    info.update(_torch_hpu_info())
    info.update(_llama_cpp_info())
    return info


def main():
    for key, value in get_sysinfo().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
