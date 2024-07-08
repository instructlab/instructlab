# SPDX-License-Identifier: Apache-2.0
# Standard
import importlib
import os
import platform
import sys
import typing

# Third Party
import click

# First Party
from instructlab import clickext


def _platform_info() -> typing.Dict[str, typing.Any]:
    """Platform and machine information"""
    d = {
        "sys.version": sys.version,
        "sys.platform": sys.platform,
        "os.name": os.name,
        "platform.release": platform.release(),
        "platform.machine": platform.machine(),
    }
    if sys.platform == "linux":
        os_release = platform.freedesktop_os_release()
        for key in ["ID", "VERSION_ID", "PRETTY_NAME"]:
            value = os_release.get(key)
            if value:
                d[f"os-release.{key}"] = value
    return d


def _torch_info() -> typing.Dict[str, typing.Any]:
    """Torch capabilities and devices"""
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
    # Third Party
    import torch

    if not torch.cuda.is_available():
        return {}

    d = {
        "torch.cuda.bf16": torch.cuda.is_bf16_supported(),
        "torch.cuda.current": torch.cuda.current_device(),
    }

    for idx in range(torch.cuda.device_count()):
        device = torch.device("cuda", idx)
        free, total = torch.cuda.mem_get_info(device)
        capmax, capmin = torch.cuda.get_device_capability(device)
        d[f"torch.cuda.{idx}.name"] = torch.cuda.get_device_name(device)
        d[f"torch.cuda.{idx}.free"] = f"{(free / 1024**3):.1f}"
        d[f"torch.cuda.{idx}.total"] = f"{(total / 1024**3):.1f}"
        d[f"torch.cuda.{idx}.capability"] = f"{capmax}.{capmin}"

    return d


def _torch_hpu_info() -> typing.Dict[str, typing.Any]:
    """Intel Gaudi (HPU) devices"""
    # Third Party
    import torch

    try:
        # Third Party
        from habana_frameworks.torch import hpu
    except ImportError:
        return {}

    d = {
        # 'habana_frameworks' has package name 'habana_torch_plugin'
        "habana_torch_plugin.version": importlib.metadata.version(
            "habana_torch_plugin"
        ),
        "torch.hpu.is_available": hpu.is_available(),
    }

    if not d["torch.hpu.is_available"]:
        return d

    d["torch.hpu.device_count"] = hpu.device_count()
    for idx in range(hpu.device_count()):
        device = torch.device("hpu", idx)
        d[f"torch.hpu.{idx}.name"] = hpu.get_device_name(device)
        d[f"torch.hpu.{idx}.capability"] = hpu.get_device_capability(device)
        prop: str = hpu.get_device_properties(device)
        d[f"torch.hpu.{idx}.properties"] = prop.strip("()")
    for key, value in sorted(os.environ.items()):
        if key.startswith(("PT_", "HABANA", "LOG_LEVEL_", "ENABLE_CONSOLE")):
            d[f"env.{key}"] = value
    return d


def _llama_cpp_info() -> typing.Dict[str, typing.Any]:
    """llama-cpp-python capabilities"""
    # Third Party
    import llama_cpp

    return {
        "llama_cpp_python.version": importlib.metadata.version("llama_cpp_python"),
        "llama_cpp_python.supports_gpu_offload": llama_cpp.llama_supports_gpu_offload(),
    }


def _instructlab_info():
    """InstructLab packages"""
    # auto-detect all instructlab packages
    pkgs = sorted(
        (dist.name, dist.version)
        for dist in importlib.metadata.distributions()
        if dist.name.startswith("instructlab")
    )
    return {f"{name}.version": ver for name, ver in pkgs}


def get_sysinfo() -> typing.Dict[str, typing.Any]:
    """Get system information"""
    d = {}
    d.update(_platform_info())
    d.update(_instructlab_info())
    d.update(_torch_info())
    d.update(_torch_cuda_info())
    d.update(_torch_hpu_info())
    d.update(_llama_cpp_info())
    return d


@click.command()
@clickext.display_params
def info():
    """Print system information"""
    for key, value in get_sysinfo().items():
        print(f"{key}: {value}")
