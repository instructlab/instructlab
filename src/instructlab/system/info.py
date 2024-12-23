# SPDX-License-Identifier: Apache-2.0
# Standard
import importlib
import json
import logging
import os
import platform
import subprocess
import sys
import typing

# Third Party
import psutil

logger = logging.getLogger(__name__)


def get_system_info():
    """Returns system information as a JSON string"""
    categories = get_sysinfo_by_category()
    return json.dumps(categories)


def _platform_info() -> typing.Dict[str, typing.Any]:
    """Platform and machine information"""
    d = {
        "sys.version": sys.version,
        "sys.platform": sys.platform,
        "os.name": os.name,
        "platform.release": platform.release(),
        "platform.machine": platform.machine(),
        "platform.node": platform.node(),
        "platform.python_version": platform.python_version(),
    }

    if sys.platform == "linux":
        os_release = platform.freedesktop_os_release()
        for key in ["ID", "VERSION_ID", "PRETTY_NAME", "VARIANT"]:
            value = os_release.get(key)
            if value:
                d[f"os-release.{key}"] = value
        variant_id = os_release.get("VARIANT_ID")
        if variant_id:
            product_key = f"{variant_id.upper()}_VERSION_ID"
            product_version_id = os_release.get(product_key)
            if product_version_id:
                d[f"os-release.{product_key}"] = product_version_id
    elif sys.platform == "darwin":
        try:
            cpu_info = (
                subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
                .decode("utf-8")
                .strip()
            )
            d["platform.cpu_brand"] = cpu_info
        except subprocess.CalledProcessError:
            d["platform.cpu_brand"] = "Unknown"

    memory_info = psutil.virtual_memory()
    d["memory.total"] = f"{(memory_info.total / 1024**3):.2f} GB"
    d["memory.available"] = f"{(memory_info.available / 1024**3):.2f} GB"
    d["memory.used"] = f"{(memory_info.used / 1024**3):.2f} GB"
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
        "torch.cuda.current.device": torch.cuda.current_device(),
    }

    for idx in range(torch.cuda.device_count()):
        device = torch.device("cuda", idx)
        free, total = torch.cuda.mem_get_info(device)
        capmax, capmin = torch.cuda.get_device_capability(device)
        d[f"torch.cuda.{idx}.name"] = torch.cuda.get_device_name(device)
        d[f"torch.cuda.{idx}.free"] = f"{(free / 1024**3):.1f} GB"
        d[f"torch.cuda.{idx}.total"] = f"{(total / 1024**3):.1f} GB"
        # NVIDIA GPU compute capability table: https://developer.nvidia.com/cuda-gpus#compute
        d[f"torch.cuda.{idx}.capability"] = (
            f"{capmax}.{capmin} (see https://developer.nvidia.com/cuda-gpus#compute)"
        )

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


def add_to_category(
    categories: dict[str, list[tuple[str, typing.Any]]],
    display_name: str,
    data: dict,
):
    """Add data to the correct category with the specified display name."""
    if display_name not in categories:
        categories[display_name] = []
    categories[display_name].extend(data.items())


def get_sysinfo_by_category() -> typing.Dict[str, list[tuple[str, typing.Any]]]:
    """Get system information and categorize it directly"""
    categories: dict[str, list[tuple[str, typing.Any]]] = {}
    add_to_category(categories, "Platform", _platform_info())
    add_to_category(categories, "InstructLab", _instructlab_info())
    add_to_category(categories, "Torch", _torch_info())
    add_to_category(categories, "Torch", _torch_cuda_info())
    add_to_category(categories, "Torch", _torch_hpu_info())
    add_to_category(categories, "llama_cpp_python", _llama_cpp_info())
    return categories
