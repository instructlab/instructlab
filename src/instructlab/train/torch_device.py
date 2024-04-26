# SPDX-License-Identifier: Apache-2.0
"""Abstraction layer for Torch devices and device properties"""

# Standard
import abc
import importlib.metadata
import inspect
import logging
import os
import subprocess
import sys
import types
import typing

# Third Party
# import typing
import torch

logger = logging.getLogger(__name__)


HABANA_FRAMEWORK_VERSION: typing.Optional[str]
try:
    # habana_framework package has 'habana_torch_plugin' dist-info
    HABANA_FRAMEWORK_VERSION = importlib.metadata.version("habana_torch_plugin")
except ModuleNotFoundError:
    HABANA_FRAMEWORK_VERSION = None


def _gib(size: int) -> str:
    return "{:.1f} GiB".format(size / 1024**3)


def lookup_device(device: torch.device) -> "AbstractTorchDevice":
    """Lookup and create Torch Device plugin for a torch.device"""
    base = AbstractTorchDevice
    plugin_class: AbstractTorchDevice | None = base._plugin_registry.get(device.type)
    if plugin_class is None:
        if any(cls.type == device.type for cls in base._plugin_classes):
            # there is at least one plugin with same device type
            raise ValueError(f"Plugin for {device.type} is not available.")
        raise ValueError(f"Unknown or unsupported device type {device.type}")

    return plugin_class(device) # type: ignore


class AbstractTorchDevice(metaclass=abc.ABCMeta):
    """Abstract base for Torch Device"""

    # Human-readable device name
    name: str
    # Torch device type (cpu, cuda, ...)
    type: str
    # Version of the device module
    version: typing.Optional[str] = None

    # plugin registry
    _plugin_classes: typing.List["AbstractTorchDevice"] = []
    _plugin_registry: typing.Dict[str, "AbstractTorchDevice"] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            cls._plugin_classes.append(cls)
            # register only supported, CUDA and ROCm both have 'cuda' device type.
            if cls.is_supported():
                default = cls._plugin_registry.setdefault(cls.type, cls)
                # check for conflict
                if default is not cls:
                    raise ValueError(
                        f"Duplicate entry for '{cls.type}': {cls}, {default}"
                    )

    @classmethod
    @abc.abstractmethod
    def is_supported(cls) -> bool:
        """Check whether the torch device is supported

        A device is supported when PyTorch has been built with device support
        or the necessary extension packages are available. Supported does not
        imply that the device is available and working.
        """

    def __init__(self, device: torch.device, **kwargs):
        self.device: torch.device = device
        self.mod: types.ModuleType = self.load_module()
        self.options: typing.Dict[str, typing.Any] = kwargs

    def __bool__(self) -> bool:
        """True if device is supported and hardware is available"""
        return self.is_supported() and self.mod.is_available()

    def __repr__(self):
        return f"'TorchDevice {self.name} ({self.device})'"

    def load_module(self) -> types.ModuleType:
        """Load and get torch module for device

        Typically torch.{TYPE}, e.g. torch.cpu, torch.cuda. May import
        extra packages for externally supported devices such as Habana
        (Intel Gaudi).

        Note: PyTorch is inconsistent. Some devices have both torch.{TYPE} and
        torch.backend.{TYPE}.
        """
        mod = getattr(torch, self.device.type)
        if not isinstance(mod, types.ModuleType):
            raise TypeError(mod)
        return mod

    def init_device(self) -> None:
        """Initialize device hardware

        Allows device to initialize the hardware early and configure Torch.
        """
        return None

    def device_info_msg(self) -> typing.Iterable[str]:
        """Device information"""
        yield f"Torch device '{self.device}'"
        yield f"{self.name} (version: {self.version})"
        yield f"Supports bf16: {self.is_bf16_supported()}"
        yield f"Training keyword args: {self.training_kwargs}"
        yield f"dynamo backends: {', '.join(torch._dynamo.list_backends())}"
        yield f"Device count: {self.mod.device_count()} (current: {self.mod.current_device()})"

    @abc.abstractmethod
    def is_bf16_supported(self) -> bool:
        """Whether bfloat16 is supported by hardware and Torch"""

    @property
    def per_device_train_batch_size(self) -> int:
        """Batch size based on current hardware capabilities"""
        return 1

    @property
    def training_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Training arguments"""
        use_bf16 = self.is_bf16_supported()
        return {
            "bf16": use_bf16,
            "fp16": not use_bf16,
            "use_cpu": False,
            "per_device_train_batch_size": self.per_device_train_batch_size,
        }

    @property
    def torch_dtype(self) -> typing.Optional[str]:
        return None

    def autocast(self, **kwargs):
        """torch.autocast() wrapper"""
        return torch.autocast(self.device.type, **kwargs)

    def model_to_device(self, model: torch.nn.Module) -> torch.nn.Module:
        """Prepare model and move it to the device"""
        assert isinstance(model, torch.nn.Module)
        if model.device != self.device:
            model = model.to(self.device)
        return model


class CPUDevice(AbstractTorchDevice):
    """CPU device"""

    name = "CPU"
    type = "cpu"

    @classmethod
    def is_supported(cls) -> bool:
        return True

    def device_info_msg(self) -> typing.Iterable[str]:
        yield from super().device_info_msg()
        yield f"CPU capability: {torch.backends.cpu.get_cpu_capability()}"

    def is_bf16_supported(self) -> bool:
        # TODO: detect AVX512 support?
        return False

    @property
    def training_kwargs(self) -> typing.Dict[str, typing.Any]:
        kwargs = super().training_kwargs
        kwargs.update(
            # TODO: detect AVX2, F16C, AVX512 support?
            bf16=False,
            fp16=False,
            use_cpu=True,
            # TODO CPU test this possible optimization
            # use_ipex=True,
        )
        return kwargs


class AbstractCudaDevice(AbstractTorchDevice):
    """Shared base class for CUDA and ROCm"""

    type = "cuda"

    def device_info_msg(self) -> typing.Iterable[str]:
        yield from super().device_info_msg()
        for idx in range(self.mod.device_count()):
            device = torch.device(self.device.type, idx)
            name = self.mod.get_device_name(device)
            free, total = self.mod.mem_get_info(device)
            capmax, capmin = self.mod.get_device_capability(device)
            yield (
                f"  {device} is '{name}' ({_gib(free)} of {_gib(total)} free, "
                f"capability: {capmax}.{capmin})"
            )

    def is_bf16_supported(self) -> bool:
        # bfloat16 is not supported on older CUDA versions and devices
        # with CUDA support level < 8.0.
        return bool(self.mod.is_bf16_supported())

    @property
    def torch_dtype(self) -> typing.Optional[str]:
        return "auto"


class CUDADevice(AbstractCudaDevice):
    """NVIDIA CUDA device"""

    name = "NVIDIA CUDA"
    version = torch.version.cuda

    @classmethod
    def is_supported(cls) -> bool:
        return cls.version is not None


class ROCmDevice(AbstractCudaDevice):
    """AMD ROCm device

    ROCm devices are implemented as 'torch.cuda' and torch.device("cuda").
    """

    name = "AMD ROCm/HIP"
    version = torch.version.hip

    @classmethod
    def is_supported(cls) -> bool:
        return cls.version is not None


class GaudiDevice(AbstractTorchDevice):
    """Intel Gaudi / Habana Labs HPU"""

    name = "Intel Gaudi / Habana Labs"
    type = "hpu"
    version = HABANA_FRAMEWORK_VERSION

    _htcore: types.ModuleType
    _hpu_backends: types.ModuleType

    @classmethod
    def is_supported(cls) -> bool:
        return cls.version is not None

    def load_module(self) -> types.ModuleType:
        # these imports register 'torch.hpu', 'hpu' device, and 'hpu_backend'
        # for dynamo (torch.compile).
        # pylint: disable=import-error
        # Third Party
        from habana_frameworks.torch import core as htcore
        from habana_frameworks.torch import hpu
        from habana_frameworks.torch.dynamo.compile_backend import (
            backends as hpu_backends,
        )

        self._htcore = htcore
        self._hpu_backends = hpu_backends
        if typing.TYPE_CHECKING:
            assert isinstance(hpu, types.ModuleType)
        return hpu

    def init_device(self) -> None:
        self.mod.init()

    def device_info_msg(self) -> typing.Iterable[str]:
        yield from super().device_info_msg()
        for idx in range(self.mod.device_count()):
            device = torch.device(self.device.type, idx)
            name: str = self.mod.get_device_name(device)
            cap: str = self.mod.get_device_capability(device)
            # property string is surrounded by '()'
            prop: str = self.mod.get_device_properties(device)
            yield f"  {device} is '{name}', cap: {cap} {prop}"

        # https://docs.habana.ai/en/latest/PyTorch/Reference/Runtime_Flags.html
        yield "PT and Habana Environment variables"
        for key, value in sorted(os.environ.items()):
            if key.startswith(("PT_", "HABANA", "LOG_LEVEL_", "ENABLE_CONSOLE")):
                yield f'  {key}="{value}"'

    def is_bf16_supported(self) -> bool:
        return True

    @property
    def per_device_train_batch_size(self):
        return 8


class MPSDevice(AbstractTorchDevice):
    """Apple Silicon Metal Performance Shaders"""

    name = "MPS"
    type = "mps"

    _hv_vmm_present: typing.Optional[bool] = None

    @classmethod
    def hv_vmm_present(cls) -> typing.Optional[bool]:
        """Check for hardware virtualization

        MPS does not work in virtualization, e.g. GitHub Actions.
        """
        if sys.platform != "darwin":
            return None

        if cls._hv_vmm_present is None:
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", "kern.hv_vmm_present"],
                    text=True,
                )
            except subprocess.CalledProcessError:
                logger.exception("Failed to query sysctl")
                return None
            cls._hv_vmm_present = out.strip() == "1"

        return cls._hv_vmm_present

    @classmethod
    def is_supported(cls) -> bool:
        return torch.backends.mps.is_available() and not cls.hv_vmm_present

    def load_module(self) -> types.ModuleType:
        # torch.mps does not have init and is_available
        return torch.backends.mps

    def is_bf16_supported(self) -> bool:
        # TODO: verify
        return False


def main():
    logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    for subclass in AbstractTorchDevice._plugin_classes:
        print(f"Probing {subclass.__name__}")
        if not subclass.is_supported():
            print(f"{subclass.__name__} is not supported.")
        else:
            device = torch.device(subclass.type)
            try:
                td = lookup_device(device)
            except ValueError as e:
                print(e)
                continue
            else:
                if td:
                    td.init_device()
                    for line in td.device_info_msg():
                        print(line)
                else:
                    print(f"{device} ({td.name}) not available")
        print()


if __name__ == "__main__":
    main()
