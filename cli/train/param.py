# Standard
import typing

# Third Party
import click


class TorchDeviceInfo(typing.NamedTuple):
    type: str
    index: typing.Optional[int]
    device_map: typing.Dict[str, typing.Union[str, int]]


CPU_DEVICE = TorchDeviceInfo("cpu", None, {"": "cpu"})


class TorchDeviceParam(click.ParamType):
    """Parse and convert device string

    Returns DeviceInfo object:
    - type is one of 'cpu' or 'cuda')
    - index is None or CUDA/ROCm device index (0 for first GPU)
    - device_map is a dict
    """

    name = "deviceinfo"

    def convert(self, value, param, ctx):
        if isinstance(value, TorchDeviceInfo):
            return value

        if value == "cpu":
            # all layers on CPU
            return CPU_DEVICE

        if not value.startswith("cuda"):
            self.fail(
                "Only 'cpu', 'cuda', cuda with device index (e.g. 'cuda:0') "
                "are currently supported.",
                param,
                ctx,
            )

        # Function local import, import torch can take more than a second
        # Third Party
        import torch

        # Detect CUDA/ROCm device
        if not torch.cuda.is_available():
            self.fail(
                f"{value}: Torch has no CUDA/ROCm support or could not detect "
                "a compatible device.",
                param,
                ctx,
            )
        try:
            device = torch.device(value)
        except RuntimeError as e:
            self.fail(str(e), param, ctx)
        # map unqualified 'cuda' to current device
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        # all layers on a single GPU
        return TorchDeviceInfo(device.type, device.index, {"": device.index})


TORCH_DEVICE = TorchDeviceParam()
