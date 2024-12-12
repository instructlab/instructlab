# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from enum import Enum
from pathlib import Path
from typing import Callable, TypeVar
import logging
import os

# First Party
from instructlab.utils import is_model_gguf, is_model_safetensors

logger = logging.getLogger(__name__)


class SUPPORTED_FORMATS(Enum):
    GGUF = "gguf"
    SAFETENSORS = "safetensors"

    def __contains__(self, item):
        return item in [fmt.value for fmt in self.__class__]


AnyConverter = TypeVar("AnyConverter", bound="type[Converter]")


class Converter:
    source_model: Path
    destination: Path
    outtype: str

    source_format: SUPPORTED_FORMATS
    target_format: SUPPORTED_FORMATS
    converter_dict: dict[
        tuple[SUPPORTED_FORMATS, SUPPORTED_FORMATS], type[Converter]
    ] = {}

    def __init__(self, model: Path, destination: Path, outtype: str):
        """
        Initialize the Converter with the source model, destination, and output type.

        Args:
            model (Path): Path to the source model.
            destination (Path): Path to the destination.
            outtype (str): Output type.
        """
        self.source_model = model
        self.destination = destination
        self.outtype = outtype

    @classmethod
    def register_format_converter(
        cls, src_fmt: SUPPORTED_FORMATS, tgt_fmt: SUPPORTED_FORMATS
    ) -> Callable[[AnyConverter], AnyConverter]:
        """
        Register a format converter for the given source and target formats.

        Args:
            src_fmt (SUPPORTED_FORMATS): Source format.
            tgt_fmt (SUPPORTED_FORMATS): Target format.

        Returns:
            Callable[[AnyConverter], AnyConverter]: Callable that registers the converter class.
        """

        def func(convertercls: AnyConverter) -> AnyConverter:
            cls.converter_dict[(src_fmt, tgt_fmt)] = convertercls
            return convertercls

        return func

    @classmethod
    def get_fmt_converter(
        cls, src_fmt: SUPPORTED_FORMATS, tgt_fmt: SUPPORTED_FORMATS
    ) -> type[Converter]:
        """
        Get the format converter for the given source and target formats.

        Args:
            src_fmt (SUPPORTED_FORMATS): Source format.
            tgt_fmt (SUPPORTED_FORMATS): Target format.

        Returns:
            type[Converter]: Converter class.

        Raises:
            ValueError: If the conversion is not supported.
        """
        try:
            return cls.converter_dict[(src_fmt, tgt_fmt)]
        except KeyError:
            raise ValueError(
                f"Conversion from {src_fmt.value} to {tgt_fmt.value} is not supported"
            ) from None

    def convert(self):
        """
        Convert the model. This method should be implemented by subclasses.
        """


# Register new converters
# To add a new converter, create a new class that inherits from Converter and
# decorate it with the register_format_converter decorator. The decorator takes
# the source and target formats as arguments. The converter class should implement
# the convert method which will contain the logic to convert the model from the


@Converter.register_format_converter(
    SUPPORTED_FORMATS.SAFETENSORS, SUPPORTED_FORMATS.GGUF
)
class SafetensorsToGGUF(Converter):
    def convert(self):
        """
        Convert the model from Safetensors to GGUF format.
        """
        super().convert()
        # First Party
        from instructlab.llamacpp.convert_to_gguf import convert_model_to_gguf

        default_gguf_filename = f"{os.path.basename(self.source_model)}.gguf"
        dest = (
            self.destination
            if str(self.destination).endswith(".gguf")
            else self.destination / default_gguf_filename
        )

        convert_model_to_gguf(
            model=self.source_model,
            outfile=dest,
            outtype=self.outtype,
        )


def convert_model(
    model: Path, destination: Path, outtype: str, target_format: SUPPORTED_FORMATS
):
    """
    Convert the model to the specified target format.

    Args:
        model (Path): Path to the source model.
        destination (Path): Path to the destination.
        outtype (str): Output type.
        target_format (SUPPORTED_FORMATS): Target format.

    Raises:
        ValueError: If the conversion fails.
    """
    try:
        src_model_fmt = get_model_format(model)
        converter_cls = Converter.get_fmt_converter(src_model_fmt, target_format)
        converter_instance = converter_cls(model, destination, outtype)
        converter_instance.convert()
        logger.info(
            f"ᕦ(òᴗóˇ)ᕤ Model convert completed successfully! ᕦ(òᴗóˇ)ᕤ\nConverted model saved to {destination}"
        )
    except ValueError as e:
        raise ValueError(f"Model conversion failed: {e}") from e


def get_model_format(model: Path) -> SUPPORTED_FORMATS:
    """
    Get the format of the model.

    Args:
        model (Path): Path to the model.

    Returns:
        SUPPORTED_FORMATS: Model format.

    Raises:
        ValueError: If the model format is unsupported.
    """

    if is_model_safetensors(model):
        return SUPPORTED_FORMATS.SAFETENSORS

    if is_model_gguf(model):
        return SUPPORTED_FORMATS.GGUF

    raise ValueError(
        f"Source model {model} is in an unsupported format. Supported formats are {SUPPORTED_FORMATS._member_names_}"
    ) from None
