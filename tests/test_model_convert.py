# Standard
from pathlib import Path
from unittest.mock import patch

# First Party
from instructlab.model.convert import (
    SUPPORTED_FORMATS,
    Converter,
    convert_model,
    get_model_format,
)


def test_get_model_format_safetensors():
    with (
        patch(
            "instructlab.model.convert.is_model_safetensors"
        ) as mock_is_model_safetensors,
        patch("instructlab.model.convert.is_model_gguf") as mock_is_model_gguf,
    ):
        mock_is_model_safetensors.return_value = True
        mock_is_model_gguf.return_value = False
        model_path = Path("model.safetensors")
        assert get_model_format(model_path) == SUPPORTED_FORMATS.SAFETENSORS


def test_get_model_format_gguf():
    with (
        patch(
            "instructlab.model.convert.is_model_safetensors"
        ) as mock_is_model_safetensors,
        patch("instructlab.model.convert.is_model_gguf") as mock_is_model_gguf,
    ):
        mock_is_model_safetensors.return_value = False
        mock_is_model_gguf.return_value = True
        model_path = Path("model.gguf")
        assert get_model_format(model_path) == SUPPORTED_FORMATS.GGUF


def test_register_format_converter():
    class DummyConverter(Converter):
        def convert(self):
            pass

    Converter.register_format_converter(
        SUPPORTED_FORMATS.SAFETENSORS, SUPPORTED_FORMATS.GGUF
    )(DummyConverter)
    assert (
        SUPPORTED_FORMATS.SAFETENSORS,
        SUPPORTED_FORMATS.GGUF,
    ) in Converter.converter_dict
    assert (
        Converter.converter_dict[
            (SUPPORTED_FORMATS.SAFETENSORS, SUPPORTED_FORMATS.GGUF)
        ]
        == DummyConverter
    )


def test_get_fmt_converter():
    class DummyConverter(Converter):
        def convert(self):
            pass

    Converter.register_format_converter(
        SUPPORTED_FORMATS.SAFETENSORS, SUPPORTED_FORMATS.GGUF
    )(DummyConverter)
    converter_cls = Converter.get_fmt_converter(
        SUPPORTED_FORMATS.SAFETENSORS, SUPPORTED_FORMATS.GGUF
    )
    assert converter_cls == DummyConverter


def test_convert_model():
    with (
        patch("instructlab.model.convert.get_model_format") as mock_get_model_format,
        patch(
            "instructlab.llamacpp.convert_to_gguf.convert_model_to_gguf"
        ) as mock_convert_model_to_gguf,
    ):
        mock_get_model_format.return_value = SUPPORTED_FORMATS.SAFETENSORS

        model_path = Path("model.safetensors")
        destination_path = Path("model.gguf")
        outtype = "f16"
        target_format = SUPPORTED_FORMATS.GGUF

        # Ensure the converter is registered
        @Converter.register_format_converter(
            SUPPORTED_FORMATS.SAFETENSORS, SUPPORTED_FORMATS.GGUF
        )
        class SafetensorsToGGUF(Converter):  # pylint: disable=unused-variable
            def convert(self):
                super().convert()
                mock_convert_model_to_gguf(
                    model=self.source_model,
                    outfile=self.destination,
                    outtype=self.outtype,
                )

        convert_model(model_path, destination_path, outtype, target_format)
        mock_convert_model_to_gguf.assert_called_once_with(
            model=model_path, outfile=destination_path, outtype=outtype
        )
