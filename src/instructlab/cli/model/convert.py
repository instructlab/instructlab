# Standard
from pathlib import Path

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.defaults import DEFAULTS
from instructlab.model.convert import SUPPORTED_FORMATS, convert_model


@click.command()
@click.option(
    "--model",
    "-m",
    required=True,
    help="Path where source model is stored",
)
@click.option(
    "-tf",
    "--target-format",
    type=click.Choice([fmt.value for fmt in SUPPORTED_FORMATS], case_sensitive=False),
    help="Format of converted model",
    required=True,
)
@click.option(
    "--destination",
    type=click.Path(path_type=Path),
    default=Path(DEFAULTS.MODELS_DIR),
    help="Path where converted model should be stored",
)
@click.option(
    "--outtype",
    type=click.Choice(["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"]),
    default="f16",
)
@click.pass_context
@clickext.display_params
def convert(
    ctx,
    model: Path,
    destination: Path,
    outtype: str,
    target_format: str = SUPPORTED_FORMATS.GGUF.value,
):
    convert_model(
        Path(model),
        Path(destination),
        outtype=outtype,
        target_format=SUPPORTED_FORMATS(target_format),
    )
