# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path

# Third Party
import click

# First Party
from instructlab import configuration as config
from instructlab import signing, utils


@click.command()
@click.option(
    "--model-path",
    type=click.Path(),
    default=config.DEFAULT_MODEL_PATH,
    show_default=True,
    help="Path to the model to be signed.",
)
@click.option(
    "--bundle-path",
    type=click.Path(),
    default=None,
    show_default=True,
    help="Path to save the Sigstore bundle file after signing.",
)
@click.option(
    "--staging",
    is_flag=True,
    help="Use Sigstore's staging environment.",
)
@click.pass_context
@utils.display_params
def sign(ctx, model_path, bundle_path, staging):
    """Signs a model with Sigstore"""

    if bundle_path is None:
        bundle_path = f"{model_path}.sigstore.json"

    signing.sign_model(
        model_path=Path(model_path), bundle_path=Path(bundle_path), staging=staging
    )
