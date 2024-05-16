# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path

# Third Party
from sigstore.errors import VerificationError
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
    "--identity",
    help="Certificate identity to verify against."
    "Typically an email provided by the OIDC identity provider."
    "Example: a model signed using a Github account under hello@instructlab.ai would use hello@instructlab.ai.",
)
@click.option(
    "--issuer",
    help="Certificate identity's issuing authority.",
    default="https://github.com/login/oauth",
    show_default=True,
)
@click.option(
    "--staging",
    is_flag=True,
    help="Use Sigstore's staging environment.",
)
@click.pass_context
@utils.display_params
def verify(ctx, model_path, bundle_path, identity, issuer, staging):
    """Signs a model with Sigstore"""

    if bundle_path is None:
        bundle_path = f"{model_path}.sigstore.json"

    try:
        signing.verify_model(
            model_path=Path(model_path),
            bundle_path=Path(bundle_path),
            identity=identity,
            issuer=issuer,
            staging=staging,
        )
        print(f"✅ {bundle_path} passed verification")
    except VerificationError as e:
        print(f"❌ {bundle_path} failed verification: {str(e)}")
