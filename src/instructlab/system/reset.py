# SPDX-License-Identifier: Apache-2.0
import click
import shutil
import random

from instructlab import utils
from instructlab.configuration import ILAB_CONFIG_HOME, ILAB_DATA_HOME


def delete_storage_dirs():
    """
    Deletes all of the data in the instructlab storage & config directories.
    """
    click.echo(f"removing {ILAB_CONFIG_HOME}...")
    shutil.rmtree(ILAB_CONFIG_HOME, ignore_errors=False)
    print(f"removing {ILAB_DATA_HOME}...")
    shutil.rmtree(ILAB_DATA_HOME)


@click.command()
@click.option(
    "--skip-verify",
    default=False,
    help="Whether or not to skip verification of data deletion.",
)
@utils.display_params
def reset(skip_verify: bool):
    """Resets the current instructlab config & storage directories"""
    if not skip_verify:
        alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789")
        random.shuffle(alphabet)
        sequence = "".join(alphabet[:5])
        click.echo(
            click.style("WARNING:", bold=True)
            + " `ilab system reset` will purge all of your local `ilab` data. To proceed, please confirm by repeating the following string:"
        )
        click.echo("> " + click.style(f"{sequence}", bold=True) + "\n")
        user_entry: str = click.prompt("> ", type=click.STRING).upper()
        if user_entry != sequence:
            click.secho(
                f"Input sequence {user_entry} did not match {sequence}, aborting."
            )
            return
        print()  # add a newline

    # recursively removes the data at the config and share directories
    delete_storage_dirs()
