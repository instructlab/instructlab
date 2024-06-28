import os
import click
import shutil

from instructlab import utils
from instructlab.configuration import ILAB_CONFIG_HOME, ILAB_DATA_HOME

@click.command()
@utils.display_params
def reset():
    """Resets the current instructlab config & storage directories"""

    # recursively removes the data at the config and share directories
    print(f"removing {ILAB_CONFIG_HOME}...")
    shutil.rmtree(ILAB_CONFIG_HOME, ignore_errors=False)
    print(f"removing {ILAB_DATA_HOME}...")
    shutil.rmtree(ILAB_DATA_HOME)
    print("done ✅")
