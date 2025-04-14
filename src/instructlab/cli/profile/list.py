# Standard
from pathlib import Path
import os

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.profile.list import format_output, list_profiles


@click.command()
@clickext.display_params
def profile_list():
    """List available profiles"""

    project_dir = Path(os.path.dirname(__file__)).parent.parent
    profiles_dir = os.path.join(project_dir, "profiles")
    click.echo("List of available profiles:")

    profile_info = list_profiles(profiles_dir)

    formatted_result = format_output(profile_info)

    for line in formatted_result:
        click.echo(line)
