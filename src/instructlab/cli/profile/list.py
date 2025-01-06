# Standard
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
    profiles_dir = os.path.join(os.path.dirname(__file__), "../../profiles")
    click.echo("List available profiles:")

    profile_info = list_profiles(profiles_dir)

    formatted_result = format_output(profile_info)

    for line in formatted_result:
        click.echo(line)
