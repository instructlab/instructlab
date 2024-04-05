from lab import ilab
import click

@ilab.group(chain=True)
@click.version_option(package_name="cli")
@click.pass_context
# pylint: disable=redefined-outer-name
def model(ctx):
    """CLI for interacting with Models in InstructLab.

    If this is your first time running ilab, it's best to start with `ilab init` to create the environment.
    """
    pass
