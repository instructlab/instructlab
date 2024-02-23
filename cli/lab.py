import click

from click_didyoumean import DYMGroup
from .generator.generate_data import generate_data


@click.group(cls=DYMGroup)
def cli():
    """CLI for interacting with labrador"""
    pass


@cli.command()
def init():
    """Initializes environment for labrador"""
    click.echo("# init TBD")


@cli.command()
def serve():
    """Start a local server"""
    click.echo("# serve TBD")


@cli.command()
@click.option("--model", default="ggml-labrador13B-model-Q4_K_M", show_default=True)
@click.option("--num_cpus", default=10, show_default=True)
def generate(model, num_cpus):
    """Generates synthetic data to enhance your example data"""
    generate_data(model_name=model, num_cpus=num_cpus)


@cli.command()
def train():
    """Trains labrador model"""
    click.echo("# train TBD")


@cli.command()
def test():
    """Perform rudimentary tests of the model"""
    click.echo("# test TBD")


@cli.command()
def chat():
    """Run a chat using the modified model"""
    click.echo("# chat TBD")
