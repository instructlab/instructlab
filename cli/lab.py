import click
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings
import uvicorn

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
@click.option("--model", default="/models/ggml-labrador13B-model-Q4_K_M.gguf", show_default=True)
@click.option("--n_gpu_layers", default=-1, show_default=True)
@click.option("--api_key", default="bogus", show_default=True)
def serve(model, n_gpu_layers):
    """Start a local server"""
    settings = Settings(model=model, n_gpu_layers=n_gpu_layers)
    app = create_app(settings=settings)
    click.echo("Starting server process")
    click.echo("After application startup complete see http://127.0.0.1:8000/docs for API.")
    click.echo("Press CTRL+C to shutdown server.")
    uvicorn.run(app)  # TODO: host params, etc...


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

@cli.command()
def download():
    """Download the model(s) to train"""
    click.echo("# download TBD")
