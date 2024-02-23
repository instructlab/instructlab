import click
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings
import uvicorn

from click_didyoumean import DYMGroup
from .generator.generate_data import generate_data
from .chat.chat import chat_cli


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
@click.argument(
    "question", nargs=-1, type=click.UNPROCESSED
)
@click.option(
    "-m", "--model", "model", help="Model to use"
)
@click.option(
    "-c", "--context", "context", help="Name of system context in config file", default="default"
)
@click.option(
    "-s", "--session", "session", help="Filepath of a dialog session file", type=click.File("r")
)
@click.option(
    "-qq", "--quick-question", "qq", help="Exist after answering question", is_flag=True
)
def chat(question, model, context, session, qq):
    """Run a chat using the modified model"""
    chat_cli(question, model, context, session, qq)


@cli.command()
def download():
    """Download the model(s) to train"""
    click.echo("# download TBD")
