import click
from click_didyoumean import DYMGroup
import llama_cpp.llama_chat_format as llama_chat_format
import llama_cpp.server.app as llama_app
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings
import uvicorn
import logging

from .generator.generate_data import generate_data
from .download_model import download_model
from .chat.chat import chat_cli
from .config import Config


class Lab(object):
    """Lab object holds high-level information about lab CLI"""

    def __init__(self):
        self.config = Config()
        # TODO: change the default loglevel to whatever user specified
        FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s"
        logging.basicConfig(format=FORMAT, level=logging.DEBUG)
        self.logger = logging.getLogger()


@click.group(cls=DYMGroup)
@click.pass_context
def cli(ctx):
    """CLI for interacting with labrador"""
    ctx.obj = Lab()


@cli.command()
@click.pass_context
def init(ctx):
    """Initializes environment for labrador"""
    click.echo("please do\n")
    click.echo("git clone git@github.com:open-labrador/taxonomy.git\n")
    click.echo("to get the taxonomy repo")


@cli.command()
@click.option("--taxonomy", default="taxonomy", show_default=True, type=click.Path())
@click.pass_context
def list(ctx, taxonomy):
    """List taxonomy YAML files"""
    from os import system
    system(f"find {taxonomy} -iname '*.yaml'")


@cli.command()
@click.pass_context
def submit(ctx):
    """Initializes environment for labrador"""
    click.echo("please use git commands and GitHub to submit a PR to the taxonomy repo")


@cli.command()
@click.option("--model", default="./models/ggml-malachite-7b-Q4_K_M.gguf", show_default=True)
@click.option("--n_gpu_layers", default=-1, show_default=True)
@click.pass_context
def serve(ctx, model, n_gpu_layers):
    """Start a local server"""
    settings = Settings(model=model, n_ctx=4096, n_gpu_layers=n_gpu_layers)
    app = create_app(settings=settings)
    llama_app._llama_proxy._current_model.chat_handler = llama_chat_format.Jinja2ChatFormatter(
        template="{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n' + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}", eos_token="<|endoftext|>", bos_token=""
    ).to_chat_handler()
    click.echo("Starting server process")
    click.echo("After application startup complete see http://127.0.0.1:8000/docs for API.")
    click.echo("Press CTRL+C to shutdown server.")
    uvicorn.run(app, port=8000)  # TODO: host params, etc...


@cli.command()
@click.option("--model", default="ggml-malachite-7b-Q4_K_M", show_default=True)
@click.option("--num_cpus", default=10, show_default=True)
@click.option("--taxonomy", default="taxonomy", show_default=True, type=click.Path())
@click.option("--seed_file", default="./cli/generator/seed_tasks.jsonl", show_default=True, type=click.Path())
@click.pass_context
def generate(ctx, model, num_cpus, taxonomy, seed_file):
    """Generates synthetic data to enhance your example data"""
    generate_data(model_name=model, num_cpus=num_cpus, taxonomy=taxonomy, seed_tasks_path=seed_file)


@cli.command()
@click.pass_context
def train(ctx):
    """Trains labrador model"""
    click.echo("# train TBD")


@cli.command()
@click.pass_context
def test(ctx):
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
    "-qq", "--quick-question", "qq", help="Exit after answering question", is_flag=True
)
@click.pass_context
def chat(ctx, question, model, context, session, qq):
    """Run a chat using the modified model"""
    chat_cli(question, model, context, session, qq)


@cli.command()
@click.option(
    "--repo",
    default="https://github.com/open-labrador/cli.git",
    show_default=True,
    help="Github repository of the hosted models."
)
@click.option(
    "--release",
    default="latest",
    show_default=True,
    help="Github release version of the hosted models."
)
@click.option(
    "--dir",
    default=".",
    show_default=True,
    help="The local directory to download the model files into."
)
@click.option(
    "--pattern",
    default="",
    show_default=True,
    help="Download only assets that match a glob pattern."
)
@click.pass_context
def download(ctx, repo, release, dir, pattern):
    """Download the model(s) to train"""
    download_model(repo, release, dir, pattern)
