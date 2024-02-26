import click
from click_didyoumean import DYMGroup
import llama_cpp.llama_chat_format as llama_chat_format
import llama_cpp.server.app as llama_app
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings
import uvicorn
import logging
from git import Repo
from os.path import splitext

from .generator.generate_data import generate_data
from .download_model import download_model
from .chat.chat import chat_cli
from .config.config import Config


class Lab(object):
    """Lab object holds high-level information about lab CLI"""

    def __init__(self):
        self.config = Config()
        FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s"
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.get_log_level())



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
    repo = Repo("taxonomy")
    updated_taxonomy_files = [u for u in repo.untracked_files if splitext(u)[1].lower() in [".yaml", ".yml"]] + \
                [d.a_path for d in repo.index.diff(None) if splitext(d.a_path)[1].lower() in [".yaml", ".yml"]]
    for f in updated_taxonomy_files:
        if splitext(f)[1] != ".yaml":
            click.secho(f"WARNING: Found {f}! Use lowercase '.yaml' extension instead.", fg="yellow")
            continue
        click.echo(f)


@cli.command()
@click.pass_context
def submit(ctx):
    """Initializes environment for labrador"""
    click.echo("please use git commands and GitHub to submit a PR to the taxonomy repo")


@cli.command()
@click.option("--model", default="./models/ggml-malachite-7b-Q4_K_M.gguf", show_default=True)
@click.option("--gpu-layers", default=-1, show_default=True)
@click.pass_context
def serve(ctx, model, gpu_layers):
    """Start a local server"""
    ctx.obj.logger.debug(f"Using model '{model}' with {gpu_layers} gpu-layers")
    settings = Settings(model=model, n_ctx=4096, n_gpu_layers=gpu_layers, verbose=False)
    app = create_app(settings=settings)
    llama_app._llama_proxy._current_model.chat_handler = llama_chat_format.Jinja2ChatFormatter(
        template="{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n' + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}", eos_token="<|endoftext|>", bos_token=""
    ).to_chat_handler()
    click.echo("Starting server process")
    click.echo("After application startup complete see http://127.0.0.1:8000/docs for API.")
    click.echo("Press CTRL+C to shutdown server.")
    uvicorn.run(app, port=8000, log_level=logging.ERROR)  # TODO: host params, etc...


@cli.command()
@click.option("--model", default="ggml-malachite-7b-Q4_K_M", show_default=True)
@click.option("--num-cpus", default=10, show_default=True)
@click.option("--num-instructions", default=100, show_default=True)
@click.option("--taxonomy", default="taxonomy", show_default=True, type=click.Path())
@click.option("--seed-file", default="./cli/generator/seed_tasks.jsonl", show_default=True, type=click.Path())
@click.pass_context
def generate(ctx, model, num_cpus, num_instructions, taxonomy, seed_file):
    """Generates synthetic data to enhance your example data"""
    ctx.obj.logger.debug(f"Generating model '{model}' using {num_cpus} cpus, taxonomy: '{taxonomy}' and seed '{seed_file}'")
    generate_data(logger=ctx.obj.logger, model_name=model, num_cpus=num_cpus,
                  num_instructions_to_generate=num_instructions, taxonomy=taxonomy, seed_tasks_path=seed_file)


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
@click.pass_context
def download(ctx):
    """Download the model(s) to train"""
    download_model()
