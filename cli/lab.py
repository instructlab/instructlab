# Standard
from os.path import basename, dirname, exists, splitext
from os import listdir
import logging
import sys

# Third Party
from click_didyoumean import DYMGroup
from llama_cpp import llama_chat_format
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings
import click
import llama_cpp.server.app as llama_app
import uvicorn

# Local
from .chat.chat import chat_cli
from .config import create_config_file, read_config, write_config, get_dict, get_default_config, DEFAULT_CONFIG
from .download import clone_taxonomy, download_model
from .generator.generate_data import generate_data, get_taxonomy_diff


# pylint: disable=unused-argument
class Lab:
    """Lab object holds high-level information about lab CLI"""

    def __init__(self, filename):
        self.config = read_config(filename)
        FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s"
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.general.log_level.upper())

def configure(ctx, param, filename):
    # skip configuration reading when invoked command is `init`
    if len(sys.argv) > 0 and sys.argv[-1] == "init":
        return

    if not exists(filename):
        raise click.ClickException(f"`{filename}` does not exists, please run `lab init` or point to a valid configuration file using `--config=<path>`.")

    ctx.obj = Lab(filename)
    # default_map holds a dictionary with default values for each command parameters
    ctx.default_map = get_dict(ctx.obj.config)


@click.group(cls=DYMGroup)
@click.option(
    "--config",
    type=click.Path(),
    default=DEFAULT_CONFIG,
    show_default=True,
    callback=configure,
    is_eager=True,
    help="Path to a configuration file.",
)
@click.pass_context
def cli(ctx, config):
    """CLI for interacting with labrador.

    If this is your first time running lab, it's best to start with `lab init` to create the environment"""


@cli.command()
@click.pass_context
@click.option(
    "--non-interactive",
    is_flag=True,
    help="Initialize the environment assuming defaults.",
)
def init(ctx, non_interactive):
    """Initializes environment for labrador"""
    if exists(DEFAULT_CONFIG):
        overwrite = click.confirm("Found `config.yaml` in the current directory, do you still want to continue?")
        if not overwrite:
            return

    if not non_interactive:
        click.echo("Welcome to labrador CLI. This guide will help you to setup your environment.")
        click.echo("Please provide the following values to initiate the environment:")

        model_path = click.prompt("Path to your model", default="models/ggml-malachite-7b-0226-Q4_K_M.gguf")

        taxonomy_path = click.prompt("Path to taxonomy repo", default="taxonomy/")
        try:
            taxonomy_contents = listdir(taxonomy_path)
        except FileNotFoundError:
            taxonomy_contents = []
        if len(taxonomy_contents) == 0:
            repository="https://github.com/open-labrador/taxonomy"
            do_clone = click.confirm(f"`{taxonomy_path}` seems to not exists or is empty. Should I clone {repository} for you?")
            if do_clone:
                click.echo(f"Cloning {repository}...")
                err = clone_taxonomy(repository, "main")
                if err:
                    click.secho("Cloning {repository} failed with the following error: {err}", fg="red")

    click.echo(f"Generating `{DEFAULT_CONFIG}` in the current directory...")
    cfg = get_default_config()
    model = splitext(basename(model_path))[0]
    cfg.chat.model = model
    cfg.generate.model = model
    cfg.serve.model_path = model_path
    cfg.generate.taxonomy_path = taxonomy_path
    cfg.list.taxonomy_path = taxonomy_path
    write_config(cfg)
    create_config_file()

    click.echo("Initialization completed successfully, you're ready to start using `lab`. Enjoy!")


@cli.command()
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    help="Path to https://github.com/open-labrador/taxonomy/ checkout.",
)
@click.pass_context
# pylint: disable=redefined-builtin
def list(ctx, taxonomy_path):
    """List taxonomy YAML files"""
    updated_taxonomy_files = get_taxonomy_diff(taxonomy_path)
    for f in updated_taxonomy_files:
        if splitext(f)[1] != ".yaml":
            click.secho(
                f"WARNING: Found {f}! Use lowercase '.yaml' instead.", fg="yellow"
            )
            continue
        click.echo(f)


@cli.command()
@click.pass_context
def submit(ctx):
    """Initializes environment for labrador"""
    click.echo("please use git commands and GitHub to submit a PR to the taxonomy repo")


@cli.command()
@click.option(
    "--model-path",
    help="Path to the model used during generation.",
)
@click.option(
    "--gpu-layers",
    type=click.INT,
    help="The number of layers to put on the GPU. The rest will be on the CPU. Defaults to -1 to move all to GPU.",
)
@click.pass_context
def serve(ctx, model_path, gpu_layers):
    """Start a local server"""
    ctx.obj.logger.info(f"Using model '{model_path}' with {gpu_layers} gpu-layers")
    settings = Settings(model=model_path, n_ctx=4096, n_gpu_layers=gpu_layers, verbose=ctx.obj.logger.level==logging.DEBUG)
    app = create_app(settings=settings)
    llama_app._llama_proxy._current_model.chat_handler = llama_chat_format.Jinja2ChatFormatter(
        template="{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n' + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}",
        eos_token="<|endoftext|>",
        bos_token="",
    ).to_chat_handler()
    click.echo("Starting server process")
    click.echo(
        "After application startup complete see http://127.0.0.1:8000/docs for API."
    )
    click.echo("Press CTRL+C to shutdown server.")
    uvicorn.run(app, port=8000, log_level=logging.ERROR)  # TODO: host params, etc...


@cli.command()
@click.option(
    "--model",
    help="Name of the model used during generation.",
)
@click.option(
    "--num-cpus",
    type=click.INT,
    help="Number of processes to use. Defaults to 10.",
)
@click.option(
    "--num-instructions",
    type=click.INT,
    help="Number of instructions to generate. Defaults to 100.",
)
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    help="Path to https://github.com/open-labrador/taxonomy/ checkout.",
)
@click.option(
    "--seed-file",
    type=click.Path(),
    help="Path to a seed file.",
)
@click.pass_context
def generate(ctx, model, num_cpus, num_instructions, taxonomy_path, seed_file):
    """Generates synthetic data to enhance your example data"""
    ctx.obj.logger.info(f"Generating model '{model}' using {num_cpus} cpus, taxonomy: '{taxonomy_path}' and seed '{seed_file}'")
    generate_data(logger=ctx.obj.logger, model_name=model, num_cpus=num_cpus,
                  num_instructions_to_generate=num_instructions, taxonomy=taxonomy_path,
                  prompt_file_path=ctx.obj.config.generate.prompt_file, seed_tasks_path=seed_file)


@cli.command()
@click.pass_context
def train(ctx):
    """Trains labrador model"""
    click.echo("# train TBD")


@cli.command()
@click.pass_context
def test(ctx, config):
    """Perform rudimentary tests of the model"""
    click.echo("# test TBD")


@cli.command()
@click.argument(
    "question",
    nargs=-1,
    type=click.UNPROCESSED,
)
@click.option(
    "-m", "--model",
    help="Model to use",
)
@click.option(
    "-c", "--context",
    default="default",
    help="Name of system context in config file",
)
@click.option(
    "-s", "--session",
    type=click.File("r"),
    help="Filepath of a dialog session file",
)
@click.option(
    "-qq", "--quick-question",
    is_flag=True,
    help="Exit after answering question",
)
@click.pass_context
def chat(ctx, question, model, context, session, quick_question):
    """Run a chat using the modified model"""
    chat_cli(question, model, context, session, quick_question)


@cli.command()
@click.option(
    "--repo",
    default="https://github.com/open-labrador/cli.git",
    show_default=True,
    help="GitHub repository of the hosted models.",
)
@click.option(
    "--release",
    default="latest",
    show_default=True,
    help="GitHub release version of the hosted models.",
)
@click.option(
    "--model-dir",
    help="The local directory to download the model files into."
)
@click.option(
    "--pattern",
    help="Download only assets that match a glob pattern.",
)
@click.option("--pattern", help="Download only assets that match a glob pattern.")
@click.pass_context
def download(ctx, repo, release, model_dir, pattern):
    """Download the model(s) to train"""
    # Use the serve model path to get the right models in the right place, if needed
    serve_model_path = ctx.obj.config.serve.model_path
    if serve_model_path:  # if set in config
        if not model_dir:  # --model_dir takes precedence
            model_dir = dirname(serve_model_path)
        if not pattern:  # --pattern takes precedence
            pattern = basename(serve_model_path).replace(".gguf", ".*")
    click.echo("Make sure the local environment has the `gh` cli: https://cli.github.com")
    click.echo(f"Downloading models from {repo}@{release} to {model_dir}...")
    download_model(repo, release, model_dir, pattern)
