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
from .download import download_model, clone_taxonomy, create_config_file
from .chat.chat import chat_cli
from .config.config import Config


class Lab(object):
    """Lab object holds high-level information about lab CLI"""

    def __init__(self, config):
        self.config = Config(config_yml_path=config)
        FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s"
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.get_log_level())


def configure(ctx, param, filename):
    create_config_file(filename)
    ctx.obj = Lab(filename)
    default_map = dict()
    # options in default_map must match the names of variables
    default_map["model"] = ctx.obj.config.get_serve_model_path()
    default_map["taxonomy"] = ctx.obj.config.get_generate_taxonomy()
    default_map["seed_file"] = ctx.obj.config.get_generate_seed_task_path()
    default_map["gpu_layers"] = ctx.obj.config.get_serve_n_gpu_layers()
    default_map["num_cpus"] = ctx.obj.config.get_generate_num_cpus()
    default_map["num_instructions"] = ctx.obj.config.get_generate_num_instructions()
    ctx.default_map = default_map
    # TODO: for now we have to inject values per command, so I'm injecting them all
    # but ideally we should morph our file to match what click expects
    cmds = ["init", "list", "submit", "serve", "generate", "train", "test", "chat", "download"]
    for cmd in cmds:
        ctx.default_map[cmd] = dict(default_map)


@click.group(cls=DYMGroup)
@click.option(
    "--config",
    type=click.Path(),
    default="./cli/config/config.yml",
    show_default=True,
    callback=configure,
    is_eager=True,
    help="Path to a configuration file.")
@click.pass_context
def cli(ctx, config):
    """CLI for interacting with labrador"""
    pass


@cli.command()
@click.option(
    "--repo",
    default="https://github.com/open-labrador/taxonomy.git",
    show_default=True,
    help="Labrador Taxonomy GitHub repository"
)
@click.option(
    "--branch",
    default="main",
    show_default=True,
    help="The GitHub branch of the taxonomy repository."
)
@click.pass_context
def init(ctx, repo, branch):
    """Initializes environment for labrador"""
    clone_taxonomy(repo, branch)


@cli.command()
@click.option("--taxonomy", type=click.Path(), help="Path to https://github.com/open-labrador/taxonomy/ checkout.")
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
@click.option("--model", help="Name of the model used during generation.")
@click.option("--gpu-layers", help="The number of layers to put on the GPU. The rest will be on the CPU. Defaults to -1 to move all to GPU.")
@click.option('--verbose', '-v', is_flag=True, help="Print verbose output.")
@click.pass_context
def serve(ctx, model, gpu_layers, verbose):
    """Start a local server"""
    ctx.obj.logger.info(f"Using model '{model}' with {gpu_layers} gpu-layers")
    settings = Settings(model=model, n_ctx=4096, n_gpu_layers=gpu_layers, verbose=verbose)
    app = create_app(settings=settings)
    llama_app._llama_proxy._current_model.chat_handler = llama_chat_format.Jinja2ChatFormatter(
        template="{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n' + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}", eos_token="<|endoftext|>", bos_token=""
    ).to_chat_handler()
    click.echo("Starting server process")
    click.echo("After application startup complete see http://127.0.0.1:8000/docs for API.")
    click.echo("Press CTRL+C to shutdown server.")
    uvicorn.run(app, port=8000, log_level=logging.ERROR)  # TODO: host params, etc...


@cli.command()
@click.option("--model", help="Name of the model used during generation.")
@click.option("--num-cpus", type=click.INT, help="Number of processes to use. Defaults to 10.")
@click.option("--num-instructions", type=click.INT, help="Number of instructions to generate. Defaults to 100.")
@click.option("--taxonomy", type=click.Path(), help="Path to https://github.com/open-labrador/taxonomy/ checkout.")
@click.option("--seed-file", type=click.Path(), help="Path to a seed file.")
@click.pass_context
def generate(ctx, model, num_cpus, num_instructions, taxonomy, seed_file):
    """Generates synthetic data to enhance your example data"""
    # load not exposed options from config
    prompt_path = ctx.obj.config.get_generate_prompt_file_path()

    ctx.obj.logger.debug(f"Generating model '{model}' using {num_cpus} cpus, taxonomy: '{taxonomy}' and seed '{seed_file}'")
    generate_data(logger=ctx.obj.logger, model_name=model, num_cpus=num_cpus,
                  num_instructions_to_generate=num_instructions, taxonomy=taxonomy,
                  prompt_file_path=prompt_path, seed_tasks_path=seed_file)


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
    help="GitHub repository of the hosted models."
)
@click.option(
    "--release",
    default="latest",
    show_default=True,
    help="GitHub release version of the hosted models."
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
