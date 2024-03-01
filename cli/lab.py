# Standard
from os import listdir
from os.path import basename, dirname, exists, splitext
import logging
import sys
import subprocess

# Third Party
from click_didyoumean import DYMGroup
from llama_cpp import llama_chat_format
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings
import click
import llama_cpp.server.app as llama_app
import uvicorn

# Local
from . import config
from .chat.chat import ChatException, chat_cli
from .download import DownloadException, clone_taxonomy, download_model
from .generator.generate_data import GenerateException, generate_data, get_taxonomy_diff


# pylint: disable=unused-argument
class Lab:
    """Lab object holds high-level information about lab CLI"""

    def __init__(self, filename):
        self.config_file = filename
        self.config = config.read_config(filename)
        FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s"
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.general.log_level.upper())


def configure(ctx, param, filename):
    # skip configuration reading when invoked command is `init`
    if len(sys.argv) > 0 and sys.argv[-1] == "init":
        return

    if not exists(filename):
        raise click.ClickException(
            f"`{filename}` does not exists, please run `lab init` or point to a valid configuration file using `--config=<path>`."
        )

    ctx.obj = Lab(filename)
    # default_map holds a dictionary with default values for each command parameters
    ctx.default_map = config.get_dict(ctx.obj.config)


@click.group(cls=DYMGroup)
@click.option(
    "--config",
    type=click.Path(),
    default=config.DEFAULT_CONFIG,
    show_default=True,
    callback=configure,
    is_eager=True,
    help="Path to a configuration file.",
)
@click.pass_context
# pylint: disable=redefined-outer-name
def cli(ctx, config):
    """CLI for interacting with labrador.

    If this is your first time running lab, it's best to start with `lab init` to create the environment
    """


@cli.command()
@click.pass_context
@click.option(
    "--interactive",
    is_flag=True,
    default=True,
    help="Initialize the environment assuming defaults.",
)
@click.option(
    "--model-path",
    type=click.Path(),
    default=config.DEFAULT_MODEL_PATH,
    help="Path to the model used during generation.",
)
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    default=config.DEFAULT_TAXONOMY_PATH,
    help=f"Path to {config.DEFAULT_TAXONOMY_REPO} clone.",
)
@click.option(
    "--repository",
    default=config.DEFAULT_TAXONOMY_REPO,
    help="Taxonomy repository location.",
)
@click.option(
    "--min_taxonomy",
    is_flag=True,
    help="Shallow clone the taxonomy repository with minimum size. "
    "Please do not use this option if you are planning to contribute back "
    "using the same taxonomy repository. ",
)
def init(ctx, interactive, model_path, taxonomy_path, repository, min_taxonomy):
    """Initializes environment for labrador"""
    if exists(config.DEFAULT_CONFIG):
        overwrite = click.confirm(
            f"Found {config.DEFAULT_CONFIG} in the current directory, do you still want to continue?"
        )
        if not overwrite:
            return

    if interactive:
        click.echo(
            "Welcome to labrador CLI. This guide will help you to setup your environment."
        )
        click.echo("Please provide the following values to initiate the environment:")

        taxonomy_path = click.prompt("Path to taxonomy repo", default=taxonomy_path)
        try:
            taxonomy_contents = listdir(taxonomy_path)
        except FileNotFoundError:
            taxonomy_contents = []
        if len(taxonomy_contents) == 0:
            do_clone = click.confirm(
                f"`{taxonomy_path}` seems to not exists or is empty. Should I clone {repository} for you?"
            )
            if do_clone:
                click.echo(f"Cloning {repository}...")
                try:
                    clone_taxonomy(repository, "main", taxonomy_path, min_taxonomy)
                except DownloadException as exc:
                    click.secho(
                        f"Cloning {repository} failed with the following error: {exc}",
                        fg="red",
                    )

        # check if models dir exists, and if so ask for which model to use
        models_dir = dirname(model_path)
        if exists(models_dir):
            model_path = click.prompt("Path to your model", default=model_path)

    # non-interactive part of the generation
    click.echo(f"Generating `{config.DEFAULT_CONFIG}` in the current directory...")
    cfg = config.get_default_config()
    model = splitext(basename(model_path))[0]
    cfg.chat.model = model
    cfg.generate.model = model
    cfg.serve.model_path = model_path
    cfg.generate.taxonomy_path = taxonomy_path
    cfg.list.taxonomy_path = taxonomy_path
    config.write_config(cfg)

    click.echo(
        "Initialization completed successfully, you're ready to start using `lab`. Enjoy!"
    )


@cli.command()
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    help=f"Path to {config.DEFAULT_TAXONOMY_REPO} clone.",
)
@click.pass_context
# pylint: disable=redefined-builtin
def list(ctx, taxonomy_path):
    """
    Lists taxonomy files that have changed (modified or untracked).
    Similar to 'git diff'
    """
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
    type=click.Path(),
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
    settings = Settings(
        model=model_path,
        n_ctx=4096,
        n_gpu_layers=gpu_layers,
        verbose=ctx.obj.logger.level == logging.DEBUG,
    )
    try:
        app = create_app(settings=settings)
    except ValueError as err:
        click.secho(
            f"Creating App using model failed with following value error: {err}",
            fg="red",
        )
    try:
        llama_app._llama_proxy._current_model.chat_handler = llama_chat_format.Jinja2ChatFormatter(
            template="{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n' + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}",
            eos_token="<|endoftext|>",
            bos_token="",
        ).to_chat_handler()
    # pylint: disable=broad-exception-caught
    except Exception as e:
        click.secho(
            f"Error creating chat handler: {e}",
            fg="red",
        )
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
    help=f"Path to {config.DEFAULT_TAXONOMY_REPO} clone.",
)
@click.option(
    "--seed-file",
    type=click.Path(),
    help="Path to a seed file.",
)
@click.pass_context
def generate(ctx, model, num_cpus, num_instructions, taxonomy_path, seed_file):
    """Generates synthetic data to enhance your example data"""
    ctx.obj.logger.info(
        f"Generating model '{model}' using {num_cpus} cpus, taxonomy: '{taxonomy_path}' and seed '{seed_file}'"
    )
    try:
        generate_data(
            logger=ctx.obj.logger,
            model_name=model,
            num_cpus=num_cpus,
            num_instructions_to_generate=num_instructions,
            taxonomy=taxonomy_path,
            prompt_file_path=ctx.obj.config.generate.prompt_file,
            seed_tasks_path=seed_file,
        )
    except GenerateException as exc:
        click.secho(
            f"Generating dataset failed with the following error: {exc}",
            fg="red",
        )


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
    "question",
    nargs=-1,
    type=click.UNPROCESSED,
)
@click.option(
    "-m",
    "--model",
    help="Model to use",
)
@click.option(
    "-c",
    "--context",
    default="default",
    help="Name of system context in config file",
)
@click.option(
    "-s",
    "--session",
    type=click.File("r"),
    help="Filepath of a dialog session file",
)
@click.option(
    "-qq",
    "--quick-question",
    is_flag=True,
    help="Exit after answering question",
)
@click.pass_context
def chat(ctx, question, model, context, session, quick_question):
    """Run a chat using the modified model"""
    try:
        chat_cli(
            ctx.obj.config.chat,
            ctx.obj.logger,
            question,
            model,
            context,
            session,
            quick_question,
        )
    except ChatException as exc:
        click.secho(f"Executing chat failed with: {exc}", fg="red")


@cli.command()
@click.option(
    "--repository",
    default="https://github.com/open-labrador/cli.git",
    show_default=True,
    help="GitHub repository of the hosted models.",
)
@click.option(
    "--release",
    default=config.DEFAULT_DOWNLOAD_TAG,
    show_default=True,
    help="GitHub release version of the hosted models.",
)
@click.option(
    "--model-dir", help="The local directory to download the model files into."
)
@click.option(
    "--pattern",
    help="Download only assets that match a glob pattern.",
)
@click.option("--pattern", help="Download only assets that match a glob pattern.")
@click.pass_context
def download(ctx, repository, release, model_dir, pattern):
    """Download the model(s) to train"""
    # Use the serve model path to get the right models in the right place, if needed
    serve_model_path = ctx.obj.config.serve.model_path
    if serve_model_path:  # if set in config
        if not model_dir:  # --model_dir takes precedence
            model_dir = dirname(serve_model_path)
        if not pattern:  # --pattern takes precedence
            pattern = basename(serve_model_path).replace(".gguf", ".*")
    click.echo(
        "Make sure the local environment has the `gh` cli: https://cli.github.com"
    )
    click.echo(f"Downloading models from {repository}@{release} to {model_dir}...")
    try:
        download_model(repository, release, model_dir, pattern)
    except DownloadException as exc:
        click.secho(
            f"Downloading models failed with the following error: {exc}",
            fg="red",
        )


@cli.command()
@click.option("--prompt", default="What should your new model infer on?", help="")
@click.option(
    "--models-dir", help="Base directory where models are stored.", default="./models"
)
def chat_mlx(prompt, models_dir):
    """
    Usage:
        lab chatmlx --prompt 'something' --models-dir ./models/lbdr_2_model

    Works like:
        lab chat -qq 'some prompt'
    """

    # run 'python -m mlx_lm --model {model} --prompt {prompt}' and stream response to standard out.

    # Will take a single prompt, give output.

    # Probably creates a subprocess with the above command line program and directs output to standard out.

    result = subprocess.run(
        ["python3", "-m", "mlx_lm.generate", "--prompt", prompt, "--model", models_dir],
        stdout=subprocess.PIPE,
    )

    print(result.stdout.decode("utf-8"))


@cli.command()
@click.option(
    "--models-dir", help="Base directory where models are stored.", default="./models"
)
def train_mlx(models_dir):
    """
    Takes synthetic data generated locally with `lab generate` and the previous model and learns a new model using the MLX API.
    On success, writes newly learned model to {models_dir}/mlx_model, which is where `chatmlx` will look for a model.
    """

    # prepare model
    #   python ./models/mlx_scripts/prepare_model.py

    # convert model
    #   python ./models/mlx_scripts/convert.py --hf-path malachite-7b

    # make data
    #   python ./models/mlx_scripts/make_data.py

    # train model
    #   python ./models/mlx_scripts/lora.py --model mlx_model --train --data data_puns --lora-layers 32 --iters 300 --save-every 10 --steps-per-eval 10
    #   Can this target a directory or does it overwrite the model on the --model directory?

    pass
