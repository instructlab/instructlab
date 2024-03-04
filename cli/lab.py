# Standard
from glob import glob
from os.path import basename, dirname, exists, splitext
import json
import logging
import os
import shutil
import sys

# Third Party
from click_didyoumean import DYMGroup
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from llama_cpp import llama_chat_format
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings
import click
import llama_cpp.server.app as llama_app
import openai
import uvicorn

# Local
from . import config, utils
from .chat.chat import ChatException, chat_cli
from .download import DownloadException, clone_taxonomy
from .generator.generate_data import (
    GenerateException,
    generate_data,
    get_taxonomy_diff,
    read_taxonomy,
)


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
    if len(sys.argv) > 0 and sys.argv[1] == "init":
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
    """CLI for interacting with InstructLab.

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
    """Initializes environment for InstructLab"""
    if exists(config.DEFAULT_CONFIG):
        overwrite = click.confirm(
            f"Found {config.DEFAULT_CONFIG} in the current directory, do you still want to continue?"
        )
        if not overwrite:
            return

    if interactive:
        click.echo(
            "Welcome to InstructLab CLI. This guide will help you to setup your environment."
        )
        click.echo("Please provide the following values to initiate the environment:")

        taxonomy_path = utils.expand_path(
            click.prompt("Path to taxonomy repo", default=taxonomy_path)
        )
        try:
            taxonomy_contents = os.listdir(taxonomy_path)
        except FileNotFoundError:
            taxonomy_contents = []
        if len(taxonomy_contents) == 0:
            do_clone = click.confirm(
                f"`{taxonomy_path}` seems to not exists or is empty. Should I clone {repository} for you?"
            )
            if do_clone:
                click.echo(f"Cloning {repository}...")
                try:
                    clone_taxonomy(
                        repository,
                        config.DEFAULT_TAXONOMY_BRANCH,
                        taxonomy_path,
                        min_taxonomy,
                    )
                except DownloadException as exc:
                    click.secho(
                        f"Cloning {repository} failed with the following error: {exc}",
                        fg="red",
                    )
                    sys.exit(1)

        # check if models dir exists, and if so ask for which model to use
        models_dir = dirname(model_path)
        if exists(models_dir):
            model_path = utils.expand_path(
                click.prompt("Path to your model", default=model_path)
            )

    # non-interactive part of the generation
    click.echo(f"Generating `{config.DEFAULT_CONFIG}` in the current directory...")
    cfg = config.get_default_config()
    model = splitext(basename(model_path))[0]
    cfg.chat.model = model
    cfg.generate.model = model
    cfg.serve.model_path = model_path
    cfg.generate.taxonomy_path = taxonomy_path
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
    Lists taxonomy files that have changed since last commit.
    Similar to 'git diff'
    """
    if not taxonomy_path:
        taxonomy_path = ctx.obj.config.generate.taxonomy_path
    updated_taxonomy_files = get_taxonomy_diff(taxonomy_path)
    for f in updated_taxonomy_files:
        if splitext(f)[1] != ".yaml":
            click.secho(
                f"WARNING: Found {f}! Use lowercase '.yaml' instead.", fg="yellow"
            )
            continue
        click.echo(f)


@cli.command()
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    help=f"Path to {config.DEFAULT_TAXONOMY_REPO} clone.",
)
@click.pass_context
def check(ctx, taxonomy_path):
    """Check that taxonomy is valid"""
    if not taxonomy_path:
        taxonomy_path = ctx.obj.config.generate.taxonomy_path
    ctx.obj.logger.debug(f"Checking taxonomy: '{taxonomy_path}'")
    read_taxonomy(ctx.obj.logger, taxonomy_path)


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
        sys.exit(1)
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
        sys.exit(1)
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
    "--output-dir",
    type=click.Path(),
    help=f"Path to output generated files",
)
@click.option(
    "--seed-file",
    type=click.Path(),
    help="Path to a seed file.",
)
@click.option(
    "--rouge-threshold",
    type=click.FLOAT,
    default=0.9,
    help="Threshold of (max) Rouge score to keep samples; 1.0 means accept all samples.",
)
@click.option(
    "--quiet",
    is_flag=True,
    help="Suppress output of synthesized instructions",
)
@click.pass_context
def generate(
    ctx,
    model,
    num_cpus,
    num_instructions,
    taxonomy_path,
    output_dir,
    seed_file,
    rouge_threshold,
    quiet,
):
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
            output_dir=output_dir,
            prompt_file_path=ctx.obj.config.generate.prompt_file,
            seed_tasks_path=seed_file,
            rouge_threshold=rouge_threshold,
            console_output=not quiet,
        )
    except GenerateException as exc:
        click.secho(
            f"Generating dataset failed with the following error: {exc}",
            fg="red",
        )
    except openai.APIConnectionError as exc:
        click.secho(
            f"Error connecting to the server: {exc.__cause__}",
            fg="red",
        )
        sys.exit(1)


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
        sys.exit(1)


@cli.command()
@click.option(
    "--repository",
    default="ibm/merlinite-7b-GGUF",
    show_default=True,
    help="Hugging Face repository of the model to download.",
)
@click.option(
    "--release",
    default="main",
    show_default=True,
    help="The git revision of the model to download - e.g. a branch, tag, or commit hash.",
)
@click.option(
    "--filename",
    default=basename(config.DEFAULT_MODEL_PATH),
    show_default=True,
    help="Name of the model file to download from the Hugging Face repository.",
)
@click.option(
    "--model-dir",
    default=dirname(config.DEFAULT_MODEL_PATH),
    show_default=True,
    help="The local directory to download the model files into.",
)
@click.pass_context
def download(ctx, repository, release, filename, model_dir):
    """Download the model(s) to train"""
    click.echo(f"Downloading model from {repository}@{release} to {model_dir}...")
    try:
        hf_hub_download(
            repo_id=repository, revision=release, filename=filename, local_dir=model_dir
        )
    except HfHubHTTPError as exc:
        click.secho(
            f"Downloading model failed with the following Hugging Face Hub error: {exc}",
            fg="red",
        )
        sys.exit(1)


@cli.command()
@click.option("--data-dir", help="Base directory where data is stored.", default=None)
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    help=f"Path to {config.DEFAULT_TAXONOMY_REPO} clone.",
)
@click.option(
    "--skip-preprocessing",
    is_flag=True,
)
@click.option(
    "--model-dir",
    help="Base directory where model is stored.",
    default="ibm/merlinite-7b",
)
@click.option("--iters", help="Number of iterations to train LoRA", default=100)
@click.option(
    "--local",
    is_flag=True,
    help="Whether or not `model_dir` is remote from HuggingFace.",
)
@click.option(
    "-sq",
    "--skip-quantize",
    is_flag=True,
    help="Whether to skip quantization while converting to MLX.",
)
@click.pass_context
@utils.macos_requirement(echo_func=click.secho, exit_exception=click.exceptions.Exit)
def train(
    ctx,
    data_dir,
    taxonomy_path,
    skip_preprocessing,
    model_dir,
    iters,
    local,
    skip_quantize,
):
    """
    Takes synthetic data generated locally with `lab generate` and the previous model and learns a new model using the MLX API.
    On success, writes newly learned model to {model_dir}/mlx_model, which is where `chatmlx` will look for a model.
    """
    cli_dir = os.path.dirname(os.path.abspath(__file__))
    if not taxonomy_path:
        taxonomy_path = ctx.obj.config.generate.taxonomy_path

    if data_dir is None:
        data_dir = "./taxonomy_data"
        try:
            os.listdir(taxonomy_path)
            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)
            train_files = glob(taxonomy_path + "/train_*")
            test_files = glob(taxonomy_path + "/test_*")
            if len(train_files) > 1 or len(test_files) > 1:
                # pylint: disable=f-string-without-interpolation
                click.secho(
                    f"Found multiple files from `lab generate`. Using the first one.",
                    fg="yellow",
                )
            shutil.copy(train_files[0], data_dir + "/train_gen.jsonl")
            shutil.copy(test_files[0], data_dir + "/test_gen.jsonl")
        except FileNotFoundError as exc:
            click.secho(
                f"Could not read taxonomy directory: {exc}",
                fg="red",
            )
            sys.exit(1)
        except OSError as exc:
            click.secho(
                f"Could not create data dir: {exc}",
                fg="red",
            )
            sys.exit(1)
        except IndexError as exc:
            click.secho(
                f"Could not copy into data directory: {exc}",
                fg="red",
            )
            sys.exit(1)

    if not skip_preprocessing:
        script = os.path.join(cli_dir, "train/lora-mlx/make_data.py")
        cmd = f"{script} --data-dir {data_dir}"
        os.system("python {}".format(cmd))

    # NOTE we can skip this if we have a way ship MLX
    # PyTorch safetensors to MLX safetensors
    model_dir_local = model_dir.replace("/", "-")
    model_dir_mlx = f"{model_dir_local}-mlx"
    model_dir_mlx_quantized = f"{model_dir_local}-mlx-q"

    dest_model_dir = ""
    quantize_arg = ""

    if not skip_quantize:
        dest_model_dir = model_dir_mlx_quantized
        quantize_arg = "-q"
    else:
        dest_model_dir = model_dir_mlx

    local_arg = "--local" if local else ""

    script = os.path.join(cli_dir, "train/lora-mlx/convert.py")
    cmd = f"{script}  --hf-path {model_dir} --mlx-path {dest_model_dir} {quantize_arg} {local_arg}"
    os.system("python {}".format(cmd))

    adapter_file_path = f"{dest_model_dir}/adapters.npz"
    script = os.path.join(cli_dir, "train/lora-mlx/lora.py")
    # train the model with LoRA
    cmd = f"{script} --model {dest_model_dir} --train --data {data_dir} --adapter-file {adapter_file_path} --iters {iters} --save-every 10 --steps-per-eval 10"
    os.system("python {}".format(cmd))

    # TODO copy some downloaded files from the PyTorch model folder
    # Seems to be not a problem if working with a remote download with convert.py


@cli.command()
@click.option(
    "--data-dir", help="Base directory where data is stored.", default="./taxonomy_data"
)
@click.option(
    "--model-dir",
    help="Base directory where model is stored.",
    default="ibm-merlinite-7b-mlx-q",
)
@click.option("--adapter-file", help="LoRA adapter to use for test.", default=None)
@utils.macos_requirement(echo_func=click.secho, exit_exception=click.exceptions.Exit)
# pylint: disable=function-redefined
def test(data_dir, model_dir, adapter_file):
    """Runs basic test to ensure model correctness"""
    if adapter_file is None:
        adapter_file = os.path.join(model_dir, "adapters.npz")
    cli_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(cli_dir, "train/lora-mlx/lora.py")

    # Load the JSON Lines file
    test_data_dir = f"{data_dir}/test.jsonl"
    with open(test_data_dir, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    SYS_PROMPT = "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
    print("system prompt:", SYS_PROMPT)
    for (idx, example) in enumerate(test_data):
        system = example["system"]
        user = example["user"]
        print("[{}]\n user prompt: {}".format(idx + 1, user))
        print("expected output:", example["assistant"])
        print("\n-----model output BEFORE training----:\n")
        cmd = f'{script} --model {model_dir} --no-adapter --max-tokens 100 --prompt "<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"'
        os.system("python {}".format(cmd))
        print("\n-----model output AFTER training----:\n")
        cmd = f'{script} --model {model_dir} --adapter-file {adapter_file} --max-tokens 100 --prompt "<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"'
        os.system("python {}".format(cmd))


@cli.command()
@click.option(
    "--model-dir",
    help="Base directory where model is stored.",
    default="ibm-merlinite-7b-mlx-q",
)
@click.option("--adapter-file", help="LoRA adapter to fuse.", default=None)
@click.option(
    "-sd",
    "--skip-de-quantize",
    help="Skip de-quantization.",
    is_flag=True,
)
@click.option(
    "-sq",
    "--skip-quantize",
    is_flag=True,
    help="Whether to skip quantization while converting to GGUF.",
)
@utils.macos_requirement(echo_func=click.secho, exit_exception=click.exceptions.Exit)
def convert(model_dir, adapter_file, skip_de_quantize, skip_quantize):
    """Converts model to GGUF"""
    if adapter_file is None:
        adapter_file = os.path.join(model_dir, "adapters.npz")
    cli_dir = os.path.dirname(os.path.abspath(__file__))

    dequantize_arg = ""
    source_model_dir = model_dir
    if not skip_de_quantize:
        dequantize_arg = " -d "

    model_dir_fused = f"{source_model_dir}-fused"

    script = os.path.join(cli_dir, "train/lora-mlx/fuse.py")
    cmd = f"{script} --model {source_model_dir} --save-path {model_dir_fused} --adapter-file {adapter_file} {dequantize_arg}"
    # this combines adapter with the original model to produce the updated model
    os.system("python {}".format(cmd))

    model_dir_fused_pt = f"{model_dir_fused}-pt"

    script = os.path.join(cli_dir, "train/lora-mlx/convert.py ")
    cmd = f"{script} --hf-path { model_dir_fused} --mlx-path {model_dir_fused_pt} --local --to-pt"
    # this converts MLX to PyTorch
    os.system("{} {}".format("python", cmd))

    script = os.path.join(cli_dir, "llamacpp/llamacpp_convert_to_gguf.py")
    cmd = f"{script} { model_dir_fused_pt} --pad-vocab"
    # use llama.cpp to convert back to GGUF
    os.system("{} {}".format("python", cmd))

    # quantize 4-bi GGUF (optional)
    if not skip_quantize:
        gguf_model_dir = f"{model_dir_fused_pt}/ggml-model-f16.gguf"
        gguf_model_q_dir = f"{model_dir_fused_pt}/ggml-model-Q4_K_M.gguf"
        script = os.path.join(cli_dir, "llamacpp/quantize")
        cmd = f"{script} {gguf_model_dir} {gguf_model_q_dir} Q4_K_M"
        os.system("{}".format(cmd))
