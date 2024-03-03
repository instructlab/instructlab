# Standard
import os
from os import listdir
from os.path import basename, dirname, exists, splitext
import logging
import sys
import subprocess
import json

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

# TODO merge requirements.txt
# copy-files target-dir:
#     cp ~/artifacts/merlinite-7b/added_tokens.json {{target-dir}}
@cli.command()
@click.option(
    "--data-dir", help="Base directory where data is stored."
)
@click.option(
    "--model-dir", help="Base directory where model is stored.", default="ibm-merlinite-7b"
)
@click.option(
    "--iters", help="Number of iterations to train LoRA", default=100
)
@click.option(
    "--remote",
    is_flag=True,
    help="Whether or not `model_dir` is remote from HuggingFace.",
)
@click.option(
    "-q",
    "--quantize",
    is_flag=True,
    help="Whether to do quantization while converting to MLX.",
)
def train(data_dir, model_dir, iters, remote, quantize):
    """
    Takes synthetic data generated locally with `lab generate` and the previous model and learns a new model using the MLX API.
    On success, writes newly learned model to {model_dir}/mlx_model, which is where `chatmlx` will look for a model.
    """
    cli_dir = os.path.dirname(os.path.abspath(__file__))

    script = os.path.join(cli_dir, "train/lora-mlx/make_data.py")
    cmd = f"{script} --data-dir {data_dir}"
    os.system('python {}'.format(cmd))

    is_macos = True # TODO detect OS
    if is_macos:
        # NOTE we can skip this if we have a way ship MLX
        # TODO convert the model from PyTorch to MLX
        # PyTorch safetensors to MLX safetensors
        model_dir_local = model_dir.replace("/", "-")
        model_dir_mlx = f"{model_dir_local}-mlx"
        model_dir_mlx_quantized = f"{model_dir_local}-mlx-q"

        dest_model_dir = ""
        quantize_arg = ""
        
        if quantize:
            dest_model_dir = model_dir_mlx_quantized
            quantize_arg =  "-q"
        else:
            dest_model_dir = model_dir_mlx

        local_arg = "" if remote else "--local"

        script = os.path.join(cli_dir, "train/lora-mlx/convert.py")
        cmd = f"{script}  --hf-path {model_dir} --mlx-path {dest_model_dir} {quantize_arg} {local_arg}"
         # python convert.py --hf-path ibm-merlinite-7b --mlx-path ibm-merlinite-7b-mlx
        os.system('python {}'.format(cmd))


        adapter_file_path = f"{dest_model_dir}/adapters.npz"
        script = os.path.join(cli_dir, "train/lora-mlx/lora.py")
        # TODO train the model with LoRA
        # python lora.py --model {{model_dir_mlx}} --train --data data_puns_shiv --adapter-file {{model_dir_mlx}}/adapters.npz --iters 300 --save-every 10 --steps-per-eval 10
        # exact command:
        # python lora.py --model ibm-merlinite-7b-mlx --train --data data_puns_shiv --adapter-file ibm-merlinite-7b-mlx/adapters.npz --iters 100 --save-every 10 --steps-per-eval 10
        cmd = f"{script} --model {dest_model_dir} --train --data {data_dir} --adapter-file {adapter_file_path} --iters {iters} --save-every 10 --steps-per-eval 10"
        os.system('python {}'.format(cmd))

        # TODO copy some downloaded files from the PyTorch model folder
        # Seems to be not a problem if working with a remote download with convert.py
        # just copy-files ibm-merlinite-7b-mlx

    else:
        click.secho(
            f"`lab train` is only implemented for macOS with M-series chips",
            fg="red",
        )

    #   Can this target a directory or does it overwrite the model on the --model directory?
    pass

@cli.command()
@click.option(
    "--data-dir", help="Base directory where data is stored."
)
@click.option(
    "--model-dir", help="Base directory where model is stored."
)
@click.option(
    "--adapter-file", help="LoRA adapter to use for test."
)
def test(data_dir, model_dir, adapter_file):
    """
    TODO
    """
    cli_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(cli_dir, "train/lora-mlx/lora.py")
    # _generate-no-lora model prompt:
    #     python lora.py --model {{model}} --no-adapter --max-tokens 100 --prompt "<|system|>\nYou are Labrador, an AI language model developed by IBM DMF (Data Model Factory) Alignment Team. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.\n<|user|>\n{{prompt}}\n<|assistant|>\n"
    # _generate model adapter prompt:
    #     python lora.py --model {{model}} --adapter-file {{model}}/{{adapter}} --max-tokens 100 --prompt "<|system|>\nYou are Labrador, an AI language model developed by IBM DMF (Data Model Factory) Alignment Team. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.\n<|user|>\n{{prompt}}\n<|assistant|>\n"


    # Load the JSON Lines file
    test_data_dir = f"{data_dir}/test.jsonl"
    with open(test_data_dir, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    SYS_PROMPT = "You are Labrador, an AI language model developed by IBM DMF (Data Model Factory) Alignment Team. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
    print("system prompt:", SYS_PROMPT)
    for (idx, example) in enumerate(test_data):
        system = example["system"]
        user = example["user"]
        print("[{}] user prompt: {}".format(idx + 1, user))
        print("expected output:", example["assistant"])
        print("\n-----model output BEFORE training----:\n")
         # just _generate ibm-merlinite-7b-mlx {{user}}
        cmd = f"{script} --model {model_dir} --no-adapter --max-tokens 100 --prompt \"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n\""
        os.system('python {}'.format(cmd))
        print("\n-----model output AFTER training----:\n")
        # just _generate ibm-merlinite-7b-mlx adapters-100.npz {{user}}
        cmd = f"{script} --model {model_dir} --adapter-file {adapter_file} --max-tokens 100 --prompt \"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n\""
        os.system('python {}'.format(cmd))

@cli.command()
@click.option(
    "--model-dir", help="Base directory where model is stored."
)
@click.option(
    "--adapter-file", help="LoRA adapter to fuse."
)
@click.option(
    "-d",
    "--de-quantize",
    help="Generate a de-quantized model.",
    is_flag=True,
)
@click.option(
    "-q",
    "--quantize",
    is_flag=True,
    help="Whether to do quantization while converting to GGUF.",
)
def convert(model_dir, adapter_file, de_quantize, quantize):
    """
    TODO
    """
    cli_dir = os.path.dirname(os.path.abspath(__file__))

    dequantize_arg = ""
    source_model_dir = model_dir
    if de_quantize:
        dequantize_arg = " -d "

    model_dir_fused= f"{source_model_dir}-fused"
    
    script = os.path.join(cli_dir, "train/lora-mlx/fuse.py")
    # cmd = cwd + " --model " + source_model_dir + " --save-path " + model_dir_fused + " --adapter-file " + adapter_file_path + dequantize_arg
    cmd = f"{script} --model {source_model_dir} --save-path {model_dir_fused} --adapter-file {adapter_file} {dequantize_arg}"
    # this combines adapter with the original model to produce the updated model
    # python fuse.py --model {{model}} --save-path {{model}}-fused --adapter-file {{model}}/adapters-100.npz
    # just copy-files {{model}}-fused
    os.system('python {}'.format(cmd))

    model_dir_fused_pt= f"{model_dir_fused}-pt"

    script = os.path.join(cli_dir, "train/lora-mlx/convert.py ")
    cmd = f"{script} --hf-path { model_dir_fused} --mlx-path {model_dir_fused_pt} --local --to-pt"
    # this converts MLX to PyTorch
    # python convert.py --hf-path {{model}} --local --mlx-path {{model}}-pt --to-pt
    # just copy-files {{model}}-pt
    os.system('{} {}'.format('python', cmd))


    script = os.path.join(cli_dir, "llamacpp/llamacpp_convert_to_gguf.py")
    cmd = f"{script} { model_dir_fused_pt} --pad-vocab"
    # use llama.cpp to convert back to GGUF
    # python $HOME/src/open-labrador/llama.cpp/convert.py {{model_dir}} --pad-vocab
    # TO DO: fix this to execute function instead
    os.system('{} {}'.format('python', cmd))

    # quantize 4-bi GGUF (optional)
    # $HOME/src/open-labrador/llama.cpp/quantize {{model_dir}}/ggml-model-f16.gguf {{model_dir}}/ggml-model-Q4_K_M.gguf Q4_K_M
    # TO DO: fix this to execute function instead
    if quantize:
        gguf_model_dir = f"{model_dir_fused_pt}/ggml-model-f16.gguf" 
        gguf_model_q_dir = f"{model_dir_fused_pt}/ggml-model-Q4_K_M.gguf"
        script = os.path.join(cli_dir, "llamacpp/quantize")
        cmd = f"{script} {gguf_model_dir} {gguf_model_q_dir} Q4_K_M"
        os.system('{}'.format(cmd))