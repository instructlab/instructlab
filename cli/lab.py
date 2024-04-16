# pylint: disable=too-many-lines

# Standard
from glob import glob
from os.path import basename, dirname, exists, splitext
import json
import logging
import os
import shutil
import sys
import typing

# Third Party
from click_didyoumean import DYMGroup
from git import GitError, Repo
from huggingface_hub import hf_hub_download
from huggingface_hub import logging as hf_logging
import click
import yaml

# Local
# NOTE: Subcommands are using local imports to speed up startup time.
from . import config, utils

if typing.TYPE_CHECKING:
    # Third Party
    import torch


class Lab:
    """Lab object holds high-level information about ilab CLI"""

    def __init__(self, filename):
        self.config_file = filename
        self.config = config.read_config(filename)
        FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s"
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.general.log_level.upper())


# pylint: disable=unused-argument
def configure(ctx, param, filename):
    """Configure is responsible for reading the config file, initiating Lab object and CLI context."""
    # skip configuration reading when invoked command is `init`
    if len(sys.argv) > 0 and sys.argv[1] in ("init", "--help"):
        return

    if not exists(filename):
        raise click.ClickException(
            f"`{filename}` does not exists, please run `ilab init` or point to a valid configuration file using `--config=<path>`."
        )

    try:
        ctx.obj = Lab(filename)
    except config.ConfigException as ex:
        raise click.ClickException(str(ex))

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
@click.version_option(package_name="cli")
@click.pass_context
# pylint: disable=redefined-outer-name
def cli(ctx, config):
    """CLI for interacting with InstructLab.

    If this is your first time running ilab, it's best to start with `ilab init` to create the environment.
    """


@cli.command()
@click.option(
    "--interactive/--non-interactive",
    default=True,
    show_default=True,
    help="Initialize the environment assuming defaults.",
)
@click.option(
    "--model-path",
    type=click.Path(),
    default=config.DEFAULT_MODEL_PATH,
    show_default=True,
    help="Path to the model used during generation.",
)
@click.option(
    "--taxonomy-base",
    default=config.DEFAULT_TAXONOMY_BASE,
    show_default=True,
    help="Base git-ref to use when listing/generating new taxonomy.",
)
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    default=config.DEFAULT_TAXONOMY_PATH,
    show_default=True,
    help=f"Path to {config.DEFAULT_TAXONOMY_REPO} clone.",
)
@click.option(
    "--repository",
    default=config.DEFAULT_TAXONOMY_REPO,
    show_default=True,
    help="Taxonomy repository location.",
)
@click.option(
    "--min-taxonomy",
    is_flag=True,
    help="Shallow clone the taxonomy repository with minimum size. "
    "Please do not use this option if you are planning to contribute back "
    "using the same taxonomy repository. ",
)
def init(
    interactive,
    model_path,
    taxonomy_path,
    taxonomy_base,
    repository,
    min_taxonomy,
):
    """Initializes environment for InstructLab"""
    if exists(config.DEFAULT_CONFIG):
        overwrite = click.confirm(
            f"Found {config.DEFAULT_CONFIG} in the current directory, do you still want to continue?"
        )
        if not overwrite:
            return

    clone_taxonomy_repo = True
    if interactive:
        click.echo(
            "Welcome to InstructLab CLI. This guide will help you to setup your environment."
        )
        click.echo(
            "Please provide the following values to initiate the "
            "environment [press Enter for defaults]:"
        )

        taxonomy_path = utils.expand_path(
            click.prompt("Path to taxonomy repo", default=taxonomy_path)
        )

    try:
        taxonomy_contents = os.listdir(taxonomy_path)
    except FileNotFoundError:
        taxonomy_contents = []
    if taxonomy_contents:
        clone_taxonomy_repo = False
    elif interactive:
        clone_taxonomy_repo = click.confirm(
            f"`{taxonomy_path}` seems to not exist or is empty. Should I clone {repository} for you?"
        )

    # clone taxonomy repo if it needs to be cloned
    if clone_taxonomy_repo:
        click.echo(f"Cloning {repository}...")
        try:
            if not min_taxonomy:
                Repo.clone_from(repository, taxonomy_path, branch="main")
            else:
                Repo.clone_from(repository, taxonomy_path, branch="main", depth=1)
        except GitError as exc:
            click.secho(f"Failed to clone taxonomy repo: {exc}", fg="red")
            click.secho(f"Please make sure to manually run `git clone {repository}`")
            raise click.exceptions.Exit(1)

    # check if models dir exists, and if so ask for which model to use
    models_dir = dirname(model_path)
    if interactive and exists(models_dir):
        model_path = utils.expand_path(
            click.prompt("Path to your model", default=model_path)
        )
    click.echo(f"Generating `{config.DEFAULT_CONFIG}` in the current directory...")
    cfg = config.get_default_config()
    model = splitext(basename(model_path))[0]
    cfg.chat.model = model
    cfg.generate.model = model
    cfg.serve.model_path = model_path
    cfg.generate.taxonomy_path = taxonomy_path
    cfg.generate.taxonomy_base = taxonomy_base
    config.write_config(cfg)

    click.echo(
        "Initialization completed successfully, you're ready to start using `ilab`. Enjoy!"
    )


@cli.command()
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    help=f"Path to {config.DEFAULT_TAXONOMY_REPO} clone or local file path.",
)
@click.option(
    "--taxonomy-base",
    help="Base git-ref to use for taxonomy.",
)
@click.option(
    "--yaml-rules",
    type=click.Path(),
    default=None,
    help="Custom rules file for YAML linting.",
)
@click.option(
    "--quiet",
    is_flag=True,
    help="Suppress all output. Call returns 0 if check passes, 1 otherwise.",
)
@click.pass_context
def diff(ctx, taxonomy_path, taxonomy_base, yaml_rules, quiet):
    """
    Lists taxonomy files that have changed since <taxonomy-base>
    and checks that taxonomy is valid. Similar to 'git diff <ref>'.
    """
    # pylint: disable=C0415
    # Local
    from .utils import get_taxonomy_diff, read_taxonomy

    if not taxonomy_base:
        taxonomy_base = ctx.obj.config.generate.taxonomy_base
    if not taxonomy_path:
        taxonomy_path = ctx.obj.config.generate.taxonomy_path
    if not ctx.obj:
        logger = logging.getLogger(__name__)
    else:
        logger = ctx.obj.logger
    if quiet:
        try:
            read_taxonomy(logger, taxonomy_path, taxonomy_base, yaml_rules)
        except (SystemExit, yaml.YAMLError) as exc:
            raise SystemExit(1) from exc
        return
    try:
        updated_taxonomy_files = get_taxonomy_diff(taxonomy_path, taxonomy_base)
    except (SystemExit, GitError) as exc:
        click.secho(
            f"Reading taxonomy failed with the following error: {exc}",
            fg="red",
        )
        return
    for f in updated_taxonomy_files:
        click.echo(f)
    try:
        read_taxonomy(logger, taxonomy_path, taxonomy_base, yaml_rules)
    except (SystemExit, yaml.YAMLError) as exc:
        click.secho(
            f"Reading taxonomy failed with the following error: {exc}",
            fg="red",
        )
        raise SystemExit(1) from exc
    click.secho(
        f"Taxonomy in /{taxonomy_path}/ is valid :)",
        fg="green",
    )


# ilab list => ilab diff
# ilab check => ilab diff --quiet
utils.make_lab_diff_aliases(cli, diff)


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
@click.option("--num-threads", type=click.INT, help="The number of CPU threads to use.")
@click.option(
    "--max-ctx-size",
    type=click.INT,
    help="The context size is the maximum number of tokens considered by the model, for both the prompt and response. Defaults to 4096.",
)
@click.pass_context
def serve(ctx, model_path, gpu_layers, num_threads, max_ctx_size):
    """Start a local server"""
    # pylint: disable=C0415
    # Local
    from .server import ServerException, server

    ctx.obj.logger.info(
        f"Using model '{model_path}' with {gpu_layers} gpu-layers and {max_ctx_size} max context size."
    )

    try:
        host = ctx.obj.config.serve.host_port.split(":")[0]
        port = int(ctx.obj.config.serve.host_port.split(":")[1])
        server(
            ctx.obj.logger,
            model_path,
            gpu_layers,
            max_ctx_size,
            num_threads,
            host,
            port,
        )
    except ServerException as exc:
        click.secho(f"Error creating server: {exc}", fg="red")
        raise click.exceptions.Exit(1)


@cli.command()
@click.option(
    "--model",
    default=config.DEFAULT_MODEL,
    show_default=True,
    help="Name of the model used during generation.",
)
@click.option(
    "--num-cpus",
    type=click.INT,
    help="Number of processes to use.",
    default=config.DEFAULT_NUM_CPUS,
    show_default=True,
)
@click.option(
    "--chunk-word-count",
    type=click.INT,
    help="Number of words to chunk the document",
    default=config.DEFAULT_CHUNK_WORD_COUNT,
    show_default=True,
)
@click.option(
    "--num-instructions",
    type=click.INT,
    help="Number of instructions to generate.",
    default=config.DEFAULT_NUM_INSTRUCTIONS,
    show_default=True,
)
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    default=config.DEFAULT_TAXONOMY_PATH,
    show_default=True,
    help=f"Path to {config.DEFAULT_TAXONOMY_REPO} clone or local file path.",
)
@click.option(
    "--taxonomy-base",
    default=config.DEFAULT_TAXONOMY_BASE,
    show_default=True,
    help="Base git-ref to use when generating new taxonomy.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=config.DEFAULT_GENERATED_FILES_OUTPUT_DIR,
    help="Path to output generated files.",
)
@click.option(
    "--rouge-threshold",
    type=click.FLOAT,
    default=0.9,
    show_default=True,
    help="Threshold of (max) Rouge score to keep samples; 1.0 means accept all samples.",
)
@click.option(
    "--quiet",
    is_flag=True,
    help="Suppress output of synthesized instructions.",
)
@click.option(
    "--endpoint-url",
    type=click.STRING,
    help="Custom URL endpoint for OpenAI-compatible API. Defaults to the `ilab serve` endpoint.",
)
@click.option(
    "--api-key",
    type=click.STRING,
    default=config.DEFAULT_API_KEY,  # Note: do not expose default API key
    help="API key for API endpoint. [default: config.DEFAULT_API_KEY]",
)
@click.option(
    "--yaml-rules",
    type=click.Path(),
    default=None,
    help="Custom rules file for YAML linting.",
)
@click.option(
    "--server-ctx-size",
    type=click.INT,
    default=config.MAX_CONTEXT_SIZE,
    show_default=True,
    help="The context size is the maximum number of tokens the server will consider.",
)
@click.option(
    "--tls-insecure",
    is_flag=True,
    help="Disable TLS verification.",
)
@click.option(
    "--tls-client-cert",
    type=click.Path(),
    default="",
    show_default=True,
    help="Path to the TLS client certificate to use.",
)
@click.option(
    "--tls-client-key",
    type=click.Path(),
    default="",
    show_default=True,
    help="Path to the TLS client key to use.",
)
@click.option(
    "--tls-client-passwd",
    type=click.STRING,
    default="",
    help="TLS client certificate password.",
)
@click.pass_context
def generate(
    ctx,
    model,
    num_cpus,
    num_instructions,
    taxonomy_path,
    taxonomy_base,
    output_dir,
    rouge_threshold,
    quiet,
    endpoint_url,
    api_key,
    yaml_rules,
    chunk_word_count,
    server_ctx_size,
    tls_insecure,
    tls_client_cert,
    tls_client_key,
    tls_client_passwd,
):
    """Generates synthetic data to enhance your example data"""
    # pylint: disable=C0415
    # Local
    from .generator.generate_data import generate_data
    from .generator.utils import GenerateException
    from .server import ensure_server

    server_process = None
    logger = logging.getLogger("TODO")
    prompt_file_path = config.DEFAULT_PROMPT_FILE
    if ctx.obj is not None:
        logger = ctx.obj.logger
        prompt_file_path = ctx.obj.config.generate.prompt_file

    if endpoint_url:
        api_base = endpoint_url
    else:
        try:
            server_process, api_base = ensure_server(
                ctx.obj.logger,
                ctx.obj.config.serve,
                tls_insecure,
                tls_client_cert,
                tls_client_key,
                tls_client_passwd,
            )
        except Exception as exc:
            click.secho(f"Failed to start server: {exc}", fg="red")
            raise click.exceptions.Exit(1)
        if not api_base:
            api_base = ctx.obj.config.serve.api_base()
    try:
        click.echo(
            f"Generating synthetic data using '{model}' model, taxonomy:'{taxonomy_path}' against {api_base} server"
        )
        generate_data(
            logger=logger,
            api_base=api_base,
            api_key=api_key,
            model_name=model,
            num_cpus=num_cpus,
            num_instructions_to_generate=num_instructions,
            taxonomy=taxonomy_path,
            taxonomy_base=taxonomy_base,
            output_dir=output_dir,
            prompt_file_path=prompt_file_path,
            rouge_threshold=rouge_threshold,
            console_output=not quiet,
            yaml_rules=yaml_rules,
            chunk_word_count=chunk_word_count,
            server_ctx_size=server_ctx_size,
            tls_insecure=tls_insecure,
            tls_client_cert=tls_client_cert,
            tls_client_key=tls_client_key,
            tls_client_passwd=tls_client_passwd,
        )
    except GenerateException as exc:
        click.secho(
            f"Generating dataset failed with the following error: {exc}",
            fg="red",
        )
        raise click.exceptions.Exit(1)
    finally:
        if server_process:
            server_process.terminate()


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
    show_default=True,
    help="Name of system context in config file.",
)
@click.option(
    "-s",
    "--session",
    type=click.File("r"),
    help="Filepath of a dialog session file.",
)
@click.option(
    "-qq",
    "--quick-question",
    is_flag=True,
    help="Exit after answering question.",
)
@click.option(
    "-gm",
    "--greedy-mode",
    is_flag=True,
    help="Use model greedy decoding. Useful for debugging and reproducing errors.",
)
@click.option(
    "--endpoint-url",
    type=click.STRING,
    help="Custom URL endpoint for OpenAI-compatible API. Defaults to the `ilab serve` endpoint.",
)
@click.option(
    "--api-key",
    type=click.STRING,
    default=config.DEFAULT_API_KEY,  # Note: do not expose default API key
    help="API key for API endpoint. [default: config.DEFAULT_API_KEY]",
)
@click.option(
    "--tls-insecure",
    is_flag=True,
    help="Disable TLS verification.",
)
@click.option(
    "--tls-client-cert",
    type=click.Path(),
    default="",
    show_default=True,
    help="Path to the TLS client certificate to use.",
)
@click.option(
    "--tls-client-key",
    type=click.Path(),
    default="",
    show_default=True,
    help="Path to the TLS client key to use.",
)
@click.option(
    "--tls-client-passwd",
    type=click.STRING,
    default="",
    help="TLS client certificate password.",
)
@click.pass_context
def chat(
    ctx,
    question,
    model,
    context,
    session,
    quick_question,
    greedy_mode,
    endpoint_url,
    api_key,
    tls_insecure,
    tls_client_cert,
    tls_client_key,
    tls_client_passwd,
):
    """Run a chat using the modified model"""
    # pylint: disable=C0415
    # Local
    from .chat.chat import ChatException, chat_cli
    from .server import ensure_server

    if endpoint_url:
        api_base = endpoint_url
        server_process = None
    else:
        try:
            server_process, api_base = ensure_server(
                ctx.obj.logger,
                ctx.obj.config.serve,
                tls_insecure,
                tls_client_cert,
                tls_client_key,
                tls_client_passwd,
            )
        except Exception as exc:
            click.secho(f"Failed to start server: {exc}", fg="red")
            raise click.exceptions.Exit(1)
        if not api_base:
            api_base = ctx.obj.config.serve.api_base()

    try:
        chat_cli(
            logger=ctx.obj.logger,
            api_base=api_base,
            api_key=api_key,
            config=ctx.obj.config.chat,
            question=question,
            model=model,
            context=context,
            session=session,
            qq=quick_question,
            greedy_mode=greedy_mode,
            tls_insecure=tls_insecure,
            tls_client_cert=tls_client_cert,
            tls_client_key=tls_client_key,
            tls_client_passwd=tls_client_passwd,
        )
    except ChatException as exc:
        click.secho(f"Executing chat failed with: {exc}", fg="red")
        raise click.exceptions.Exit(1)
    finally:
        if server_process:
            server_process.terminate()


@cli.command()
@click.option(
    "--repository",
    default="ibm/merlinite-7b-GGUF",  # TODO: add to config.yaml
    show_default=True,
    help="Hugging Face repository of the model to download.",
)
@click.option(
    "--release",
    default="main",  # TODO: add to config.yaml
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
        if ctx.obj is not None:
            hf_logging.set_verbosity(ctx.obj.config.general.log_level.upper())

        hf_hub_download(
            repo_id=repository,
            revision=release,
            filename=filename,
            local_dir=model_dir,
        )
    except Exception as exc:
        click.secho(
            f"Downloading model failed with the following Hugging Face Hub error: {exc}",
            fg="red",
        )
        raise click.exceptions.Exit(1)


class TorchDeviceParam(click.ParamType):
    """Parse and convert device string

    Returns a torch.device object:
    - type is one of 'cpu' or 'cuda')
    - index is None or CUDA/ROCm device index (0 for first GPU)
    """

    name = "deviceinfo"
    supported_devices = {"cuda", "cpu"}

    def convert(self, value, param, ctx) -> "torch.device":
        # pylint: disable=C0415
        # Function local import, import torch can take more than a second
        # Third Party
        import torch

        if not isinstance(value, torch.device):
            try:
                device = torch.device(value)
            except RuntimeError as e:
                self.fail(str(e), param, ctx)

        if device.type not in {"cuda", "cpu"}:
            supported = ", ".join(repr(s) for s in sorted(self.supported_devices))
            self.fail(
                f"Unsupported device type '{device.type}'. Only devices "
                f"types {supported}, and indexed device strings like 'cuda:0' "
                "are supported for now.",
                param,
                ctx,
            )

        # Detect CUDA/ROCm device
        if device.type == "cuda":
            if not torch.cuda.is_available():
                self.fail(
                    f"{value}: Torch has no CUDA/ROCm support or could not detect "
                    "a compatible device.",
                    param,
                    ctx,
                )
            # map unqualified 'cuda' to current device
            if device.index is None:
                device = torch.device(device.type, torch.cuda.current_device())

        return device


TORCH_DEVICE = TorchDeviceParam()


@cli.command()
@click.option("--data-dir", help="Base directory where data is stored.", default=None)
@click.option(
    "--input-dir",
    type=click.Path(),
    show_default=True,  # TODO: set to None and change help message
    help="Path to generated files to use as input.",
)
@click.option(
    "--skip-preprocessing",
    is_flag=True,
)
@click.option(
    "--tokenizer-dir",
    help="Base directory where tokenizer is stored.",
    default=None,
    show_default=True,
)
@click.option(
    "--gguf-model-path",
    help="Local directory where gguf model is stored.",
    default=None,
    show_default=True,
)
@click.option(
    "--model-dir",
    help="Base directory where model is stored.",
    default="ibm/merlinite-7b",
    show_default=True,
)
@click.option("--iters", help="Number of iterations to train LoRA.", default=100)
@click.option(
    "--local",
    is_flag=True,
    help="Whether or not `model_dir` is remote from HuggingFace.",
)
@click.option(
    "-sq",
    "--skip-quantize",
    is_flag=True,
    help="Whether to skip quantization while converting to MLX. This parameter will be ignored if --gguf-model-path and --tokenizer-dir are specified.",
)
@click.option(
    "--num-epochs",
    type=click.INT,
    default=1,  # TODO: change this to a more reasonable default
    show_default=True,
    help="The number of times the training data is passed through the training algorithm. Please note that this value is used on Linux platforms only.",
)
@click.option(
    "--device",
    type=TORCH_DEVICE,
    show_default=True,
    default="cpu",
    help=(
        "PyTorch device for Linux training (default: 'cpu'). Use 'cuda' "
        "for NVidia CUDA / AMD ROCm GPU, 'cuda:0' for first GPU."
    ),
)
@click.option(
    "--4-bit-quant",
    "four_bit_quant",
    is_flag=True,
    show_default=True,
    default=False,
    # TODO: hidden option until llamacpp_convert_to_gguf.py supports
    # quantized models, https://github.com/instruct-lab/cli/issues/579
    hidden=True,
    help=(
        "Use BitsAndBytes for 4-bit quantization "
        "(reduces GPU VRAM usage and may slow down training)"
    ),
)
@click.pass_context
def train(
    ctx,
    data_dir,
    input_dir,
    skip_preprocessing,
    tokenizer_dir,
    gguf_model_path,
    model_dir,
    iters,
    local,
    skip_quantize,
    num_epochs,
    device: "torch.device",
    four_bit_quant: bool,
):
    """
    Takes synthetic data generated locally with `ilab generate` and the previous model and learns a new model using the MLX API.
    On success, writes newly learned model to {model_dir}/mlx_model, which is where `chatmlx` will look for a model.
    """
    # pylint: disable=C0415
    if not input_dir:
        # By default, generate output-dir is used as train input-dir
        input_dir = ctx.obj.config.generate.output_dir

    if four_bit_quant and device.type != "cuda":
        ctx.fail("--4-bit-quant option requires --device=cuda")

    # NOTE: If given a data_dir, input-dir is ignored in favor of existing!
    if data_dir is None:
        data_dir = "./taxonomy_data"
        try:
            os.listdir(input_dir)  # Test to throw FileNotFound exception
            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)
            # generated input files reverse sorted by name (contains timestamp)
            train_files = sorted(glob(input_dir + "/train_*"), reverse=True)
            test_files = sorted(glob(input_dir + "/test_*"), reverse=True)
            if len(train_files) > 1 or len(test_files) > 1:
                # pylint: disable=f-string-without-interpolation
                click.secho(
                    f"Found multiple files from `ilab generate`. Using the most recent generation.",
                    fg="yellow",
                )
            # First file is latest (by above reverse sort and timestamped names)
            shutil.copy(train_files[0], data_dir + "/train_gen.jsonl")
            shutil.copy(test_files[0], data_dir + "/test_gen.jsonl")
        except FileNotFoundError as exc:
            click.secho(
                f"Could not read directory: {exc}",
                fg="red",
            )
            raise click.exceptions.Exit(1)
        except OSError as exc:
            click.secho(
                f"Could not create data dir: {exc}",
                fg="red",
            )
            raise click.exceptions.Exit(1)
        except IndexError as exc:
            click.secho(
                f"Could not copy into data directory: {exc}",
                fg="red",
            )
            raise click.exceptions.Exit(1)

    if not utils.is_macos_with_m_chip():
        # Local
        from .llamacpp.llamacpp_convert_to_gguf import convert_llama_to_gguf
        from .train.linux_train import linux_train

        linux_train(
            train_file=train_files[0],
            test_file=test_files[0],
            num_epochs=num_epochs,
            device=device,
            four_bit_quant=four_bit_quant,
        )

        training_results_dir = "./training_results"
        os.makedirs(training_results_dir, exist_ok=True)

        final_results_dir = training_results_dir + "/final"
        os.makedirs(final_results_dir, exist_ok=True)

        # TODO: Figure out what to do when there are multiple checkpoint dirs.
        # Right now it's just copying files from the first one numerically not necessarily the best one
        added_tokens_file = glob(
            training_results_dir + "/checkpoint-*/added_tokens.json"
        )
        special_tokens_map = glob(
            training_results_dir + "/checkpoint-*/special_tokens_map.json"
        )
        tokenizer_json = glob(training_results_dir + "/checkpoint-*/tokenizer.json")
        tokenizer_model = glob(training_results_dir + "/checkpoint-*/tokenizer.model")
        tokenizer_config_json = glob(
            training_results_dir + "/checkpoint-*/tokenizer_config.json"
        )
        config_json = glob(training_results_dir + "/merged_model/config.json")
        generation_config_json = glob(
            training_results_dir + "/merged_model/generation_config.json"
        )
        safe_tensors = glob(training_results_dir + "/merged_model/*.safetensors")

        shutil.copy(added_tokens_file[0], final_results_dir)
        print("Copied ", added_tokens_file[0], "to ", final_results_dir)
        shutil.copy(special_tokens_map[0], final_results_dir)
        print("Copied ", special_tokens_map[0], "to ", final_results_dir)
        shutil.copy(tokenizer_json[0], final_results_dir)
        print("Copied ", tokenizer_json[0], "to ", final_results_dir)
        shutil.copy(tokenizer_model[0], final_results_dir)
        print("Copied ", tokenizer_model[0], "to ", final_results_dir)
        shutil.copy(tokenizer_config_json[0], final_results_dir)
        print("Copied ", tokenizer_config_json[0], "to ", final_results_dir)
        shutil.copy(config_json[0], final_results_dir)
        print("Copied ", config_json[0], "to ", final_results_dir)
        shutil.copy(generation_config_json[0], final_results_dir)
        print("Copied ", generation_config_json[0], "to ", final_results_dir)
        for file in safe_tensors:
            shutil.copy(file, final_results_dir)
            print("Copied ", file, "to ", final_results_dir)

        convert_llama_to_gguf(model=final_results_dir, pad_vocab=True)

        gguf_models_dir = "./models"
        if not os.path.isdir(gguf_models_dir):
            os.mkdir(gguf_models_dir)
        shutil.copy(final_results_dir + "/ggml-model-f16.gguf", gguf_models_dir)
        # cleanup original copy of model
        os.remove(final_results_dir + "/ggml-model-f16.gguf")
        # cleanup checkpoint dir since it's name is unpredictable
        # TODO: figure out how checkpoint dirs should be cleaned up
        # checkpoint_dirs = glob(training_results_dir + "/checkpoint*")
        # shutil.rmtree(checkpoint_dirs[0])
    else:
        # Local
        from .mlx_explore.gguf_convert_to_mlx import load
        from .mlx_explore.utils import fetch_tokenizer_from_hub
        from .train.lora_mlx.convert import convert_between_mlx_and_pytorch
        from .train.lora_mlx.lora import load_and_train
        from .train.lora_mlx.make_data import make_data

        if not skip_preprocessing:
            try:
                make_data(data_dir=data_dir)
            except FileNotFoundError as exc:
                click.secho(
                    f"Could not read from data directory: {exc}",
                    fg="red",
                )
                raise click.exceptions.Exit(1)

        # NOTE we can skip this if we have a way ship MLX
        # PyTorch safetensors to MLX safetensors
        model_dir_local = model_dir.replace("/", "-")
        model_dir_mlx = f"{model_dir_local}-mlx"
        model_dir_mlx_quantized = f"{model_dir_local}-mlx-q"

        if skip_quantize:
            dest_model_dir = model_dir_mlx
            quantize_arg = False
        else:
            dest_model_dir = model_dir_mlx_quantized
            quantize_arg = True

        if tokenizer_dir is not None and gguf_model_path is not None:
            if not local:
                tokenizer_dir_local = tokenizer_dir.replace("/", "-")
                fetch_tokenizer_from_hub(tokenizer_dir, tokenizer_dir_local)

            # no need to pass quantize_arg for now, script automatically detects if quantization is necessary based on whether gguf model is quantized or not
            load(gguf=gguf_model_path, repo=tokenizer_dir, mlx_path=dest_model_dir)

            for filename in os.listdir(model_dir_local):
                shutil.copy(
                    os.path.join(model_dir_local, filename),
                    os.path.join(dest_model_dir, filename),
                )
            shutil.rmtree(model_dir_local, ignore_errors=True)

        else:
            # Downloading PyTorch SafeTensor and Converting to MLX SafeTensor
            convert_between_mlx_and_pytorch(
                hf_path=model_dir,
                mlx_path=dest_model_dir,
                quantize=quantize_arg,
                local=local,
            )

        adapter_file_path = f"{dest_model_dir}/adapters.npz"
        # train the model with LoRA

        load_and_train(
            model=dest_model_dir,
            train=True,
            data=data_dir,
            adapter_file=adapter_file_path,
            iters=iters,
            save_every=10,
            steps_per_eval=10,
        )

        # TODO copy some downloaded files from the PyTorch model folder
        # Seems to be not a problem if working with a remote download with convert.py


@cli.command()
@click.option(
    "--data-dir",
    help="Base directory where data is stored.",
    default="./taxonomy_data",
    show_default=True,
)
@click.option(
    "--model-dir",
    help="Base directory where model is stored.",
    default="ibm-merlinite-7b-mlx-q",
    show_default=True,
)
@click.option(
    "--adapter-file",
    help="LoRA adapter to use for test. Set to 'None' to force only testing behavior from before training.",
    default="auto",
    show_default=True,
)
@utils.macos_requirement(echo_func=click.secho, exit_exception=click.exceptions.Exit)
# pylint: disable=function-redefined
def test(data_dir, model_dir, adapter_file):
    """Runs basic test to ensure model correctness"""
    # pylint: disable=C0415
    # Local
    from .train.lora_mlx.lora import load_and_train

    if adapter_file == "auto":
        adapter_file = os.path.join(model_dir, "adapters.npz")
    elif adapter_file.lower() == "none":
        adapter_file = None

    adapter_file_exists = adapter_file and os.path.exists(adapter_file)
    if adapter_file and not adapter_file_exists:
        print(
            "NOTE: Adapter file does not exist. Testing behavior before training only. - %s\n"
            % adapter_file
        )

    # Load the JSON Lines file
    test_data_dir = f"{data_dir}/test.jsonl"
    with open(test_data_dir, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    print("system prompt:", utils.get_sysprompt())
    for idx, example in enumerate(test_data):
        system = example["system"]
        user = example["user"]
        print("[{}]\n user prompt: {}".format(idx + 1, user))
        prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>"
        print("expected output:", example["assistant"])

        print("\n-----model output BEFORE training----:\n")
        load_and_train(model=model_dir, no_adapter=True, max_tokens=100, prompt=prompt)

        if adapter_file_exists:
            print("\n-----model output AFTER training----:\n")
            load_and_train(
                model=model_dir,
                adapter_file=adapter_file,
                max_tokens=100,
                prompt=prompt,
            )


@cli.command()
@click.option(
    "--model-dir",
    help="Base directory where model is stored.",
    default="ibm-merlinite-7b-mlx-q",
    show_default=True,
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
    # pylint: disable=C0415
    # Local
    from .llamacpp.llamacpp_convert_to_gguf import convert_llama_to_gguf
    from .train.lora_mlx.convert import convert_between_mlx_and_pytorch
    from .train.lora_mlx.fuse import fine_tune

    if adapter_file is None:
        adapter_file = os.path.join(model_dir, "adapters.npz")
    cli_dir = os.path.dirname(os.path.abspath(__file__))

    source_model_dir = model_dir
    model_dir_fused = f"{source_model_dir}-fused"

    # this combines adapter with the original model to produce the updated model
    fine_tune(
        model=source_model_dir,
        save_path=model_dir_fused,
        adapter_file=adapter_file,
        de_quantize=not skip_de_quantize,
    )

    model_dir_fused_pt = f"{model_dir_fused}-pt"

    # this converts MLX to PyTorch
    convert_between_mlx_and_pytorch(
        hf_path=model_dir_fused, mlx_path=model_dir_fused_pt, local=True, to_pt=True
    )

    convert_llama_to_gguf(model=model_dir_fused_pt, pad_vocab=True)

    # quantize 4-bi GGUF (optional)
    if not skip_quantize:
        gguf_model_dir = f"{model_dir_fused_pt}/ggml-model-f16.gguf"
        gguf_model_q_dir = f"{model_dir_fused_pt}/ggml-model-Q4_K_M.gguf"
        script = os.path.join(cli_dir, "llamacpp/quantize")
        cmd = f"{script} {gguf_model_dir} {gguf_model_q_dir} Q4_K_M"
        os.system("{}".format(cmd))
