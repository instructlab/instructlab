# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import pathlib

# Third Party
import click

# First Party
from instructlab import configuration as config
from instructlab import utils

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model",
    default=config.DEFAULT_MODEL,
    show_default=True,
    help="Name of the model used during generation.",
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
    "--quiet",
    is_flag=True,
    help="Suppress output of synthesized instructions.",
)
@click.option(
    "--endpoint-url",
    type=click.STRING,
    help="Custom URL endpoint for OpenAI-compatible API. Defaults to the `ilab model serve` endpoint.",
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
@click.option(
    "--model-family",
    help="Force model family to use when picking a generation template",
)
@click.pass_context
@utils.display_params
def generate(
    ctx,
    model,
    num_instructions,
    taxonomy_path,
    taxonomy_base,
    output_dir,
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
    model_family,
):
    """Generates synthetic data to enhance your example data"""
    # pylint: disable=C0415
    # Third Party
    from instructlab.sdg.generate_data import generate_data
    from instructlab.sdg.utils import GenerateException

    # First Party
    from instructlab.model.backends.llama_cpp import ensure_server

    backend_instance = None
    if endpoint_url:
        api_base = endpoint_url
    else:
        # Third Party
        import llama_cpp as llama_cpp_python

        # First Party
        from instructlab.model.backends import backends, llama_cpp, vllm

        if not llama_cpp_python.llama_supports_gpu_offload():
            # TODO: check for working offloading. The function only checks
            # for compile time defines like `GGML_USE_CUDA`.
            click.secho(
                "llama_cpp_python is built without hardware acceleration. "
                "ilab data generate will be very slow.",
                fg="red",
            )
        model_path = pathlib.Path(ctx.obj.config.serve.model_path)
        backend = ctx.obj.config.serve.backend
        try:
            backend = backends.get(logger, model_path, backend)
        except ValueError as e:
            click.secho(f"Failed to determine backend: {e}", fg="red")
            raise click.exceptions.Exit(1)

        host = ctx.obj.config.serve.host_port.split(":")[0]
        port = int(ctx.obj.config.serve.host_port.split(":")[1])

        if backend == backends.LLAMA_CPP:
            # Instantiate the llama server
            backend_instance = llama_cpp.Server(
                logger=logger,
                api_base=ctx.obj.config.serve.api_base(),
                model_path=model_path,
                gpu_layers=ctx.obj.config.serve.gpu_layers,
                max_ctx_size=ctx.obj.config.serve.max_ctx_size,
                num_threads=None,  # exists only as a flag not a config
                model_family=model_family,
                host=host,
                port=port,
            )

        if backend == backends.VLLM:
            # Instantiate the vllm server
            backend_instance = vllm.Server(
                logger=logger,
                api_base=ctx.obj.config.serve.api_base(),
                model_path=model_path,
                model_family=model_family,
                host=host,
                port=port,
            )

        try:
            # Run the llama server
            backend_instance.run_detached(
                tls_insecure, tls_client_cert, tls_client_key, tls_client_passwd
            )
            # api_base will be set by run_detached
            api_base = backend_instance.api_base
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
            logger=logging.getLogger("instructlab.sdg"),  # TODO: remove
            api_base=api_base,
            api_key=api_key,
            model_family=model_family,
            model_name=model,
            num_instructions_to_generate=num_instructions,
            taxonomy=taxonomy_path,
            taxonomy_base=taxonomy_base,
            output_dir=output_dir,
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
        if backend_instance is not None:
            backend_instance.shutdown()
