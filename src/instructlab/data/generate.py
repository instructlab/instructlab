# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
import click

# First Party
from instructlab import config


@click.command()
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
    model_family,
):
    """Generates synthetic data to enhance your example data"""
    # pylint: disable=C0415
    # First Party
    from instructlab.server import ensure_server

    # Local
    from .generator.generate_data import generate_data
    from .generator.utils import GenerateException

    server_process = None
    server_queue = None
    logger = logging.getLogger("TODO")
    prompt_file_path = config.DEFAULT_PROMPT_FILE
    if ctx.obj is not None:
        logger = ctx.obj.logger
        prompt_file_path = ctx.obj.config.generate.prompt_file

    if endpoint_url:
        api_base = endpoint_url
    else:
        # Third Party
        import llama_cpp

        if not llama_cpp.llama_supports_gpu_offload():
            # TODO: check for working offloading. The function only checks
            # for compile time defines like `GGML_USE_CUDA`.
            click.secho(
                "llama_cpp_python is built without hardware acceleration. "
                "ilab generate will be very slow.",
                fg="red",
            )

        try:
            server_process, api_base, server_queue = ensure_server(
                ctx.obj.logger,
                ctx.obj.config.serve,
                tls_insecure,
                tls_client_cert,
                tls_client_key,
                tls_client_passwd,
                model_family,
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
            model_family=model_family,
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
        if server_process and server_queue:
            server_process.terminate()
            server_process.join(timeout=30)
            server_queue.close()
            server_queue.join_thread()
