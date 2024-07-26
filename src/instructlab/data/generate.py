# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.configuration import DEFAULTS

# Local
from ..utils import http_client

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model",
    default=lambda: DEFAULTS.DEFAULT_MODEL,
    show_default="The default model used by the instructlab system, located in the data directory.",
    help="Name of the model used during generation.",
)
@click.option(
    "--num-cpus",
    type=click.INT,
    help="Number of processes to use.",
    default=DEFAULTS.NUM_CPUS,
    show_default=True,
)
@click.option(
    "--chunk-word-count",
    type=click.INT,
    help="Number of words to chunk the document",
    default=DEFAULTS.CHUNK_WORD_COUNT,
    show_default=True,
)
# TODO - DEPRECATED - Remove in a future release
@click.option(
    "--num-instructions",
    type=click.INT,
    default=-1,
    hidden=True,
)
@click.option(
    "--sdg-scale-factor",
    type=click.INT,
    help="Number of instructions to generate for each seed example. The examples map to sample q&a pairs for new skills. For knowledge, examples are generated with both the sample q&a pairs, as well as chunks of the knowledge document(s), so the resulting data set is typically larger for a knowledge addition for the same value of `--sdg-scale-factor`.",
    show_default=True,
)
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    default=lambda: DEFAULTS.TAXONOMY_DIR,
    show_default="The default taxonomy path used by instructlab, located in the data directory.",
    help="Path to where the taxonomy is located.",
)
@click.option(
    "--taxonomy-base",
    default=DEFAULTS.TAXONOMY_BASE,
    show_default=True,
    help="Base git-ref to use when generating new taxonomy.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=lambda: DEFAULTS.DATASETS_DIR,
    show_default="The default output directory used by instructlab, located in the data directory.",
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
    default=DEFAULTS.API_KEY,  # Note: do not expose default API key
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
    default=DEFAULTS.MAX_CONTEXT_SIZE,
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
@click.option(
    "--pipeline",
    type=click.STRING,
    default="simple",
    # Hidden until instructlab-sdg releases a version with multiple pipelines
    # For now only "simple" is supported in the latest release.
    hidden=True,
    help="Data generation pipeline to use. Available: simple, full, or a valid path to a directory of pipeline worlfow YAML files. Note that 'full' requires a larger teacher model, Mixtral-8x7b.",
)
@click.option(
    "--enable-serving-output",
    is_flag=True,
    help="Print serving engine logs.",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(),
    default=None,
    show_default="The directory used by instructlab to checkpoint/load pipeline output to/from",
    help="Path to output generated files.",
)
@click.pass_context
@clickext.display_params
def generate(
    ctx,
    model,
    num_cpus,
    num_instructions,
    sdg_scale_factor,
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
    pipeline,
    enable_serving_output,
    checkpoint_dir,
):
    """Generates synthetic data to enhance your example data"""
    # pylint: disable=import-outside-toplevel
    # Third Party
    from instructlab.sdg.generate_data import generate_data
    from instructlab.sdg.utils import GenerateException

    if num_instructions != -1:
        click.secho(
            "The --num-instructions flag is deprecated. Please use --sdg-scale-factor instead.",
            fg="yellow",
        )

    prompt_file_path = DEFAULTS.PROMPT_FILE

    if ctx.obj is not None:
        prompt_file_path = ctx.obj.config.generate.prompt_file

    backend_instance = None
    if endpoint_url:
        api_base = endpoint_url
    else:
        # First Party
        from instructlab.model.backends import backends

        ctx.obj.config.serve.llama_cpp.llm_family = model_family
        backend_instance = backends.select_backend(ctx.obj.config.generate.teacher)

        try:
            # Run the backend server
            api_base = backend_instance.run_detached(
                http_client(ctx.params), background=not enable_serving_output
            )
        except Exception as exc:
            click.secho(f"Failed to start server: {exc}", fg="red")
            raise click.exceptions.Exit(1)
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
            num_cpus=num_cpus,
            num_instructions_to_generate=sdg_scale_factor,
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
            pipeline=pipeline,
            checkpoint_dir=checkpoint_dir
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
