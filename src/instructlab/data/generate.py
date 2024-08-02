# SPDX-License-Identifier: Apache-2.0

# Standard
import copy
import logging
import os.path

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
    "model_path",
    cls=clickext.ConfigOption,
    config_sections="teacher",
    show_default=False,
    help="Name of the model used during generation.",
)
@click.option(
    "--num-cpus",
    type=click.INT,
    help="Number of processes to use.",
    cls=clickext.ConfigOption,
)
@click.option(
    "--chunk-word-count",
    type=click.INT,
    help="Number of words to chunk the document",
    cls=clickext.ConfigOption,
)
# TODO - DEPRECATED - Remove in a future release
@click.option(
    "--num-instructions",
    type=click.INT,
    hidden=True,
    cls=clickext.ConfigOption,
)
@click.option(
    "--sdg-scale-factor",
    type=click.INT,
    help="Number of instructions to generate for each seed example. The examples map to sample q&a pairs for new skills. For knowledge, examples are generated with both the sample q&a pairs, as well as chunks of the knowledge document(s), so the resulting data set is typically larger for a knowledge addition for the same value of `--sdg-scale-factor`.",
    cls=clickext.ConfigOption,
)
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    help="Path to where the taxonomy is located.",
    cls=clickext.ConfigOption,
)
@click.option(
    "--taxonomy-base",
    help="Base git-ref to use when generating new taxonomy.",
    cls=clickext.ConfigOption,
)
@click.option(
    "--output-dir",
    type=click.Path(),
    help="Path to output generated files.",
    cls=clickext.ConfigOption,
)
# TODO: no config option for --rouge-threshold
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
    default=DEFAULTS.API_KEY,
    show_default=True,
    help="API key for API endpoint.",
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
    type=click.STRING,
    help="Force model family to use when picking a generation template",
)
@click.option(
    "--pipeline",
    type=click.STRING,
    default="simple",
    help="Data generation pipeline to use. Available: simple, full, or a valid path to a directory of pipeline workflow YAML files. Note that 'full' requires a larger teacher model, Mixtral-8x7b.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=0),
    default=None,
    help="Number of elements to process in each batch through the SDG pipeline. Enabled by default for the vLLM serving backend, with a batch size of 8 chosen based on experiments to optimize for throughput. Use 0 to disable.",
)
@click.option(
    "--enable-serving-output",
    is_flag=True,
    help="Print serving engine logs.",
)
@click.option(
    "--gpus",
    type=click.IntRange(min=0),
    cls=clickext.ConfigOption,
    help="Number of GPUs to run generation on",
)
@click.pass_context
@clickext.display_params
def generate(
    ctx,
    model_path,
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
    tls_insecure: bool,
    tls_client_cert: str | None,
    tls_client_key: str | None,
    tls_client_passwd: str | None,
    model_family,
    pipeline,
    enable_serving_output,
    batch_size,
    gpus,
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

    # If batch size is not set explicitly, default to 8
    # Once https://github.com/instructlab/sdg/issues/224 is resolved we can
    # pass batch_size=None to the library instead
    if batch_size is None:
        batch_size = 8

    backend_instance = None
    if endpoint_url:
        api_base = endpoint_url
    else:
        # First Party
        from instructlab.model.backends import backends
        from instructlab.model.backends.llama_cpp import Server as llama_cpp_server
        from instructlab.model.backends.vllm import contains_argument

        # TODO (cdoern): we really should not edit the cfg object
        gen_cfg = copy.deepcopy(ctx.obj.config)
        gen_cfg.generate.teacher.llama_cpp.llm_family = (
            model_family or gen_cfg.generate.teacher.llama_cpp.llm_family
        )
        gen_cfg.generate.teacher.vllm.llm_family = (
            model_family or gen_cfg.generate.teacher.vllm.llm_family
        )
        if gpus is not None:
            tps_prefix = "--tensor-parallel-size"
            if contains_argument(tps_prefix, gen_cfg.generate.teacher.vllm.vllm_args):
                click.secho(
                    "Using gpus from --gpus. Ignoring --tensor-parallel-size configured in generate.teacher vllm_args",
                    fg="yellow",
                )
            gen_cfg.generate.teacher.vllm.vllm_args.extend([tps_prefix, str(gpus)])
        backend_instance = backends.select_backend(
            cfg=gen_cfg.generate.teacher, model_path=model_path
        )
        if (
            backend_instance.get_backend_type() is not backends.VLLM
            and gpus is not None
        ):
            logger.debug(
                "Cannot specify '--gpus' with a llama-cpp backend, ignoring this flag."
            )

        try:
            # Run the backend server
            api_base = backend_instance.run_detached(
                http_client(
                    {
                        "tls_client_cert": tls_client_cert,
                        "tls_client_key": tls_client_key,
                        "tls_client_passwd": tls_client_passwd,
                        "tls_insecure": tls_insecure,
                    }
                ),
                background=not enable_serving_output,
            )
        except Exception as exc:
            click.secho(f"Failed to start server: {exc}", fg="red")
            raise click.exceptions.Exit(1)

        # disable batching when running with the local llama.cpp server
        if isinstance(backend_instance, llama_cpp_server):
            if batch_size is not None:
                logger.warning(
                    "Disabling SDG batching - unsupported with llama.cpp serving"
                )
            batch_size = 0

    # Specify checkpoint dir if batching is enabled
    checkpoint_dir = None
    if batch_size > 0:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")

    try:
        click.echo(
            f"Generating synthetic data using '{model_path}' model, taxonomy:'{taxonomy_path}' against {api_base} server"
        )
        generate_data(
            logger=logging.getLogger("instructlab.sdg"),  # TODO: remove
            api_base=api_base,
            api_key=api_key,
            model_family=model_family,
            model_name=model_path,
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
            batch_size=batch_size,
            checkpoint_dir=checkpoint_dir,
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
