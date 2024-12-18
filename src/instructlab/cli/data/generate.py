# SPDX-License-Identifier: Apache-2.0

# Standard
import copy
import logging
import os
import os.path
import pathlib

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.client_utils import HttpClientParams
from instructlab.configuration import DEFAULTS
from instructlab.data.generate_data import gen_data  # type: ignore
from instructlab.defaults import ILAB_PROCESS_MODES
from instructlab.utils import (
    contains_argument,
    get_model_arch,
    get_sysprompt,
    use_legacy_pretraining_format,
)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model",
    "model_path",
    cls=clickext.ConfigOption,
    config_sections="teacher",
    show_default=False,
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
    cls=clickext.ConfigOption,
)
@click.option(
    "--taxonomy-base",
    cls=clickext.ConfigOption,
)
@click.option(
    "--output-dir",
    type=click.Path(),
    cls=clickext.ConfigOption,
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
    type=click.STRING,
    help="Force model family to use when picking a generation template",
)
@click.option(
    "--pipeline",
    type=click.STRING,
    cls=clickext.ConfigOption,
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
    config_sections="teacher.vllm",
)
@click.option(
    "--max-num-tokens",
    type=click.IntRange(min=512),
    cls=clickext.ConfigOption,
)
@click.option(
    "-dt", "--detached", is_flag=True, help="Run ilab data generate in the background"
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
    max_num_tokens,
    detached,
):
    """Generates synthetic data to enhance your example data"""

    # if --pipeline is not used, pipeline defaults to the value of ctx.obj.config.generate.pipeline
    # set in the config file. A user could intentionally set this to 'null' in the config file
    # if they want to ensure --pipeline needs to be used.
    # This would happen if the type of pipeline needs to be different across different runs of
    # `ilab data generate`.
    if not pipeline:
        click.secho(
            "Pipeline not set. Please use the --pipeline flag or set it in the config file.",
            fg="red",
        )
        raise click.exceptions.Exit(1)

    if num_instructions != -1:
        click.secho(
            "The --num-instructions flag is deprecated. Please use --sdg-scale-factor instead.",
            fg="yellow",
        )

    # If batch size is not set explicitly, default to 8
    # Once https://github.com/instructlab/sdg/issues/224 is resolved we can
    # pass batch_size=None to the library instead
    if batch_size is None:
        batch_size = 8

    # Specify checkpoint dir if batching is enabled
    checkpoint_dir = None
    if batch_size > 0:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")

    serve_cfg = copy.deepcopy(ctx.obj.config.generate.teacher)
    serve_cfg.llama_cpp.llm_family = model_family or serve_cfg.llama_cpp.llm_family
    serve_cfg.vllm.llm_family = model_family or serve_cfg.vllm.llm_family
    serve_cfg.vllm.vllm_args = serve_cfg.vllm.vllm_args or []
    if gpus is not None:
        tps_prefix = "--tensor-parallel-size"
        if contains_argument(tps_prefix, serve_cfg.vllm.vllm_args):
            click.secho(
                "Using gpus from --gpus. Ignoring --tensor-parallel-size configured in generate.teacher vllm_args",
                fg="yellow",
            )
        serve_cfg.vllm.vllm_args.extend([tps_prefix, str(gpus)])

    http_client_params = HttpClientParams(
        {
            "tls_client_cert": tls_client_cert,
            "tls_client_key": tls_client_key,
            "tls_client_passwd": tls_client_passwd,
            "tls_insecure": tls_insecure,
        }
    )

    # determine student model arch from train section of config and pick system prompt to
    # pass to SDG appropriately
    student_model_path = pathlib.Path(ctx.obj.config.train.model_path)
    student_model_arch = get_model_arch(student_model_path)
    system_prompt = get_sysprompt(student_model_arch)

    # Check if student model specifies a tokenizer config. If so, check if the special tokens specified
    # match those of granite-7b. If so, set legacy_pretraining_format to true. If no tokenizer config is
    # available, rely on model architecture to make that decision
    if ctx.obj.config.general.use_legacy_tmpl:
        legacy_pretraining_format = True
    else:
        legacy_pretraining_format = use_legacy_pretraining_format(
            student_model_path, student_model_arch
        )

    process_mode = ILAB_PROCESS_MODES.ATTACHED
    if detached:
        process_mode = ILAB_PROCESS_MODES.DETACHED

    try:
        gen_data(
            serve_cfg,
            model_path,
            num_cpus,
            sdg_scale_factor,
            taxonomy_path,
            taxonomy_base,
            output_dir,
            quiet,
            endpoint_url,
            api_key,
            yaml_rules,
            chunk_word_count,
            server_ctx_size,
            http_client_params,
            model_family,
            pipeline,
            enable_serving_output,
            batch_size,
            gpus,
            checkpoint_dir,
            max_num_tokens,
            system_prompt,
            legacy_pretraining_format,
            process_mode=process_mode,
            log_level=ctx.obj.config.general.log_level,
        )
    except Exception as exc:
        click.secho(f"failed to generate data with exception: {exc}", fg="red")
        raise click.exceptions.Exit(1)
