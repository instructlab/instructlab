# SPDX-License-Identifier: Apache-2.0

# pylint: disable=ungrouped-imports
# Standard
import logging

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.model.backends import backends
from instructlab.model.evaluate import Benchmark, evaluate_model

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model",
    type=click.STRING,
    cls=clickext.ConfigOption,
)
@click.option(
    "--base-model",
    type=click.STRING,
    cls=clickext.ConfigOption,
    required=True,  # default from config
)
@click.option(
    "--benchmark",
    type=click.Choice([m.value for m in Benchmark.__members__.values()]),
    required=True,
    help="Benchmarks to run during evaluation",
)
@click.option(
    "--judge-model",
    default=None,
    type=click.STRING,
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(),
)
@click.option(
    "--max-workers",
    type=click.STRING,
    cls=clickext.ConfigOption,
    config_sections="mt_bench",
)
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    cls=clickext.ConfigOption,
    config_sections="mt_bench_branch",
)
@click.option(
    "--branch",
    type=click.STRING,
    cls=clickext.ConfigOption,
)
@click.option(
    "--base-branch",
    type=click.STRING,
    cls=clickext.ConfigOption,
)
@click.option(
    "--few-shots",
    type=click.INT,
    cls=clickext.ConfigOption,
    config_sections="mmlu",
)
@click.option(
    "--batch-size",
    type=click.STRING,
    cls=clickext.ConfigOption,
    config_sections="mmlu",
)
@click.option(
    "--tasks-dir",
    type=click.Path(),
    cls=clickext.ConfigOption,
    config_sections="mmlu_branch",
)
@click.option(
    "--gpus",
    type=click.IntRange(min=0),
    help="Number of GPUs to utilize for evaluation (not applicable to llama-cpp)",
)
@click.option(
    "--merge-system-user-message",
    is_flag=True,
    help="Indicates whether to merge system and user message for mt_bench and mt_bench_branch (required for Mistral based judges)",
)
@click.option(
    "--backend",
    type=click.Choice(tuple(backends.SUPPORTED_BACKENDS)),
    help="Serving backend to use for the model and base model (if applicable) during evaluation. Options are vllm and llama-cpp.",
)
@click.option(
    "--judge-backend",
    type=click.Choice(tuple(backends.SUPPORTED_BACKENDS)),
    help="Serving backend to use for the judge model for during mt_bench or mt_bench_branch evaluation. Options are vllm and llama-cpp.",
)
@click.option(
    "--tls-insecure",
    is_flag=True,
    help="Disable TLS verification for model serving.",
)
@click.option(
    "--tls-client-cert",
    type=click.Path(),
    default="",
    show_default=True,
    help="Path to the TLS client certificate to use for model serving.",
)
@click.option(
    "--tls-client-key",
    type=click.Path(),
    default="",
    show_default=True,
    help="Path to the TLS client key to use for model serving.",
)
@click.option(
    "--tls-client-passwd",
    type=click.STRING,
    default="",
    help="TLS client certificate password for model serving.",
)
@click.option(
    "--enable-serving-output",
    is_flag=True,
    help="Print serving engine logs.",
)
@click.option(
    "--skip-server",
    is_flag=True,
    help="Skip launching the server and evaluate directly with the HuggingFace model. This option supports mmlu and mmlu_branch benchmarks.",
)
@click.option(
    "--input-questions",
    type=click.STRING,
    cls=clickext.ConfigOption,
    config_sections="dk_bench",
)
@click.option(
    "--output-file-formats",
    type=click.STRING,
    cls=clickext.ConfigOption,
    config_sections="dk_bench",
)
@click.option(
    "--system-prompt",
    type=click.STRING,
    cls=clickext.ConfigOption,
)
@click.option(
    "--temperature",
    type=click.FloatRange(min=0.0, max=1.0),
    cls=clickext.ConfigOption,
)
@click.pass_context
@clickext.display_params
def evaluate(
    ctx,
    model,
    base_model,
    benchmark,
    judge_model,
    output_dir,
    max_workers: str | int,
    taxonomy_path,
    branch,
    base_branch,
    few_shots,
    batch_size: str | int,
    tasks_dir,
    gpus,
    merge_system_user_message,
    backend,
    judge_backend,
    tls_insecure,  # pylint: disable=unused-argument
    tls_client_cert,  # pylint: disable=unused-argument
    tls_client_key,  # pylint: disable=unused-argument
    tls_client_passwd,  # pylint: disable=unused-argument
    enable_serving_output,
    skip_server: bool,
    input_questions,
    output_file_formats,
    system_prompt,
    temperature,
) -> None:
    """Evaluates a trained model"""
    try:
        evaluate_model(
            ctx,
            model,
            base_model,
            benchmark,
            judge_model,
            output_dir,
            max_workers,
            taxonomy_path,
            branch,
            base_branch,
            few_shots,
            batch_size,
            tasks_dir,
            gpus,
            merge_system_user_message,
            backend,
            judge_backend,
            tls_insecure,
            tls_client_cert,
            tls_client_key,
            tls_client_passwd,
            enable_serving_output,
            skip_server,
            input_questions,
            output_file_formats,
            system_prompt,
            temperature,
        )
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")
        raise click.exceptions.Exit(1) from e
