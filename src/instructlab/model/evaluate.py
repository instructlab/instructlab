# SPDX-License-Identifier: Apache-2.0

# pylint: disable=ungrouped-imports
# Standard
import enum
import logging
import multiprocessing
import os
import pathlib

# Third Party
import click

# First Party
from instructlab.configuration import _serve
from instructlab.model.backends import backends

# Local
from ..client_utils import http_client
from ..utils import is_model_gguf, is_model_safetensors

logger = logging.getLogger(__name__)


# Python 3.10 does not have StrEnum
class Benchmark(str, enum.Enum):
    MMLU = "mmlu"
    MMLU_BRANCH = "mmlu_branch"
    MT_BENCH = "mt_bench"
    MT_BENCH_BRANCH = "mt_bench_branch"


def validate_options(
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
):
    """takes in arguments from the CLI and uses 'benchmark' to validate other arguments
    if all needed configuration is present, raises an exception for the missing values
    """

    # ensure skills benchmarks have proper arguments if selected
    if benchmark in {Benchmark.MT_BENCH, Benchmark.MT_BENCH_BRANCH}:
        required_args = [
            model,
            judge_model,
            output_dir,
            max_workers,
        ]
        required_arg_names = [
            "model",
            "judge-model",
        ]

        if benchmark == Benchmark.MT_BENCH_BRANCH:
            required_args.append(taxonomy_path)
            required_args.append(branch)
            required_args.append(base_branch)
            required_args.append(base_model)
            required_arg_names.append("taxonomy-path")
            required_arg_names.append("branch")
            required_arg_names.append("base-branch")
            required_arg_names.append("base-model")
        if None in required_args:
            click.secho(
                f"Benchmark {benchmark} requires the following args to be set: {required_arg_names}",
                fg="red",
            )
            raise click.exceptions.Exit(1)

        validate_model(model)
        validate_model(judge_model, "--judge-model")
        if benchmark == Benchmark.MT_BENCH_BRANCH:
            validate_model(base_model, "--base-model")

        if (isinstance(max_workers, str) and max_workers != "auto") or (
            isinstance(max_workers, int) and max_workers < 1
        ):
            click.secho(
                "max-workers must be specified as a positive integer or 'auto'",
                fg="red",
            )
            raise click.exceptions.Exit(1)

    # ensure knowledge benchmarks have proper arguments if selected
    if benchmark in [Benchmark.MMLU, Benchmark.MMLU_BRANCH]:
        required_args = [model, few_shots, batch_size]
        required_arg_names = ["model"]
        if benchmark == Benchmark.MMLU_BRANCH:
            required_args.append(tasks_dir)
            required_args.append(base_model)
            required_arg_names.append("tasks-dir")
            required_arg_names.append("base-model")
        if None in required_args:
            click.secho(
                f"Benchmark {benchmark} requires the following args to be set: {required_arg_names}",
                fg="red",
            )
            raise click.exceptions.Exit(1)

        validate_model(model, allow_gguf=False)
        if benchmark == Benchmark.MMLU_BRANCH:
            validate_model(base_model, "--base-model", allow_gguf=False)


def validate_model(model: str, model_arg: str = "--model", allow_gguf: bool = True):
    if os.path.exists(model):
        model_path = pathlib.Path(model)
        valid_model = False
        if model_path.is_dir():
            valid_model = is_model_safetensors(model_path)
        elif model_path.is_file():
            if allow_gguf:
                valid_model = is_model_gguf(model_path)
            else:
                click.secho(
                    "MMLU and MMLUBranch can currently only be used with a safetensors directory",
                    fg="red",
                )
                raise click.exceptions.Exit(1)
        if not valid_model:
            click.secho(
                f"Evaluate '{model_arg}' needs to be passed either a safetensors directory or a GGUF file",
                fg="red",
            )
            raise click.exceptions.Exit(1)
        click.secho(
            f"Using local model found at '{model_path}' for '{model_arg}'",
            fg="blue",
        )
    else:
        click.secho(
            f"Model could not be found at '{model}' for '{model_arg}'",
            fg="red",
        )
        raise click.exceptions.Exit(1)


def sort_score(pairing: tuple[str, float, float, float]) -> float:
    """helper func for display_branch_eval_summary
    takes a tuple pairing and returns just the score
    """
    return pairing[1]


def get_benchmark_max_score(benchmark: Benchmark) -> str:
    # total score for Benchmark.MT_BENCH_BRANCH or Benchmark.MT_Bench
    max_score = "10.0"
    if benchmark in (Benchmark.MMLU_BRANCH, Benchmark.MMLU):
        max_score = "1.0"
    return max_score


def display_models_and_scores(
    benchmark, model, base_model, model_score, base_model_score
) -> None:
    """prints the base_model and model with a header"""
    max_score = get_benchmark_max_score(benchmark)

    base_model_score = round(base_model_score, 2)
    model_score = round(model_score, 2)
    print("## BASE MODEL (SCORE)")
    display_model(base_model, base_model_score, max_score)
    print("\n## MODEL (SCORE)")
    display_model(model, model_score, max_score)


def display_model(model, model_score, max_score) -> None:
    """prints the given model with a header"""
    model_score = round(model_score, 2)
    print(f"{model} ({model_score}/{max_score})")


def display_error_rate(error_rate) -> None:
    """prints the error rate with a header"""
    if error_rate > 0:
        print("\n### ERROR RATE:")
        print(round(error_rate, 2))


def display_branch_eval_summary(
    benchmark: Benchmark,
    improvements: list[tuple[str, float, float, float]],
    regressions: list[tuple[str, float, float, float]],
    no_changes: list[tuple[str, float]],
    new=None,
):
    """takes in results lists from mt_bench_branch benchmark evaluation
    prints out diff between the branches to the user
    """
    # total score for MT-BENCH-BRANCH
    max_score = get_benchmark_max_score(benchmark)

    if len(improvements) > 0:
        improvements.sort(key=sort_score, reverse=True)
        print(f"\n### IMPROVEMENTS (0.0 to {max_score}):")
        for index, improvement in enumerate(improvements):
            task, delta, base_score, new_score = improvement
            base_score = round(base_score, 2)
            new_score = round(new_score, 2)
            print(f"{index+1}. {task}: {base_score} -> {new_score} (+{delta})")

    if len(regressions) > 0:
        regressions.sort(key=sort_score)
        print(f"\n### REGRESSIONS (0.0 to {max_score}):")
        for index, regression in enumerate(regressions):
            task, delta, base_score, new_score = regression
            base_score = round(base_score, 2)
            new_score = round(new_score, 2)
            print(f"{index+1}. {task}: {base_score} -> {new_score} ({delta})")

    if len(no_changes) > 0:
        print(f"\n### NO CHANGE (0.0 to {max_score}):")
        for index, entry in enumerate(no_changes):
            task, avg_score = entry
            avg_score = round(avg_score, 2)
            print(f"{index+1}. {task} ({avg_score})")

    if new is not None and len(new) > 0:
        print(f"\n### NEW (0.0 to {max_score}):")
        for index, entry in enumerate(new):
            qna, avg_score = entry
            avg_score = round(avg_score, 2)
            print(f"{index+1}. {qna} ({avg_score})")


def qa_pairs_to_qna_to_avg_scores(qa_pairs: list[dict]) -> dict[str, float]:
    """takes in a list of qa_pair dicts
    returns a dict of average scores per qna file
    """
    qna_to_scores: dict[str, list[float]] = {}
    for qa_pair in qa_pairs:
        qna_file = qa_pair["qna_file"]
        score = qa_pair["score"]
        scores = qna_to_scores.get(qna_file)
        if scores is None:
            qna_to_scores[qna_file] = [score]
        else:
            scores.append(score)
    qna_to_avg_scores = {}
    for qna, scores in qna_to_scores.items():
        qna_to_avg_scores[qna] = sum(scores) / len(scores)
    return qna_to_avg_scores


def get_model_name(model_path):
    return os.path.basename(os.path.normpath(model_path))


def get_cpu_count():
    """Returns the available cpu count to this process"""
    try:
        # Not available on all platforms
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        return multiprocessing.cpu_count()


def get_gpus(eval_serve, gpus=None) -> tuple[int | None, int]:
    """Return the number of gpus explicitly selected through --gpus or config
    The second value in the tuple is the effective gpus that will be used by
    serving. If gpus is specified, the two values will be the same. 0 is the min
    value for effective_gpus.
    """
    # First Party
    from instructlab.model.backends.vllm import get_argument

    gpus = gpus or eval_serve.vllm.gpus

    effective_gpus = gpus
    if effective_gpus is None:
        try:
            tps = get_argument("--tensor-parallel-size", eval_serve.vllm.vllm_args)
            if tps is not None:
                effective_gpus = int(tps)
        except ValueError:
            logger.warning("Invalid --tensor-parallel-size found in serve vllm_args")
    effective_gpus = effective_gpus or 0
    return gpus, effective_gpus


def get_backend(backend, model):
    """Return the backend based on specified backend and model detection"""
    if backend is None:
        try:
            return backends.get(pathlib.Path(model), backend)
        except ValueError as e:
            click.secho(f"Failed to determine backend: {e}", fg="red")
            raise click.exceptions.Exit(1)
    return backend


def launch_server(
    eval_serve: _serve,
    tls_client_cert: str | None,
    tls_client_key: str | None,
    tls_client_passwd: str | None,
    tls_insecure: bool,
    model: str,
    model_name: str,
    max_workers: str | int | None,
    gpus: int | None,
    backend: str | None,
    enable_serving_output: bool,
) -> tuple:
    # eval_serve = deepcopy(ctx.obj.config.serve)
    eval_serve.backend = backend = get_backend(backend, model)

    effective_gpus = 0
    if backend == backends.VLLM:
        eval_serve.vllm.vllm_args = eval_serve.vllm.vllm_args or []
        eval_serve.vllm.vllm_args.extend(["--served-model-name", model_name])

        # First Party
        from instructlab.model.backends.vllm import contains_argument

        gpus, effective_gpus = get_gpus(eval_serve, gpus)
        if gpus:
            tps_prefix = "--tensor-parallel-size"
            if contains_argument(tps_prefix, eval_serve.vllm.vllm_args):
                click.secho(
                    "Using gpus from --gpus or config and ignoring --tensor-parallel-size configured in serve vllm_args",
                    fg="yellow",
                )
            eval_serve.vllm.vllm_args.extend([tps_prefix, str(gpus)])
        elif effective_gpus < 1:
            click.secho(
                "Evaluate is currently not configured to use GPUs. If you are on a GPU-enabled system edit your config or pass the number of GPUs you would like to use with '--gpus'",
                fg="yellow",
            )

        if max_workers is not None and isinstance(max_workers, int):
            # Recommend max-workers based on hardware configuration: min(#GPUs being used * 10, #CPU cores) +- 50%
            # Edge cases:
            # - Many GPUs, not many CPUs: Unlikely, workers might not be able to keep the GPUs busy but recommendation can be ignored.
            # - Many CPUs, not many GPUs: More likely, 10 workers per GPU should still be reasonable.
            target_max_workers = min(max(effective_gpus, 1) * 10, get_cpu_count())
            recommended_min_workers = max(target_max_workers // 2, 1)
            recommended_max_workers = max(int(target_max_workers // 0.5), 1)
            if (
                max_workers > recommended_max_workers
                or max_workers < recommended_min_workers
            ):
                logger.warning(
                    f"Based on your hardware configuration, when using vLLM, we recommend setting max-workers between {recommended_min_workers} and {recommended_max_workers} for optimal performance"
                )
    elif backend == backends.LLAMA_CPP:
        if eval_serve.llama_cpp.max_ctx_size < 5120:
            eval_serve.llama_cpp.max_ctx_size = 5120
            logger.debug(
                "Evaluate requires a context size of >= 5120, ignoring serve configuration for max_ctx_size"
            )
        if max_workers is not None and isinstance(max_workers, int):
            # llama-cpp fails fast on too many incoming requests and returns errors to client
            recommended_workers = max(get_cpu_count() // 2, 1)
            if max_workers > recommended_workers:
                logger.warning(
                    f"Based on your hardware configuration, when using llama-cpp, we recommend setting max-workers to a maximum of {recommended_workers}"
                )
        if gpus:
            logger.debug("Ignoring --gpus option for llama-cpp serving")

    eval_serve.model_path = model

    backend_instance = backends.select_backend(eval_serve, backend)
    try:
        # http_client is handling tls params
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
            foreground_allowed=True,
            max_startup_retries=1,
        )
    except Exception as exc:
        click.secho(f"Failed to start server: {exc}", fg="red")
        raise click.exceptions.Exit(1)
    return backend_instance, api_base, effective_gpus
