# SPDX-License-Identifier: Apache-2.0

# pylint: disable=ungrouped-imports
# Standard
import contextlib
import enum
import logging
import multiprocessing
import os
import pathlib

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.configuration import _serve
from instructlab.model.backends import backends

# Local
from ..utils import http_client, is_model_gguf, is_model_safetensors

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
    type=click.STRING,
    cls=clickext.ConfigOption,
    config_sections="mt_bench",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    cls=clickext.ConfigOption,
    config_sections="mt_bench",
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
):
    """Evaluates a trained model"""

    # Third Party
    from instructlab.eval.exceptions import EvalError

    with contextlib.suppress(ValueError):
        max_workers = int(max_workers)
    with contextlib.suppress(ValueError):
        batch_size = int(batch_size)

    try:
        # get appropriate evaluator class from Eval lib
        validate_options(
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
        )

        if benchmark == Benchmark.MT_BENCH:
            # Third Party
            from instructlab.eval.mt_bench import MTBenchEvaluator

            model_name = get_model_name(model)
            judge_model_name = get_model_name(judge_model)
            evaluator = MTBenchEvaluator(
                model_name,
                judge_model_name,
                output_dir,
                merge_system_user_message=merge_system_user_message,
            )
            print("Generating answers...")
            server = None
            try:
                server, api_base, effective_gpus = launch_server(
                    eval_serve=ctx.obj.config.serve,
                    tls_client_cert=ctx.params["tls_client_cert"],
                    tls_client_key=ctx.params["tls_client_key"],
                    tls_client_passwd=ctx.params["tls_client_passwd"],
                    tls_insecure=ctx.params["tls_insecure"],
                    model=model,
                    model_name=model_name,
                    max_workers=max_workers,
                    gpus=gpus,
                    backend=backend,
                    enable_serving_output=enable_serving_output,
                )
                evaluator.gen_answers(
                    api_base, max_workers=max_workers, serving_gpus=effective_gpus
                )
            finally:
                if server is not None:
                    server.shutdown()

            print("Evaluating answers...")
            try:
                server, api_base, effective_gpus = launch_server(
                    eval_serve=ctx.obj.config.serve,
                    tls_client_cert=ctx.params["tls_client_cert"],
                    tls_client_key=ctx.params["tls_client_key"],
                    tls_client_passwd=ctx.params["tls_client_passwd"],
                    tls_insecure=ctx.params["tls_insecure"],
                    model=judge_model,
                    model_name=judge_model_name,
                    max_workers=max_workers,
                    gpus=gpus,
                    backend=judge_backend,
                    enable_serving_output=enable_serving_output,
                )
                overall_score, qa_pairs, turn_scores, error_rate = (
                    evaluator.judge_answers(
                        api_base, max_workers=max_workers, serving_gpus=effective_gpus
                    )
                )
            finally:
                if server is not None:
                    server.shutdown()

            max_score = get_benchmark_max_score(Benchmark.MT_BENCH)
            print("# SKILL EVALUATION REPORT")
            print("\n## MODEL (SCORE)")
            display_model(model, overall_score, max_score)
            print(f"\n### TURN ONE (0.0 to {max_score}):")
            print(round(turn_scores[0], 2))
            print(f"\n### TURN TWO (0.0 to {max_score}):")
            turn2_score = turn_scores[1]
            if isinstance(turn2_score, float):
                turn2_score = round(turn2_score, 2)
            print(turn2_score)
            display_error_rate(error_rate)

        elif benchmark == Benchmark.MT_BENCH_BRANCH:
            # Third Party
            from instructlab.eval.mt_bench import MTBenchBranchEvaluator

            model_name = get_model_name(model)
            base_model_name = get_model_name(base_model)
            judge_model_name = get_model_name(judge_model)
            evaluators = [
                MTBenchBranchEvaluator(
                    model_name,
                    judge_model_name,
                    taxonomy_path,
                    branch,
                    output_dir,
                    merge_system_user_message=merge_system_user_message,
                ),
                MTBenchBranchEvaluator(
                    base_model_name,
                    judge_model_name,
                    taxonomy_path,
                    base_branch,
                    output_dir,
                    merge_system_user_message=merge_system_user_message,
                ),
            ]
            branches = [branch, base_branch]
            m_paths = [model, base_model]
            m_names = [model_name, base_model_name]
            qa_pairs_and_errors = []
            server = None

            for i, evaluator in enumerate(evaluators):
                branch = branches[i]
                m_path = m_paths[i]
                m_name = m_names[i]

                print(
                    f"Generating questions and reference answers from qna files for branch {branch}..."
                )
                try:
                    server, api_base, effective_gpus = launch_server(
                        eval_serve=ctx.obj.config.serve,
                        tls_client_cert=ctx.params["tls_client_cert"],
                        tls_client_key=ctx.params["tls_client_key"],
                        tls_client_passwd=ctx.params["tls_client_passwd"],
                        tls_insecure=ctx.params["tls_insecure"],
                        model=m_path,
                        model_name=m_name,
                        max_workers=max_workers,
                        gpus=gpus,
                        backend=backend,
                        enable_serving_output=enable_serving_output,
                    )
                    evaluator.gen_answers(
                        api_base, max_workers=max_workers, serving_gpus=effective_gpus
                    )
                finally:
                    if server is not None:
                        server.shutdown()

            try:
                # Share the judge model server for the two model evaluations
                server, api_base, effective_gpus = launch_server(
                    eval_serve=ctx.obj.config.serve,
                    tls_client_cert=ctx.params["tls_client_cert"],
                    tls_client_key=ctx.params["tls_client_key"],
                    tls_client_passwd=ctx.params["tls_client_passwd"],
                    tls_insecure=ctx.params["tls_insecure"],
                    model=judge_model,
                    model_name=get_model_name(judge_model),
                    max_workers=max_workers,
                    gpus=gpus,
                    backend=judge_backend,
                    enable_serving_output=enable_serving_output,
                )
                for i, evaluator in enumerate(evaluators):
                    branch = branches[i]
                    print(f"Evaluating answers for branch {branch}...")
                    overall_score, qa_pairs, error_rate = evaluator.judge_answers(
                        api_base, max_workers=max_workers, serving_gpus=effective_gpus
                    )
                    qa_pairs_and_errors.append((overall_score, qa_pairs, error_rate))
            finally:
                if server is not None:
                    server.shutdown()

            overall_score, qa_pairs, error_rate = qa_pairs_and_errors[0]
            base_overall_score, base_qa_pairs, base_error_rate = qa_pairs_and_errors[1]

            qna_to_avg_scores = qa_pairs_to_qna_to_avg_scores(qa_pairs)
            base_qna_to_avg_scores = qa_pairs_to_qna_to_avg_scores(base_qa_pairs)

            print("# SKILL EVALUATION REPORT\n")
            display_models_and_scores(
                Benchmark.MT_BENCH_BRANCH,
                model,
                base_model,
                overall_score,
                base_overall_score,
            )

            improvements, regressions, no_changes, new_qnas = [], [], [], []
            for qna, avg_score in qna_to_avg_scores.items():
                base_avg_score = base_qna_to_avg_scores.get(qna)
                if base_avg_score is not None:
                    if avg_score > base_avg_score:
                        improvements.append(
                            (
                                qna,
                                round(avg_score - base_avg_score, 2),
                                base_avg_score,
                                avg_score,
                            )
                        )
                    elif avg_score == base_avg_score:
                        no_changes.append((qna, avg_score))
                    else:
                        regressions.append(
                            (
                                qna,
                                round(avg_score - base_avg_score, 2),
                                base_avg_score,
                                avg_score,
                            )
                        )
                else:
                    new_qnas.append((qna, avg_score))

            # display summary of evaluation before exiting
            display_branch_eval_summary(
                Benchmark.MT_BENCH_BRANCH,
                improvements,
                regressions,
                no_changes,
                new_qnas,
            )
            display_error_rate((error_rate + base_error_rate) / 2)

        elif benchmark == Benchmark.MMLU:
            # Third Party
            from instructlab.eval.mmlu import MMLU_TASKS, MMLUEvaluator

            tasks = MMLU_TASKS
            if os.environ.get("INSTRUCTLAB_EVAL_MMLU_MIN_TASKS") is not None:
                tasks = tasks[:4]

            evaluator = MMLUEvaluator(
                model,
                tasks=tasks,
                few_shots=few_shots,
                batch_size=batch_size,
            )

            server = None
            try:
                server, api_base, _ = launch_server(
                    eval_serve=ctx.obj.config.serve,
                    tls_client_cert=ctx.params["tls_client_cert"],
                    tls_client_key=ctx.params["tls_client_key"],
                    tls_client_passwd=ctx.params["tls_client_passwd"],
                    tls_insecure=ctx.params["tls_insecure"],
                    model=model,
                    model_name=model,
                    max_workers=None,
                    gpus=gpus,
                    backend=backend,
                    enable_serving_output=enable_serving_output,
                )
                overall_score, individual_scores = evaluator.run(api_base)
            finally:
                if server is not None:
                    server.shutdown()

            max_score = get_benchmark_max_score(Benchmark.MMLU)
            print("# KNOWLEDGE EVALUATION REPORT")
            print("\n## MODEL (SCORE)")
            display_model(model, overall_score, max_score)

            print(f"\n### SCORES (0.0 to {max_score}):")
            for task, score in individual_scores.items():
                s = round(score["score"], 2)
                print(f"{task} - {s}")

        elif benchmark == Benchmark.MMLU_BRANCH:
            # Third Party
            from instructlab.eval.mmlu import MMLUBranchEvaluator

            evaluators = [
                MMLUBranchEvaluator(
                    model,
                    tasks_dir,
                    ["mmlu_pr"],
                    few_shots=few_shots,
                    batch_size=batch_size,
                ),
                MMLUBranchEvaluator(
                    base_model,
                    tasks_dir,
                    ["mmlu_pr"],
                    few_shots=few_shots,
                    batch_size=batch_size,
                ),
            ]
            m_paths = [model, base_model]
            overall_scores = []
            individual_scores_list = []
            for i, evaluator in enumerate(evaluators):
                m_path = m_paths[i]
                server = None
                try:
                    server, api_base, _ = launch_server(
                        eval_serve=ctx.obj.config.serve,
                        tls_client_cert=ctx.params["tls_client_cert"],
                        tls_client_key=ctx.params["tls_client_key"],
                        tls_client_passwd=ctx.params["tls_client_passwd"],
                        tls_insecure=ctx.params["tls_insecure"],
                        model=m_path,
                        model_name=m_path,
                        max_workers=None,
                        gpus=gpus,
                        backend=backend,
                        enable_serving_output=enable_serving_output,
                    )
                    overall_score, individual_scores = evaluator.run(api_base)
                    overall_scores.append(overall_score)
                    individual_scores_list.append(individual_scores)
                finally:
                    if server is not None:
                        server.shutdown()

            overall_score = overall_scores[0]
            base_overall_score = overall_scores[1]
            individual_scores = individual_scores_list[0]
            base_individual_scores = individual_scores_list[1]

            print("# KNOWLEDGE EVALUATION REPORT\n")
            display_models_and_scores(
                Benchmark.MMLU_BRANCH,
                model,
                base_model,
                overall_score,
                base_overall_score,
            )

            improvements, regressions, no_changes = [], [], []
            for task, score in individual_scores.items():
                base_score = base_individual_scores[task]
                s = score["score"]
                b_s = base_score["score"]
                d = round(s - b_s, 2)
                if s > b_s:
                    improvements.append((task, d, b_s, s))
                elif b_s > s:
                    regressions.append((task, d, b_s, s))
                else:
                    no_changes.append((task, s))

            # display summary of evaluation before exiting
            display_branch_eval_summary(
                Benchmark.MMLU_BRANCH, improvements, regressions, no_changes
            )
    except EvalError as ee:
        print(ee.message)
        raise click.exceptions.Exit(1)
