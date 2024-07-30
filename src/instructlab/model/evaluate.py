# SPDX-License-Identifier: Apache-2.0

# pylint: disable=ungrouped-imports
# Standard
from copy import deepcopy
import enum
import logging
import multiprocessing
import os
import pathlib
import typing

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.model.backends import backends

# Local
from ..utils import http_client

if typing.TYPE_CHECKING:
    # Third Party
    from instructlab.eval.evaluator import Evaluator

logger = logging.getLogger(__name__)


JUDGE_MODEL_NAME = "judge_model"
TEST_MODEL_NAME = "test_model"
BASE_TEST_MODEL_NAME = "base_test_model"


# Python 3.10 does not have StrEnum
class Benchmark(str, enum.Enum):
    MMLU = "mmlu"
    MMLU_BRANCH = "mmlu_branch"
    MT_BENCH = "mt_bench"
    MT_BENCH_BRANCH = "mt_bench_branch"


def get_evaluator(
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
    merge_system_user_message,
) -> "Evaluator":
    """takes in arguments from the CLI and uses 'benchmark' to validate other arguments
    if all needed configuration is present, returns the appropriate Evaluator class for the benchmark
    otherwise raises an exception for the missing values
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
        if os.path.exists(model):
            model = pathlib.Path(model)
            valid_model = False
            if model.is_dir():
                valid_model = backends.is_model_safetensors(model)
            elif model.is_file():
                valid_model = backends.is_model_gguf(model)
            if not valid_model:
                click.secho(
                    "MTBench and MTBenchBranch need to be passed either a safetensors directory or a GGUF file",
                    fg="red",
                )
                raise click.exceptions.Exit(1)
            click.secho(
                f"Using local model found at '{model}' for '--model'",
                fg="blue",
            )
        if benchmark == Benchmark.MT_BENCH:
            # Third Party
            from instructlab.eval.mt_bench import MTBenchEvaluator

            return MTBenchEvaluator(
                TEST_MODEL_NAME,
                JUDGE_MODEL_NAME,
                output_dir,
                max_workers,
                merge_system_user_message=merge_system_user_message,
            )
        # Third Party
        from instructlab.eval.mt_bench import MTBenchBranchEvaluator

        return MTBenchBranchEvaluator(
            TEST_MODEL_NAME,
            JUDGE_MODEL_NAME,
            taxonomy_path,
            branch,
            output_dir,
            max_workers,
            merge_system_user_message=merge_system_user_message,
        )

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
        # ensure user is passing full safetensors if they specify a local directory
        # TODO: also allow GGUF once the following is resolved: https://github.com/instructlab/eval/issues/50
        if os.path.isdir(model):
            if not backends.is_model_safetensors(pathlib.Path(model)):
                click.secho(
                    "MMLU and MMLUBranch can currently only be used with a safetensors directory",
                    fg="red",
                )
                raise click.exceptions.Exit(1)
            click.secho(
                f"Using local safetensors found at '{model}' for '--model'",
                fg="blue",
            )
        else:
            click.secho(
                f"Using safetensors from Hugging Face repo '{model}' for '--model'",
                fg="blue",
            )
        if benchmark == Benchmark.MMLU:
            # Third Party
            from instructlab.eval.mmlu import MMLUEvaluator

            min_tasks = os.environ.get("INSTRUCTLAB_EVAL_MMLU_MIN_TASKS")
            if min_tasks is not None:
                tasks = ["mmlu_abstract_algebra", "mmlu_anatomy", "mmlu_astronomy"]
                evaluator = MMLUEvaluator(
                    model,
                    tasks=tasks,
                    few_shots=few_shots,
                    batch_size=batch_size,
                )
            else:
                evaluator = MMLUEvaluator(
                    model, few_shots=few_shots, batch_size=batch_size
                )
            return evaluator
        # Third Party
        from instructlab.eval.mmlu import MMLUBranchEvaluator

        return MMLUBranchEvaluator(
            model,
            tasks_dir,
            ["mmlu_pr"],
            few_shots=few_shots,
            batch_size=batch_size,
        )


def sort_score(pairing: tuple[str, float]) -> float:
    """helper func for display_branch_eval_summary
    takes a tuple pairing and returns just the score
    """
    return pairing[1]


def display_models(model, base_model) -> None:
    """prints the base_model and model with a header"""
    print("## BASE MODEL")
    print(base_model)
    display_model(model)


def display_model(model) -> None:
    """prints the given model with a header"""
    print("\n## MODEL")
    print(model)


def display_error_rate(error_rate) -> None:
    """prints the error rate with a header"""
    if error_rate > 0:
        print("\n### ERROR RATE:")
        print(round(error_rate, 2))


def display_branch_eval_summary(
    improvements: list[tuple[str, float]],
    regressions: list[tuple[str, float]],
    no_changes: list[str],
    new=None,
):
    """takes in results lists from mt_bench_branch benchmark evaluation
    prints out diff between the branches to the user
    """
    if len(improvements) > 0:
        improvements.sort(key=sort_score, reverse=True)
        print("\n### IMPROVEMENTS:")
        for index, improvement in enumerate(improvements):
            task, delta = improvement
            print(f"{index+1}. {task} (+{delta})")

    if len(regressions) > 0:
        regressions.sort(key=sort_score)
        print("\n### REGRESSIONS:")
        for index, regression in enumerate(regressions):
            task, delta = regression
            print(f"{index+1}. {task} ({delta})")

    if len(no_changes) > 0:
        print("\n### NO CHANGE:")
        for index, task in enumerate(no_changes):
            print(f"{index+1}. {task}")

    if new is not None and len(new) > 0:
        print("\n### NEW:")
        for index, qna in enumerate(new):
            print(f"{index+1}. {qna}")


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


def launch_server(
    ctx: click.Context,
    model: str,
    model_name: str,
    max_workers: int,
    gpus: int,
    backend: str | None,
    enable_serving_output: bool,
) -> tuple:
    eval_serve = deepcopy(ctx.obj.config.serve)
    if backend is None:
        try:
            backend = eval_serve.backend = backends.get(pathlib.Path(model), backend)
        except ValueError as e:
            click.secho(f"Failed to determine backend: {e}", fg="red")
            raise click.exceptions.Exit(1)

    if backend == backends.VLLM:
        eval_serve.vllm.vllm_args.extend(["--served-model-name", model_name])
        # Recommend max-workers based on hardware configuration. #cpus +- 50%
        cpu_count = multiprocessing.cpu_count()
        recommended_min_workers = max(cpu_count // 1.5, 1)
        recommended_max_workers = max(cpu_count // 0.5, 1)
        if (
            max_workers > recommended_max_workers
            or max_workers < recommended_min_workers
        ):
            logger.warning(
                f"Based on your hardware configuration, when using vLLM, we recommend setting max-workers between {recommended_min_workers} and {recommended_max_workers} for optimal performance"
            )
        if gpus:
            # Warn when overriding from vllm_args
            tps_prefix = "--tensor-parallel-size"
            if any(
                s == tps_prefix or s.startswith(tps_prefix + "=")
                for s in eval_serve.vllm.vllm_args
            ):
                # Either tps_prefix value or tps_prefix=value
                click.secho(
                    "Using gpus from --gpus or evaluate config and ignoring --tensor-parallel-size configured in serve vllm_args",
                    fg="yellow",
                )
            eval_serve.vllm.vllm_args.extend([tps_prefix, str(gpus)])
    elif backend == backends.LLAMA_CPP:
        if ctx.obj.config.serve.llama_cpp.max_ctx_size < 5120:
            eval_serve.llama_cpp.max_ctx_size = 5120
            logger.debug(
                "Evaluate requires a context size of >= 5120, ignoring serve configuration for max_ctx_size"
            )
        # llama-cpp fails fast on too many incoming requests and returns errors to client
        recommended_workers = max(multiprocessing.cpu_count() // 2, 1)
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
            http_client(ctx.params), background=not enable_serving_output
        )
    except Exception as exc:
        click.secho(f"Failed to start server: {exc}", fg="red")
        raise click.exceptions.Exit(1)
    return backend_instance, api_base


@click.command()
@click.option(
    "--model",
    type=click.STRING,
    cls=clickext.ConfigOption,
    help="Model to be evaluated - can be a local path or the name of a Hugging Face repository",
)
@click.option(
    "--base-model",
    type=click.STRING,
    cls=clickext.ConfigOption,
    required=True,  # default from config
    help="Base model to compare with 'model' for mt_bench_branch and mmlu_branch - can be a local path or the name of a Hugging Face repository",
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
    help="Model to be used as a judge for running mt_bench or mt_bench_branch - can be a local path or the name of a Hugging Face repository",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    cls=clickext.ConfigOption,
    config_sections="mt_bench",
    help="The directory to use for evaluation output from mt_bench or mt_bench_branch",
)
@click.option(
    "--max-workers",
    type=click.INT,
    cls=clickext.ConfigOption,
    config_sections="mt_bench",
    help="Max parallel workers to run the evaluation with for mt_bench or mt_bench_branch",
)
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    cls=clickext.ConfigOption,
    config_sections="mt_bench_branch",
    help="Taxonomy git repo path for running mt_bench_branch",
)
@click.option(
    "--branch",
    type=click.STRING,
    cls=clickext.ConfigOption,
    help="Branch of taxonomy repo to eval QNAs against model",
)
@click.option(
    "--base-branch",
    type=click.STRING,
    cls=clickext.ConfigOption,
    help="Base branch of taxonomy repo to eval QNAs against model for mt_bench_branch",
)
@click.option(
    "--few-shots",
    type=click.INT,
    cls=clickext.ConfigOption,
    config_sections="mmlu",
    help="Number of examples. Needed for running mmlu or mmlu_branch.",
)
@click.option(
    "--batch-size",
    type=click.STRING,
    cls=clickext.ConfigOption,
    config_sections="mmlu",
    help="Batch size for mmlu and mmlu_branch evaluation. Valid values are a positive integer, 'auto' to select the largest batch size that will fit in memory, or 'auto:N' to reselect the largest batch size N times'.",
)
@click.option(
    "--tasks-dir",
    type=click.Path(),
    cls=clickext.ConfigOption,
    config_sections="mmlu_branch",
    help="Path where all the MMLU Branch tasks are stored. Needed for running mmlu_branch.",
)
@click.option(
    "--gpus",
    type=click.INT,
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
    tls_insecure,  # pylint: disable=unused-argument
    tls_client_cert,  # pylint: disable=unused-argument
    tls_client_key,  # pylint: disable=unused-argument
    tls_client_passwd,  # pylint: disable=unused-argument
    enable_serving_output,
):
    """Evaluates a trained model"""
    # get appropriate evaluator class from Eval lib
    evaluator = get_evaluator(
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
        merge_system_user_message,
    )

    # Third Party
    from instructlab.eval.exceptions import EvalError

    try:
        if benchmark == Benchmark.MT_BENCH:
            print("Generating answers...")
            server = None
            try:
                server, api_base = launch_server(
                    ctx,
                    model,
                    TEST_MODEL_NAME,
                    max_workers,
                    gpus,
                    backend,
                    enable_serving_output,
                )
                evaluator.gen_answers(api_base)
            finally:
                if server is not None:
                    server.shutdown()

            print("Evaluating answers...")
            try:
                server, api_base = launch_server(
                    ctx,
                    judge_model,
                    JUDGE_MODEL_NAME,
                    max_workers,
                    gpus,
                    judge_backend,
                    enable_serving_output,
                )
                overall_score, qa_pairs, turn_scores, error_rate = (
                    evaluator.judge_answers(api_base)
                )
            finally:
                if server is not None:
                    server.shutdown()

            print("# SKILL EVALUATION REPORT")
            display_model(model)
            print("\n### AVERAGE:")
            print(f"{round(overall_score, 2)} (across {len(qa_pairs)})")
            print("\n### TURN ONE:")
            print(round(turn_scores[0], 2))
            print("\n### TURN TWO:")
            turn2_score = turn_scores[1]
            if isinstance(turn2_score, float):
                turn2_score = round(turn2_score, 2)
            print(turn2_score)
            display_error_rate(error_rate)

        elif benchmark == Benchmark.MT_BENCH_BRANCH:
            # Third Party
            from instructlab.eval.mt_bench import MTBenchBranchEvaluator

            evaluators = [
                evaluator,
                MTBenchBranchEvaluator(
                    BASE_TEST_MODEL_NAME,
                    JUDGE_MODEL_NAME,
                    taxonomy_path,
                    base_branch,
                    output_dir,
                    max_workers,
                    merge_system_user_message=merge_system_user_message,
                ),
            ]
            branches = [branch, base_branch]
            m_paths = [model, base_model]
            m_names = [TEST_MODEL_NAME, BASE_TEST_MODEL_NAME]
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
                    server, api_base = launch_server(
                        ctx,
                        m_path,
                        m_name,
                        max_workers,
                        gpus,
                        backend,
                        enable_serving_output,
                    )
                    evaluator.gen_answers(api_base)
                finally:
                    if server is not None:
                        server.shutdown()

            try:
                # Share the judge model server for the two model evaluations
                server, api_base = launch_server(
                    ctx,
                    judge_model,
                    JUDGE_MODEL_NAME,
                    max_workers,
                    gpus,
                    judge_backend,
                    enable_serving_output,
                )
                for i, evaluator in enumerate(evaluators):
                    branch = branches[i]
                    print(f"Evaluating answers for branch {branch}...")
                    qa_pairs, error_rate = evaluator.judge_answers(api_base)
                    qa_pairs_and_errors.append((qa_pairs, error_rate))
            finally:
                if server is not None:
                    server.shutdown()

            qa_pairs, error_rate = qa_pairs_and_errors[0]
            base_qa_pairs, base_error_rate = qa_pairs_and_errors[1]

            qna_to_avg_scores = qa_pairs_to_qna_to_avg_scores(qa_pairs)
            base_qna_to_avg_scores = qa_pairs_to_qna_to_avg_scores(base_qa_pairs)

            print("# SKILL EVALUATION REPORT\n")
            display_models(model, base_model)

            improvements, regressions, no_changes, new_qnas = [], [], [], []
            for qna, avg_score in qna_to_avg_scores.items():
                base_avg_score = base_qna_to_avg_scores.get(qna)
                if base_avg_score is not None:
                    if avg_score > base_avg_score:
                        improvements.append((qna, round(avg_score - base_avg_score, 2)))
                    elif avg_score == base_avg_score:
                        no_changes.append(qna)
                    else:
                        regressions.append((qna, round(avg_score - base_avg_score, 2)))
                else:
                    new_qnas.append((qna))

            # display summary of evaluation before exiting
            display_branch_eval_summary(improvements, regressions, no_changes, new_qnas)
            display_error_rate((error_rate + base_error_rate) / 2)

        elif benchmark == Benchmark.MMLU:
            overall_score, individual_scores = evaluator.run()

            print("# KNOWLEDGE EVALUATION REPORT")
            display_model(model)
            print("\n### AVERAGE:")
            print(f"{round(overall_score, 2)} (across {len(individual_scores)})\n")

            print("### SCORES:")
            for task, score in individual_scores.items():
                s = round(score["score"], 2)
                print(f"{task} - {s}")

        elif benchmark == Benchmark.MMLU_BRANCH:
            # Third Party
            from instructlab.eval.mmlu import MMLUBranchEvaluator

            evaluators = [
                evaluator,
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
            for evaluator in evaluators:
                overall_score, individual_scores = evaluator.run()
                overall_scores.append(overall_score)
                individual_scores_list.append(individual_scores)

            overall_score = overall_scores[0]
            base_overall_score = overall_scores[1]
            individual_scores = individual_scores_list[0]
            base_individual_scores = individual_scores_list[1]

            print("# KNOWLEDGE EVALUATION REPORT\n")
            display_models(model, base_model)

            print("\n### AVERAGE:")
            delta = round(overall_score - base_overall_score, 2)
            if delta >= 0:
                delta_display = f"+{delta}"
            else:
                delta_display = delta

            print(f"{delta_display} (across {len(individual_scores)})")

            improvements, regressions, no_changes = [], [], []
            for task, score in individual_scores.items():
                base_score = base_individual_scores[task]
                s = score["score"]
                b_s = base_score["score"]
                d = round(s - b_s, 2)
                if s > b_s:
                    improvements.append((task, d))
                elif b_s > s:
                    regressions.append((task, d))
                else:
                    no_changes.append(task)

            # display summary of evaluation before exiting
            display_branch_eval_summary(improvements, regressions, no_changes)
    except EvalError as ee:
        print(ee.message)
        raise click.exceptions.Exit(1)
