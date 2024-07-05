# Standard
import enum
import logging
import os
import typing

# Third Party
import click

# First Party
from instructlab import clickext

# Local
from ..utils import display_params, http_client

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
    sdg_path,
) -> "Evaluator":
    """takes in arguments from the CLI and uses 'benchmark' to validate other arguments
    if all needed configuration is present, returns the appropriate Evaluator class for the benchmark
    otherwise raises an exception for the missing values
    """
    # Third Party
    from instructlab.eval.mmlu import MMLUBranchEvaluator, MMLUEvaluator
    from instructlab.eval.mt_bench import MTBenchBranchEvaluator, MTBenchEvaluator

    benchmark_map = {
        Benchmark.MMLU: MMLUEvaluator,
        Benchmark.MMLU_BRANCH: MMLUBranchEvaluator,
        Benchmark.MT_BENCH: MTBenchEvaluator,
        Benchmark.MT_BENCH_BRANCH: MTBenchBranchEvaluator,
    }

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
        evaluator_class = benchmark_map[benchmark]
        if benchmark == Benchmark.MT_BENCH:
            return evaluator_class(
                TEST_MODEL_NAME, JUDGE_MODEL_NAME, output_dir, max_workers
            )
        return evaluator_class(
            TEST_MODEL_NAME,
            JUDGE_MODEL_NAME,
            taxonomy_path,
            branch,
            output_dir,
            max_workers,
        )

    # ensure knowledge benchmarks have proper arguments if selected
    if benchmark in [Benchmark.MMLU, Benchmark.MMLU_BRANCH]:
        required_args = [model, few_shots, batch_size]
        required_arg_names = ["model"]
        if benchmark == Benchmark.MMLU_BRANCH:
            required_args.append(sdg_path)
            required_args.append(base_model)
            required_arg_names.append("sdg-path")
            required_arg_names.append("base-model")
        if None in required_args:
            click.secho(
                f"Benchmark {benchmark} requires the following args to be set: {required_arg_names}",
                fg="red",
            )
            raise click.exceptions.Exit(1)
        evaluator_class = benchmark_map[benchmark]
        if benchmark == Benchmark.MMLU:
            min_tasks = os.environ.get("INSTRUCTLAB_EVAL_MMLU_MIN_TASKS")
            if min_tasks is not None:
                tasks = ["mmlu_abstract_algebra", "mmlu_anatomy", "mmlu_astronomy"]
                evaluator = evaluator_class(
                    model,
                    tasks=tasks,
                    few_shots=few_shots,
                    batch_size=batch_size,
                )
            else:
                evaluator = evaluator_class(
                    model, few_shots=few_shots, batch_size=batch_size
                )
            return evaluator
        return evaluator_class(
            model,
            sdg_path,
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
    ctx,
    model,
    model_name,
    max_workers,
) -> tuple:
    # pylint: disable=import-outside-toplevel
    # First Party
    from instructlab.model.backends import backends

    if not ctx.obj.config.serve.backend:
        ctx.obj.config.serve.backend = backends.VLLM
    if ctx.obj.config.serve.backend == backends.VLLM:
        ctx.obj.config.serve.vllm.vllm_args.extend(["--served-model-name", model_name])
    elif ctx.obj.config.serve.backend == backends.LLAMA_CPP:
        # mt_bench requires a larger context size
        ctx.obj.config.serve.llama_cpp.max_ctx_size = 5120
        # llama-cpp fails fast on too many incoming requests and returns errors to client
        ctx.obj.config.evaluate.mt_bench.max_workers = min(max_workers, 16)

    ctx.obj.config.serve.model_path = model

    backend_instance = backends.select_backend(logger, ctx.obj.config.serve)
    try:
        # http_client is handling tls params
        api_base = backend_instance.run_detached(http_client(ctx.params))
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
    help="Model to be used as a judge for running mt_bench or mt_bench_branch - can be a local path or the name of a Hugging Face repository",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    help="The directory to use for evaluation output from mt_bench or mt_bench_branch",
)
@click.option(
    "--max-workers",
    type=click.INT,
    help="Max parallel workers to run the evaluation with for mt_bench or mt_bench_branch",
)
@click.option(
    "--taxonomy-path",
    type=click.Path(),
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
    help="Number of examples. Needed for running mmlu or mmlu_branch.",
)
@click.option(
    "--batch-size",
    type=click.INT,
    help="Number of GPUs. Needed for running mmlu or mmlu_branch.",
)
@click.option(
    "--sdg-path",
    type=click.Path(),
    help="Path where all the MMLU Branch tasks are stored. Needed for running mmlu_branch.",
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
@click.pass_context
@display_params
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
    sdg_path,
    tls_insecure,  # pylint: disable=unused-argument
    tls_client_cert,  # pylint: disable=unused-argument
    tls_client_key,  # pylint: disable=unused-argument
    tls_client_passwd,  # pylint: disable=unused-argument
):
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
        sdg_path,
    )

    if benchmark == Benchmark.MT_BENCH:
        print("Generating answers...")
        server = None
        try:
            server, api_base = launch_server(
                ctx,
                model,
                TEST_MODEL_NAME,
                max_workers,
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
            )
            # TODO: Change back to standard tuple handling after version bump in eval library
            judgment = evaluator.judge_answers(api_base)
            overall_score = judgment[0]
            qa_pairs = judgment[1]
            turn_scores = judgment[2]
            error_rate = 0
            if len(judgment) > 3:
                error_rate = judgment[3]
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
            )
            for i, evaluator in enumerate(evaluators):
                branch = branches[i]
                print(f"Evaluating answers for branch {branch}...")
                judgment = evaluator.judge_answers(api_base)
                # TODO: Change back to standard tuple handling after version bump in eval library
                error_rate = 0
                if isinstance(judgment, tuple):
                    qa_pairs = judgment[0]
                    error_rate = judgment[1]
                else:
                    qa_pairs = judgment
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
                sdg_path,
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
