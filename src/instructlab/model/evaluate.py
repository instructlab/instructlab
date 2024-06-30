# Standard
import os
import subprocess
import time

# Third Party
from instructlab.eval.evaluator import Evaluator
from instructlab.eval.mmlu import MMLUBranchEvaluator, MMLUEvaluator
from instructlab.eval.mt_bench import MTBenchBranchEvaluator, MTBenchEvaluator
import click

BENCHMARK_TO_CLASS_MAP = {
    "mmlu": MMLUEvaluator,
    "mmlu_branch": MMLUBranchEvaluator,
    "mt_bench": MTBenchEvaluator,
    "mt_bench_branch": MTBenchBranchEvaluator,
}

# TODO: Remove after proper vLLM integration
# pylint: disable=consider-using-with


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
) -> Evaluator:
    """takes in arguments from the CLI and uses 'benchmark' to validate other arguments
    if all needed configuration is present, returns the appropriate Evaluator class for the benchmark
    otherwise raises an exception for the missing values
    """

    # ensure skills benchmarks have proper arguments if selected
    if benchmark in ["mt_bench", "mt_bench_branch"]:
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
        if benchmark == "mt_bench_branch":
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
        evaluator_class = BENCHMARK_TO_CLASS_MAP[benchmark]
        if benchmark == "mt_bench":
            return evaluator_class("test_model", "judge_model", output_dir, max_workers)
        return evaluator_class(
            "test_model",
            "judge_model",
            taxonomy_path,
            branch,
            output_dir,
            max_workers,
        )

    # ensure knowledge benchmarks have proper arguments if selected
    if benchmark in ["mmlu", "mmlu_branch"]:
        required_args = [model, few_shots, batch_size]
        required_arg_names = ["model"]
        if benchmark == "mmlu_branch":
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
        evaluator_class = BENCHMARK_TO_CLASS_MAP[benchmark]
        if benchmark == "mmlu":
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


def sort_score(pairing: tuple) -> float:
    """helper func for display_branch_eval_summary
    takes a tuple pairing and returns just the score
    """
    _, score = pairing
    return score


def display_models(model, base_model) -> None:
    """prints the base_model and model with a header"""
    print("## BASE MODEL")
    print(base_model)
    display_model(model)


def display_model(model) -> None:
    """helper func for display_models
    prints the given model with a header
    """
    print("\n## MODEL")
    print(model)


def display_branch_eval_summary(
    improvements: list, regressions: list, no_changes: list, new=None
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


def qa_pairs_to_qna_to_avg_scores(qa_pairs: list[dict]) -> dict:
    """takes in a list of qa_pair dicts
    returns a dict of average scores per qna file
    """
    qna_to_scores = {}
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


@click.command()
@click.option(
    "--model",
    type=click.STRING,
    help="Model to be evaluated - can be a local path or the name of a Hugging Face repository",
)
@click.option(
    "--base-model",
    type=click.STRING,
    help="Base model to compare with 'model' for mt_bench_branch and mmlu_branch - can be a local path or the name of a Hugging Face repository",
)
@click.option(
    "--benchmark",
    type=click.Choice(list(BENCHMARK_TO_CLASS_MAP.keys())),
    # case_sensitive=False,
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
    help="Branch of taxonomy repo to eval QNAs against model",
)
@click.option(
    "--base-branch",
    type=click.STRING,
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
@click.pass_context
def evaluate(
    ctx,  # pylint: disable=unused-argument
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

    if benchmark == "mt_bench":
        # TODO: Replace temp Popen hack with serving library calls. Current library doesn't support server-model-name.
        print("Generating answers...")
        proc = None
        try:
            proc = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--model",
                    model,
                    "--tensor-parallel-size",
                    "1",
                    "--served-model-name",
                    "test_model",
                ]
            )
            time.sleep(60)
            evaluator.gen_answers("http://localhost:8000/v1")
        finally:
            if proc:
                proc.terminate()

        print("Evaluating answers...")
        proc = None
        try:
            proc = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--model",
                    judge_model,
                    "--tensor-parallel-size",
                    "1",
                    "--served-model-name",
                    "judge_model",
                ]
            )
            time.sleep(60)
            overall_score, qa_pairs, turn_scores = evaluator.judge_answers(
                "http://localhost:8000/v1"
            )
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
        finally:
            if proc:
                proc.terminate()

    elif benchmark == "mt_bench_branch":
        # TODO: Replace temp Popen hack with serving library calls.  Current library doesn't support server-model-name.
        evaluators = [
            evaluator,
            MTBenchBranchEvaluator(
                "base_test_model",
                "base_judge_model",
                taxonomy_path,
                base_branch,
                output_dir,
                max_workers,
            ),
        ]
        branches = [branch, base_branch]
        m_paths = [model, base_model]
        m_names = ["test_model", "base_test_model"]
        judge_model_names = ["judge_model", "base_judge_model"]
        qa_pairs_list = []

        for i, evaluator in enumerate(evaluators):
            branch = branches[i]
            m_path = m_paths[i]
            m_name = m_names[i]
            j_model_name = judge_model_names[i]

            print(
                f"Generating questions and reference answers from qna files for branch {branch}..."
            )
            try:
                proc = subprocess.Popen(
                    [
                        "python",
                        "-m",
                        "vllm.entrypoints.openai.api_server",
                        "--model",
                        m_path,
                        "--tensor-parallel-size",
                        "1",
                        "--served-model-name",
                        m_name,
                    ]
                )
                time.sleep(60)
                evaluator.gen_answers("http://localhost:8000/v1")
            finally:
                proc.terminate()

            print(f"Evaluating answers for branch {branch}...")
            try:
                proc = subprocess.Popen(
                    [
                        "python",
                        "-m",
                        "vllm.entrypoints.openai.api_server",
                        "--model",
                        judge_model,
                        "--tensor-parallel-size",
                        "1",
                        "--served-model-name",
                        j_model_name,
                    ]
                )
                time.sleep(60)
                qa_pairs = evaluator.judge_answers("http://localhost:8000/v1")
                qa_pairs_list.append(qa_pairs)
            finally:
                proc.terminate()

        qa_pairs = qa_pairs_list[0]
        base_qa_pairs = qa_pairs_list[1]

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

    elif benchmark == "mmlu":
        overall_score, individual_scores = evaluator.run()

        print("# KNOWLEDGE EVALUATION REPORT")
        display_model(model)
        print("\n### AVERAGE:")
        print(f"{round(overall_score, 2)} (across {len(individual_scores)})\n")

        print("### SCORES:")
        for task, score in individual_scores.items():
            s = round(score["score"], 2)
            print(f"{task} - {s}")

    elif benchmark == "mmlu_branch":
        # TODO: Need to resolve the appropriate task list for the branches involved
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
        for i, evaluator in enumerate(evaluators):
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
