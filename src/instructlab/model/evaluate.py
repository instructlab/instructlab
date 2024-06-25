# Third Party
from click_didyoumean import DYMGroup
from instructlab.eval import (
    Evaluator,
    MMLUBranchEvaluator,
    MMLUEvaluator,
    MTBenchBranchEvaluator,
    MTBenchEvaluator,
)
import click

# First Party
from instructlab import configuration as config

BENCHMARK_TO_CLASS_MAP = frozenset(
    {
        "mmlu": MMLUEvaluator,
        "mmlu_branch": MMLUBranchEvaluator,
        "mt_bench": MTBenchEvaluator,
        "mt_bench_branch": MTBenchBranchEvaluator,
    }
)


def get_evaluator(
    model_name,
    benchmark,
    judge_model_name,
    output_dir,
    max_workers,
    taxonomy_path,
    branch,
    tasks,
    few_shots,
    batch_size,
    task,
    sdg_path,
) -> Evaluator:
    """takes in arguments from the CLI and uses 'benchmark' to validate other arguments
    if all needed configuration is present, returns the appropriate Evaluator class for the benchmark
    otherwise raises an exception for the missing values
    """

    # ensure skills benchmarks have proper arguments if selected
    if benchmark in ["mt_bench", "mt_bench_branch"]:
        required_args = [
            model_name,
            judge_model_name,
            output_dir,
            max_workers,
        ]
        if benchmark == "mt_bench_branch":
            required_args.append(taxonomy_path, branch)
        if any(required_args) is None:
            click.secho(
                f"Benchmark {benchmark} requires the following args to be set: {required_args}",
                fg="red",
            )
            raise click.exceptions.Exit(1)
        else:
            evaluator_class = BENCHMARK_TO_CLASS_MAP[benchmark]
            if benchmark == "mt_bench":
                return evaluator_class(
                    model_name, judge_model_name, output_dir, max_workers
                )
            else:
                return evaluator_class(
                    model_name,
                    judge_model_name,
                    taxonomy_path,
                )

    # ensure knowledge benchmarks have proper arguments if selected
    if benchmark in ["mmlu", "mmlu_branch"]:
        required_args = [few_shots, batch_size]
        if benchmark == "mmlu":
            required_args.append(tasks)
        elif benchmark == "mmlu_branch":
            required_args.extend([task, sdg_path])
        if any(required_args) is None:
            click.secho(
                f"Benchmark {benchmark} requires the following args to be set: {required_args}",
                fg="red",
            )
            raise click.exceptions.Exit(1)
        else:
            evaluator_class = BENCHMARK_TO_CLASS_MAP[benchmark]
            if benchmark == "mmlu":
                return evaluator_class(tasks, few_shots, batch_size)
            else:
                return evaluator_class(sdg_path, task, few_shots, batch_size)


@click.command(cls=DYMGroup)
@click.option(
    "--model-name",
    type=click.STRING,
    help="Name of the model to be evaluated",
)
@click.option(
    "--benchmark",
    type=click.Choice(list(BENCHMARK_TO_CLASS_MAP.keys())),
    case_sensitive=False,
    help="Benchmarks to run during evaluation",
)
@click.option(
    "--judge-model-name",
    type=click.STRING,
    help="Name of the model to be used as a judge for running mt_bench or mt_bench_branch",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=config.DEFAULT_EVAL_PATH,
    help="The directory to use for evaluation output from mt_bench or mt_bench_branch",
)
@click.option(
    "--max-workers",
    type=click.INT,
    default=40,
    show_default=True,
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
    "--few-shots",
    type=click.INT,
    default=2,
    show_default=True,
    help="Number of examples. Needed for running mmlu or mmlu_branch.",
)
@click.option(
    "--batch-size",
    type=click.INT,
    default=5,
    show_default=True,
    help="Number of GPUs. Needed for running mmlu or mmlu_branch.",
)
@click.option(
    "--tasks",
    type=click.STRING,
    multiple=True,
    help="List of tasks for mmlu to test the model with. Needed for running mmlu.",
)
@click.option(
    "--task",
    type=click.STRING,
    multiple=True,
    help="Group name that is shared by all the MMLU Branch tasks. Needed for running mmlu_branch.",
)
@click.option(
    "--sdg-path",
    type=click.Path(),
    multiple=True,
    help="Path where all the MMLU Branch tasks are stored. Needed for running mmlu_branch.",
)
@click.pass_context
def evaluate(
    ctx,
    model_name,
    benchmark,
    judge_model_name,
    output_dir,
    max_workers,
    taxonomy_path,
    branch,
    tasks,
    few_shots,
    batch_size,
    task,
    sdg_path,
):
    # get appropriate evalautor class from Eval lib
    evaluator = get_evaluator(
        model_name,
        benchmark,
        judge_model_name,
        output_dir,
        max_workers,
        taxonomy_path,
        branch,
        tasks,
        few_shots,
        batch_size,
        task,
        sdg_path,
    )

    # execute given evaluator and capture results
    results = evaluator.run()
    print(results)
