# Third Party
from click_didyoumean import DYMGroup
from instructlab.eval import (
    MMLU_Evaluator,
    MT_Bench_Evaluator,
    PR_Bench_Evaluator,
    PR_MMLU_Evaluator,
)
import click

benchmark_names_to_classes = {
    "mmlu": MMLU_Evaluator,
    "mt_bench": MT_Bench_Evaluator,
    "pr_bench": PR_Bench_Evaluator,
    "pr_mmlu": PR_MMLU_Evaluator,
}


@click.command(cls=DYMGroup)
@click.option(
    "--benchmark",
    type=click.Choice(list(benchmark_names_to_classes.keys())),
    case_sensitive=False,
    help="Benchmarks to run during evaluation",
)
@click.option(
    "--server-url",
    type=click.STRING,
    help="vLLM or llama-cpp server endpoint. Needed for running mt_bench or pr_bench.",
)
@click.option(
    "--questions",
    type=click.STRING,
    multiple=True,
    help="Questions to be asked. Needed for running pr_bench.",
)
@click.option(
    "--few-shots",
    type=click.INT,
    default=2,
    show_default=True,
    help="Number of examples. Needed for running mmlu or pr_mmlu.",
)
@click.option(
    "--batch-size",
    type=click.INT,
    default=5,
    show_default=True,
    help="Number of GPUs. Needed for running mmlu or pr_mmlu.",
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
    help="Group name that is shared by all the PR MMLU tasks. Needed for running pr_mmlu.",
)
@click.option(
    "--sdg-path",
    type=click.Path(),
    multiple=True,
    help="Path where all the PR MMLU tasks are stored. Needed for running pr_mmlu.",
)
@click.pass_context
def evaluate(
    ctx, benchmark, server_url, questions, tasks, few_shots, batch_size, task, sdg_path
):
    # ensure skills benchmarks have proper arguments if selected
    if benchmark in ["mt_bench", "pr_bench"]:
        required_args = [server_url]
        if benchmark == "pr_bench":
            required_args.append(questions)
        if any(required_args) is None:
            click.secho(
                f"Benchmark {benchmark} requires the following args to be set: {required_args}",
                fg="red",
            )
            raise click.exceptions.Exit(1)
        else:
            evaluator_class = benchmark_names_to_classes[benchmark]
            if benchmark == "mt_bench":
                evaluator = evaluator_class(server_url)
            else:
                evaluator = evaluator_class(server_url, questions)

    # ensure knowledge benchmarks have proper arguments if selected
    if benchmark in ["mmlu", "pr_mmlu"]:
        required_args = [few_shots, batch_size]
        if benchmark == "mmlu":
            required_args.append(tasks)
        if benchmark == "pr_mmlu":
            required_args.extend([task, sdg_path])
        if any(required_args) is None:
            click.secho(
                f"Benchmark {benchmark} requires the following args to be set: {required_args}",
                fg="red",
            )
            raise click.exceptions.Exit(1)
        else:
            evaluator_class = benchmark_names_to_classes[benchmark]
            if benchmark == "mmlu":
                evaluator = evaluator_class(tasks, few_shots, batch_size)
            else:
                evaluator = evaluator_class(sdg_path, task, few_shots, batch_size)

    # execute given evaluator and capture results
    results = evaluator.run()
    print(results)
