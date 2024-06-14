import click
from click_didyoumean import DYMGroup
from instructlab_eval import eval


benchmark_names_to_classes = {
    "mmlu": eval.mmlu,
    "mt": eval.mt
}

@click.command(cls=DYMGroup)
@click.option(
    "--benchmarks", 
    type={}, 
    help="benchmarks to run during evaluation",
)
@click.pass_context
def evaluate(ctx, benchmarks):
    # do eval here

    for bench in benchmarks:
        if bench in benchmark_names_to_classes:
            # run the proper bench
            benchmark_names_to_classes[bench].run()
