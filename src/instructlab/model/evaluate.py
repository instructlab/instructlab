# Standard
import os

# Third Party
from click_didyoumean import DYMGroup
from instructlab.eval.evaluator import Evaluator
from instructlab.eval.mmlu import MMLUBranchEvaluator, MMLUEvaluator
from instructlab.eval.mt_bench import MTBenchBranchEvaluator, MTBenchEvaluator
import click

# First Party
from instructlab import configuration as config

BENCHMARK_TO_CLASS_MAP = {
    "mmlu": MMLUEvaluator,
    "mmlu_branch": MMLUBranchEvaluator,
    "mt_bench": MTBenchEvaluator,
    "mt_bench_branch": MTBenchBranchEvaluator,
}


def get_evaluator(
    model_name,
    benchmark,
    judge_model_name,
    output_dir,
    max_workers,
    taxonomy_path,
    branch,
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
            model_name,
            judge_model_name,
            output_dir,
            max_workers,
        ]
        if benchmark == "mt_bench_branch":
            required_args.append(taxonomy_path)
            required_args.append(branch)
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
                    branch,
                )

    # ensure knowledge benchmarks have proper arguments if selected
    if benchmark in ["mmlu", "mmlu_branch"]:
        required_args = [model_name, few_shots, batch_size]
        if benchmark == "mmlu_branch":
            required_args.append(sdg_path)
        if any(required_args) is None:
            click.secho(
                f"Benchmark {benchmark} requires the following args to be set: {required_args}",
                fg="red",
            )
            raise click.exceptions.Exit(1)
        else:
            evaluator_class = BENCHMARK_TO_CLASS_MAP[benchmark]

            # TODO Make this more robust or wait till we are serving the model from instructlab
            model_dir = os.path.join(os.getcwd(), "models")
            model_path = os.path.join(model_dir, model_name)
            if benchmark == "mmlu":
                min_tasks = os.environ.get("INSTRUCTLAB_EVAL_MMLU_MIN_TASKS")
                if min_tasks is not None:
                    tasks = ["mmlu_abstract_algebra","mmlu_anatomy","mmlu_astronomy"]
                else:
                    tasks = [
                    "mmlu_abstract_algebra",
                    "mmlu_anatomy",
                    "mmlu_astronomy",
                    "mmlu_business_ethics",
                    "mmlu_clinical_knowledge",
                    "mmlu_college_biology",
                    "mmlu_college_chemistry",
                    "mmlu_college_computer_science",
                    "mmlu_college_mathematics",
                    "mmlu_college_medicine",
                    "mmlu_college_physics",
                    "mmlu_computer_security",
                    "mmlu_conceptual_physics",
                    "mmlu_econometrics",
                    "mmlu_electrical_engineering",
                    "mmlu_elementary_mathematics",
                    "mmlu_formal_logic",
                    "mmlu_global_facts",
                    "mmlu_high_school_biology",
                    "mmlu_high_school_chemistry",
                    "mmlu_high_school_computer_science",
                    "mmlu_high_school_european_history",
                    "mmlu_high_school_geography",
                    "mmlu_high_school_government_and_politics",
                    "mmlu_high_school_macroeconomics",
                    "mmlu_high_school_mathematics",
                    "mmlu_high_school_microeconomics",
                    "mmlu_high_school_physics",
                    "mmlu_high_school_psychology",
                    "mmlu_high_school_statistics",
                    "mmlu_high_school_us_history",
                    "mmlu_high_school_world_history",
                    "mmlu_human_aging",
                    "mmlu_human_sexuality",
                    "mmlu_humanities",
                    "mmlu_international_law",
                    "mmlu_jurisprudence",
                    "mmlu_logical_fallacies",
                    "mmlu_machine_learning",
                    "mmlu_management",
                    "mmlu_marketing",
                    "mmlu_medical_genetics",
                    "mmlu_miscellaneous",
                    "mmlu_moral_disputes",
                    "mmlu_moral_scenarios",
                    "mmlu_nutrition",
                    "mmlu_other",
                    "mmlu_philosophy",
                    "mmlu_prehistory",
                    "mmlu_professional_accounting",
                    "mmlu_professional_law",
                    "mmlu_professional_medicine",
                    "mmlu_professional_psychology",
                    "mmlu_public_relations",
                    "mmlu_security_studies",
                    "mmlu_social_sciences",
                    "mmlu_sociology",
                    "mmlu_stem",
                    "mmlu_us_foreign_policy",
                    "mmlu_virology",
                    "mmlu_world_religions",
                ]
                return evaluator_class(
                    model_path, tasks, "float16", few_shots, batch_size
                )
            else:
                return evaluator_class(
                    model_path, sdg_path, ["mmlu_pr"], "float16", few_shots, batch_size
                )


@click.command()
@click.option(
    "--model-name",
    type=click.STRING,
    help="Name of the model to be evaluated",
)
@click.option(
    "--benchmark",
    type=click.Choice(list(BENCHMARK_TO_CLASS_MAP.keys())),
    # case_sensitive=False,
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
    few_shots,
    batch_size,
    sdg_path,
):
    # get appropriate evaluator class from Eval lib
    evaluator = get_evaluator(
        model_name,
        benchmark,
        judge_model_name,
        output_dir,
        max_workers,
        taxonomy_path,
        branch,
        few_shots,
        batch_size,
        sdg_path,
    )

    if benchmark == "mt_bench":
        # TODO: Serve model
        print("Generating answers...")
        evaluator.gen_answers("http://localhost:8000/v1")

        # TODO: Serve judge model
        print("Evaluating answers...")
        overall_score, qa_pairs, turn_scores = evaluator.judge_answers(
            "http://localhost:8000/v1"
        )
        print(f"Overall Score: {overall_score}")
        print(f"Turn 1 Score: {turn_scores[0]}")
        print(f"Turn 2 Score: {turn_scores[1]}")
        print(f"QA Pairs Length: {len(qa_pairs)}")

    elif benchmark == "mt_bench_branch":
        # TODO: Serve model
        # TODO: Should taxonomy dir come from config instead?
        print("Generating questions and reference answers from qna files...")
        evaluator.gen_answers("http://localhost:8000/v1")

        # TODO: Serve judge model
        print("Evaluating answers...")
        qa_pairs = evaluator.judge_answers("http://localhost:8000/v1")
        print(f"qa_pairs length: {len(qa_pairs)}")

    elif benchmark == "mmlu":
        overall_score, individual_scores = evaluator.run()
        print(f"Overall Score: {overall_score}")
        print("Individual Scores:")
        print(individual_scores)
    
    elif benchmark == "mmlu_branch":
        overall_score, individual_scores = evaluator.run()
        print(f"Overall Score: {overall_score}")
        print("Individual Scores:")
        print(individual_scores)
