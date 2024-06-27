# Standard
import os
import subprocess
import time

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
    model_path,
    base_model_path,
    benchmark,
    judge_model_path,
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
            model_path,
            judge_model_path,
            output_dir,
            max_workers,
        ]
        required_arg_names = [
            "model-path",
            "judge-model-path",
        ]
        if benchmark == "mt_bench_branch":
            required_args.append(taxonomy_path)
            required_args.append(branch)
            required_args.append(base_branch)
            required_args.append(base_model_path)
            required_arg_names.append("branch")
            required_arg_names.append("base-branch")
            required_arg_names.append("base-model-path")
        if None in required_args:
            click.secho(
                f"Benchmark {benchmark} requires the following args to be set: {required_arg_names}",
                fg="red",
            )
            raise click.exceptions.Exit(1)
        else:
            evaluator_class = BENCHMARK_TO_CLASS_MAP[benchmark]
            if benchmark == "mt_bench":
                return evaluator_class(
                    "test_model", "judge_model", output_dir, max_workers
                )
            else:
                return evaluator_class(
                    "test_model",
                    "judge_model",
                    taxonomy_path,
                    branch,
                )

    # ensure knowledge benchmarks have proper arguments if selected
    if benchmark in ["mmlu", "mmlu_branch"]:
        required_args = [model_path, few_shots, batch_size]
        required_arg_names = ["model-path"]
        if benchmark == "mmlu_branch":
            required_args.append(sdg_path)
        if None in required_args:
            click.secho(
                f"Benchmark {benchmark} requires the following args to be set: {required_arg_names}",
                fg="red",
            )
            raise click.exceptions.Exit(1)
        else:
            evaluator_class = BENCHMARK_TO_CLASS_MAP[benchmark]

            if benchmark == "mmlu":
                min_tasks = os.environ.get("INSTRUCTLAB_EVAL_MMLU_MIN_TASKS")
                if min_tasks is not None:
                    tasks = ["mmlu_abstract_algebra", "mmlu_anatomy", "mmlu_astronomy"]
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
    "--model-path",
    type=click.Path(),
    help="Path of the model to be evaluated",
)
@click.option(
    "--base-model-path",
    type=click.Path(),
    help="Path of the model to compare to for mt_bench_branch and mmlu_branch",
)
@click.option(
    "--benchmark",
    type=click.Choice(list(BENCHMARK_TO_CLASS_MAP.keys())),
    # case_sensitive=False,
    help="Benchmarks to run during evaluation",
)
@click.option(
    "--judge-model-path",
    type=click.Path(),
    help="Path of the model to be used as a judge for running mt_bench or mt_bench_branch",
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
    ctx,
    model_path,
    base_model_path,
    benchmark,
    judge_model_path,
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
        model_path,
        base_model_path,
        benchmark,
        judge_model_path,
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
        # TODO: Replace temp Popen hack with serving library calls.  Current library doesn't support server-model-name.
        print("Generating answers...")
        try:
            proc = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--model",
                    model_path,
                    "--tensor-parallel-size",
                    "1",
                    "--served-model-name",
                    "test_model",
                ]
            )
            time.sleep(60)
            evaluator.gen_answers("http://localhost:8000/v1")
        finally:
            proc.terminate()

        print("Evaluating answers...")
        try:
            proc = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--model",
                    judge_model_path,
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
            proc.terminate()
            print(f"Overall Score: {overall_score}")
            print(f"Turn 1 Score: {turn_scores[0]}")
            print(f"Turn 2 Score: {turn_scores[1]}")
            print(f"QA Pairs Length: {len(qa_pairs)}")
        finally:
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
            ),
        ]
        branches = [branch, base_branch]
        m_paths = [model_path, base_model_path]
        m_names = ["test_model", "base_test_model"]
        judge_model_names = ["judge_model", "base_judge_model"]
        qa_pairs_list = []

        for i, evaluator in enumerate(evaluators):
            branch = branches[i]
            m_path = m_paths[i]
            m_name = m_names[i]
            judge_model_name = judge_model_names[i]

            print("Generating questions and reference answers from qna files...")
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

            print("Evaluating answers...")
            try:
                proc = subprocess.Popen(
                    [
                        "python",
                        "-m",
                        "vllm.entrypoints.openai.api_server",
                        "--model",
                        judge_model_path,
                        "--tensor-parallel-size",
                        "1",
                        "--served-model-name",
                        judge_model_name,
                    ]
                )
                time.sleep(60)
                qa_pairs = evaluator.judge_answers("http://localhost:8000/v1")
                print(f"QA Pairs Length: {len(qa_pairs)}")
                qa_pairs_list.append(qa_pairs)
            finally:
                proc.terminate()

            # TODO Compare qa_pairs across run

    elif benchmark == "mmlu":
        overall_score, individual_scores = evaluator.run()
        print(f"Overall Score: {overall_score}")
        print("Individual Scores:")
        print(individual_scores)

    elif benchmark == "mmlu_branch":
        # TODO: This really needs to compare two models and two branches to be useful
        overall_score, individual_scores = evaluator.run()
        print(f"Overall Score: {overall_score}")
        print("Individual Scores:")
        print(individual_scores)
