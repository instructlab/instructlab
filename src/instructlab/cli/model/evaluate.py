# SPDX-License-Identifier: Apache-2.0

# pylint: disable=ungrouped-imports
# Standard
import contextlib
import logging
import os
import pathlib

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.model.backends import backends
from instructlab.model.evaluate import (
    Benchmark,
    display_branch_eval_summary,
    display_error_rate,
    display_model,
    display_models_and_scores,
    get_benchmark_max_score,
    get_model_name,
    launch_server,
    qa_pairs_to_qna_to_avg_scores,
    validate_options,
)
from instructlab.utils import get_model_arch, get_sysprompt

logger = logging.getLogger(__name__)


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

            # MMLU needs the system prompt to correctly evaluate the model
            model_path = pathlib.Path(model)
            system_prompt = get_sysprompt(get_model_arch(model_path))

            evaluator = MMLUEvaluator(
                model,
                tasks=tasks,
                few_shots=few_shots,
                batch_size=batch_size,
                system_prompt=system_prompt,
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

            # MMLU needs the system prompt to correctly evaluate the model.
            # Accounts for the case when the baseline & model being compared differ in architectures
            model_path, base_model_path = pathlib.Path(model), pathlib.Path(base_model)
            base_model_system_prompt = get_sysprompt(get_model_arch(base_model_path))
            model_system_prompt = get_sysprompt(get_model_arch(model_path))

            evaluators = [
                MMLUBranchEvaluator(
                    model_path=model,
                    tasks_dir=tasks_dir,
                    tasks=["mmlu_pr"],
                    few_shots=few_shots,
                    batch_size=batch_size,
                    system_prompt=model_system_prompt,
                ),
                MMLUBranchEvaluator(
                    model_path=base_model,
                    tasks_dir=tasks_dir,
                    tasks=["mmlu_pr"],
                    few_shots=few_shots,
                    batch_size=batch_size,
                    system_prompt=base_model_system_prompt,
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
        logger.debug("Traceback", exc_info=True)
        raise click.exceptions.Exit(1)
