# Standard
from unittest import mock
from unittest.mock import patch
import os
import re
import textwrap

# Third Party
from click.testing import CliRunner
from git import Repo

# First Party
from instructlab import lab

# Local
from . import common


def clean_output(text):
    # Remove log entries and ANSI color codes
    text = re.sub(
        r"(ERROR|INFO) \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} [\w.:]+: ", "", text
    )
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)  # Remove ANSI color codes
    # Split by newlines and take the first non-empty line
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines[0] if lines else ""


def gen_qa_pairs(odd):
    i = 1
    qa_pairs = []
    score = 0
    while i < 5:
        if i % 2:
            if odd:
                score = 0.2
            else:
                score = 0.1
        elif not i % 2:
            if odd:
                score = 0.3
            else:
                score = 0.4
        qa_pairs.append(
            {
                "question_id": i,
                "score": score,
                "qna_file": f"category{i}/qna.yaml",
            }
        )
        i = i + 1
    qa_pairs.append(
        {
            "question_id": i,
            "score": 0.5,
            "qna_file": f"category{i}/qna.yaml",
        }
    )
    if odd:
        qa_pairs.append(
            {
                "question_id": i + 1,
                "score": 0.6,
                "qna_file": f"category{i+1}/qna.yaml",
            }
        )
    return qa_pairs


def gen_individual_scores(odd):
    i = 1
    individual_scores = {}
    score = 0
    while i < 5:
        if i % 2:
            if odd:
                score = 0.2
            else:
                score = 0.1
        elif not i % 2:
            if odd:
                score = 0.3
            else:
                score = 0.4
        individual_scores[f"task{i}"] = {"score": score}
        i = i + 1
    individual_scores[f"task{i}"] = {"score": 0.5}
    return individual_scores


def setup_taxonomy(taxonomy_path):
    repo = Repo.init(taxonomy_path, initial_branch="main")
    file_name = "README.md"
    file_path = os.path.join(taxonomy_path, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("#Taxonomy")
    repo.index.add([file_name])
    repo.index.commit("initial commit")


def run_mt_bench(cli_runner, error_rate):
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "evaluate",
            "--benchmark",
            "mt_bench",
            "--model",
            "instructlab/granite-7b-lab",
            "--judge-model",
            "instructlab/merlinite-7b-lab",
        ],
    )
    if result.exit_code != 0:
        print(result.output)
    assert result.exit_code == 0
    expected = textwrap.dedent(
        """\
        Generating answers...
        Evaluating answers...
        # SKILL EVALUATION REPORT

        ## MODEL (SCORE)
        instructlab/granite-7b-lab (1.5/10.0)

        ### TURN ONE (0.0 to 10.0):
        1.0

        ### TURN TWO (0.0 to 10.0):
        2
        """
    )
    if error_rate > 0:
        expected += textwrap.dedent(
            f"""\

            ### ERROR RATE:
            {round(error_rate, 2)}
            """
        )
    expected += "\nᕦ(òᴗóˇ)ᕤ Model evaluate with MTBench completed! ᕦ(òᴗóˇ)ᕤ\n"
    cleaned_result = clean_output(result.output)
    cleaned_expected = clean_output(expected)

    assert cleaned_result == cleaned_expected


def run_mt_bench_branch(cli_runner, error_rate):
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "evaluate",
            "--benchmark",
            "mt_bench_branch",
            "--model",
            "instructlab/granite-7b-lab",
            "--judge-model",
            "instructlab/merlinite-7b-lab",
            "--base-model",
            "instructlab/granite-7b-lab",
            "--branch",
            "rc",
            "--base-branch",
            "main",
            "--taxonomy-path",
            "taxonomy",
        ],
    )
    if result.exit_code != 0:
        print(result.output)
    assert result.exit_code == 0
    expected = textwrap.dedent(
        """\
        Generating questions and reference answers from qna files for branch rc...
        Generating questions and reference answers from qna files for branch main...
        Evaluating answers for branch rc...
        Evaluating answers for branch main...
        # SKILL EVALUATION REPORT

        ## BASE MODEL (SCORE)
        instructlab/granite-7b-lab (0/10.0)

        ## MODEL (SCORE)
        instructlab/granite-7b-lab (0/10.0)

        ### IMPROVEMENTS (0.0 to 10.0):
        1. category1/qna.yaml: 0.1 -> 0.2 (+0.1)
        2. category3/qna.yaml: 0.1 -> 0.2 (+0.1)

        ### REGRESSIONS (0.0 to 10.0):
        1. category2/qna.yaml: 0.4 -> 0.3 (-0.1)
        2. category4/qna.yaml: 0.4 -> 0.3 (-0.1)

        ### NO CHANGE (0.0 to 10.0):
        1. category5/qna.yaml (0.5)

        ### NEW (0.0 to 10.0):
        1. category6/qna.yaml (0.6)
        """
    )
    if error_rate > 0:
        expected += textwrap.dedent(
            f"""\

            ### ERROR RATE:
            {round(error_rate, 2)}
            """
        )
    expected += "\nᕦ(òᴗóˇ)ᕤ Model evaluate with MTBenchBranch completed! ᕦ(òᴗóˇ)ᕤ\n"
    cleaned_result = clean_output(result.output)
    cleaned_expected = clean_output(expected)

    assert cleaned_result == cleaned_expected


@patch("instructlab.model.evaluate.validate_model")
@patch(
    "instructlab.model.evaluate.launch_server",
    return_value=(mock.MagicMock(), "http://127.0.0.1:8000/v1", 1),
)
@patch("instructlab.eval.mt_bench.MTBenchEvaluator.gen_answers")
@patch(
    "instructlab.eval.mt_bench.MTBenchEvaluator.judge_answers",
    side_effect=(
        (1.5001, [{}, {}], [1.002, 2], 0),
        (1.5001, [{}, {}], [1.002, 2], 0.768),
    ),
)
def test_evaluate_mt_bench(
    judge_answers_mock,
    gen_answers_mock,
    launch_server_mock,
    validate_model_mock,
    cli_runner: CliRunner,
):
    run_mt_bench(cli_runner, 0)
    assert validate_model_mock.call_count == 2
    judge_answers_mock.assert_called_once()
    gen_answers_mock.assert_called_once()
    assert launch_server_mock.call_count == 2
    run_mt_bench(cli_runner, 0.768)
    assert validate_model_mock.call_count == 4
    assert judge_answers_mock.call_count == 2
    assert gen_answers_mock.call_count == 2
    assert launch_server_mock.call_count == 4


@patch("instructlab.model.evaluate.validate_model")
@patch(
    "instructlab.model.evaluate.launch_server",
    return_value=(mock.MagicMock(), "http://127.0.0.1:8000/v1", 1),
)
@patch("instructlab.eval.mt_bench.MTBenchBranchEvaluator.gen_answers")
@patch(
    "instructlab.eval.mt_bench.MTBenchBranchEvaluator.judge_answers",
    side_effect=[
        (0, gen_qa_pairs(True), 0),
        (0, gen_qa_pairs(False), 0),
        (0, gen_qa_pairs(True), 0.4567),
        (0, gen_qa_pairs(False), 0.4567),
    ],
)
def test_evaluate_mt_bench_branch(
    judge_answers_mock,
    gen_answers_mock,
    launch_server_mock,
    validate_model_mock,
    cli_runner: CliRunner,
):
    run_mt_bench_branch(cli_runner, 0)
    assert validate_model_mock.call_count == 3
    assert judge_answers_mock.call_count == 2
    assert gen_answers_mock.call_count == 2
    assert launch_server_mock.call_count == 3
    run_mt_bench_branch(cli_runner, 0.4567)
    assert validate_model_mock.call_count == 6
    assert judge_answers_mock.call_count == 4
    assert gen_answers_mock.call_count == 4
    assert launch_server_mock.call_count == 6


@patch("instructlab.model.evaluate.validate_model")
@patch(
    "instructlab.model.evaluate.launch_server",
    return_value=(mock.MagicMock(), "http://127.0.0.1:8000/v1", 1),
)
@patch(
    "instructlab.eval.mmlu.MMLUEvaluator.run",
    return_value=(0.5, {"task1": {"score": 0.1}, "task2": {"score": 0.9}}),
)
def test_evaluate_mmlu(
    run_mock, launch_server_mock, validate_model_mock, cli_runner: CliRunner
):
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "evaluate",
            "--benchmark",
            "mmlu",
            "--model",
            "instructlab/granite-7b-lab",
        ],
    )
    validate_model_mock.assert_called_once()
    launch_server_mock.assert_called_once()
    run_mock.assert_called_once()
    assert result.exit_code == 0
    expected = textwrap.dedent(
        """\
        # KNOWLEDGE EVALUATION REPORT

        ## MODEL (SCORE)
        instructlab/granite-7b-lab (0.5/1.0)

        ### SCORES (0.0 to 1.0):
        task1 - 0.1
        task2 - 0.9
        """
    )
    expected += "\nᕦ(òᴗóˇ)ᕤ Model evaluate with MMLU completed! ᕦ(òᴗóˇ)ᕤ\n"

    cleaned_result = clean_output(result.output)
    cleaned_expected = clean_output(expected)

    assert cleaned_result == cleaned_expected


@patch("instructlab.model.evaluate.validate_model")
@patch(
    "instructlab.model.evaluate.launch_server",
    return_value=(mock.MagicMock(), "http://127.0.0.1:8000/v1", 1),
)
@patch(
    "instructlab.eval.mmlu.MMLUBranchEvaluator.run",
    side_effect=[
        (0.5, gen_individual_scores(True)),
        (0.6, gen_individual_scores(False)),
    ],
)
def test_evaluate_mmlu_branch(
    run_mock, launch_server_mock, validate_model_mock, cli_runner: CliRunner
):
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "evaluate",
            "--benchmark",
            "mmlu_branch",
            "--model",
            "instructlab/granite-7b-lab",
            "--base-model",
            "instructlab/granite-7b-lab",
            "--tasks-dir",
            "generated",
        ],
    )
    assert validate_model_mock.call_count == 2
    assert launch_server_mock.call_count == 2
    assert run_mock.call_count == 2
    assert result.exit_code == 0
    expected = textwrap.dedent(
        """\
        # KNOWLEDGE EVALUATION REPORT

        ## BASE MODEL (SCORE)
        instructlab/granite-7b-lab (0.6/1.0)

        ## MODEL (SCORE)
        instructlab/granite-7b-lab (0.5/1.0)

        ### IMPROVEMENTS (0.0 to 1.0):
        1. task1: 0.1 -> 0.2 (+0.1)
        2. task3: 0.1 -> 0.2 (+0.1)

        ### REGRESSIONS (0.0 to 1.0):
        1. task2: 0.4 -> 0.3 (-0.1)
        2. task4: 0.4 -> 0.3 (-0.1)

        ### NO CHANGE (0.0 to 1.0):
        1. task5 (0.5)
        """
    )
    expected += "\nᕦ(òᴗóˇ)ᕤ Model evaluate with MMLUBranch completed! ᕦ(òᴗóˇ)ᕤ\n"
    cleaned_result = clean_output(result.output)
    cleaned_expected = clean_output(expected)

    assert cleaned_result == cleaned_expected


@patch("instructlab.model.evaluate.validate_model")
def test_no_model_mt_bench(_, cli_runner: CliRunner):
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "evaluate",
            "--benchmark",
            "mt_bench",
        ],
    )
    expected_output = "Benchmark mt_bench requires the following args to be set: ['model', 'judge-model']"
    assert clean_output(result.output) == expected_output


def test_missing_benchmark(cli_runner: CliRunner):
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "evaluate",
        ],
    )
    assert "Missing option '--benchmark'" in result.output
    assert result.exit_code != 0


def test_invalid_model_mt_bench(cli_runner: CliRunner):
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "evaluate",
            "--benchmark",
            "mt_bench",
            "--model",
            "invalid",
            "--judge-model",
            "instructlab/merlinite-7b-lab",
        ],
    )
    assert "Model could not be found at 'invalid'" in result.output
    assert result.exit_code != 0


@patch("instructlab.model.evaluate.validate_model")
def test_invalid_max_workers(_, cli_runner: CliRunner):
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "evaluate",
            "--benchmark",
            "mt_bench",
            "--model",
            "instructlab/granite-7b-lab",
            "--judge-model",
            "instructlab/merlinite-7b-lab",
            "--max-workers",
            "invalid",
        ],
    )
    assert (
        "max-workers must be specified as a positive integer or 'auto'" in result.output
    )
    assert result.exit_code != 0


def test_invalid_model_mmlu(cli_runner: CliRunner):
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "evaluate",
            "--benchmark",
            "mmlu",
            "--model",
            "invalid",
        ],
    )
    assert "Model could not be found at 'invalid'" in result.output
    assert result.exit_code != 0


def test_invalid_gguf_model_mmlu(cli_runner: CliRunner):
    with open("model.gguf", "w", encoding="utf-8") as gguf_file:
        gguf_file.write("")
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "evaluate",
            "--benchmark",
            "mmlu",
            "--model",
            "model.gguf",
        ],
    )
    assert (
        "MMLU and MMLUBranch can currently only be used with a safetensors directory"
        in result.output
    )
    assert result.exit_code != 0


@patch("instructlab.model.evaluate.validate_model")
@patch(
    "instructlab.eval.mmlu.MMLUEvaluator",
    side_effect=Exception("Exiting to check call_args"),
)
def test_int_batchsize_mmlu(mmlu_mock, _, cli_runner: CliRunner):
    cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "evaluate",
            "--benchmark",
            "mmlu",
            "--model",
            "instructlab/granite-7b-lab",
            "--batch-size",
            "1",
        ],
    )
    assert mmlu_mock.call_args_list[0][1]["batch_size"] == 1


@patch("instructlab.model.evaluate.validate_model")
@patch(
    "instructlab.model.evaluate.launch_server",
    return_value=(mock.MagicMock(), "http://127.0.0.1:8000/v1", 1),
)
def test_invalid_taxonomy_mt_bench_branch(launch_server_mock, _, cli_runner: CliRunner):
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "evaluate",
            "--benchmark",
            "mt_bench_branch",
            "--model",
            "instructlab/granite-7b-lab",
            "--judge-model",
            "instructlab/merlinite-7b-lab",
            "--base-model",
            "instructlab/granite-7b-lab",
            "--branch",
            "rc",
            "--base-branch",
            "main",
            "--taxonomy-path",
            "invalid_taxonomy",
        ],
    )
    launch_server_mock.assert_called_once()
    assert "Taxonomy git repo not found" in result.output
    assert result.exit_code != 0


@patch("instructlab.model.evaluate.validate_model")
@patch(
    "instructlab.model.evaluate.launch_server",
    return_value=(mock.MagicMock(), "http://127.0.0.1:8000/v1", 1),
)
def test_invalid_branch_mt_bench_branch(launch_server_mock, _, cli_runner: CliRunner):
    setup_taxonomy("taxonomy")
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "evaluate",
            "--benchmark",
            "mt_bench_branch",
            "--model",
            "instructlab/granite-7b-lab",
            "--judge-model",
            "instructlab/merlinite-7b-lab",
            "--base-model",
            "instructlab/granite-7b-lab",
            "--branch",
            "invalid",
            "--base-branch",
            "main",
            "--taxonomy-path",
            "taxonomy",
        ],
    )
    launch_server_mock.assert_called_once()
    assert "Invalid git branch" in result.output
    assert result.exit_code != 0


@patch("instructlab.model.evaluate.validate_model")
@patch(
    "instructlab.model.evaluate.launch_server",
    return_value=(mock.MagicMock(), "http://127.0.0.1:8000/v1", 1),
)
def test_invalid_tasks_dir(_, __, cli_runner: CliRunner):
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "evaluate",
            "--benchmark",
            "mmlu_branch",
            "--model",
            "instructlab/granite-7b-lab",
            "--base-model",
            "instructlab/granite-7b-lab",
            "--tasks-dir",
            "invalid",
        ],
    )
    assert "Tasks dir not found:" in result.output
    assert result.exit_code != 0


def test_invalid_model_path_mmlu(cli_runner: CliRunner, tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "evaluate",
            "--benchmark",
            "mmlu",
            "--model",
            test_dir,
        ],
    )
    assert (
        "Evaluate '--model' needs to be passed either a safetensors directory or a GGUF file"
        in result.output
    )
    assert result.exit_code != 0


def test_invalid_model_path_mt_bench(cli_runner: CliRunner, tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "evaluate",
            "--benchmark",
            "mt_bench",
            "--model",
            test_dir,
        ],
    )
    assert (
        "Evaluate '--model' needs to be passed either a safetensors directory or a GGUF file"
        in result.output
    )
    assert result.exit_code != 0


@patch("instructlab.model.evaluate.validate_model")
def test_vllm_args_null(_, cli_runner: CliRunner):
    fname = common.setup_gpus_config(section_path="serve", vllm_args=lambda: None)
    args = common.vllm_setup_test(
        cli_runner,
        [
            f"--config={fname}",
            "model",
            "evaluate",
            "--benchmark",
            "mt_bench",
            "--model",
            "instructlab/granite-7b-lab",
            "--judge-model",
            "instructlab/merlinite-7b-lab",
            "--gpus",
            "4",
        ],
    )
    common.assert_tps(args, "4")


@patch("instructlab.model.evaluate.validate_model")
def test_no_gpu_warning(validate_model_mock, cli_runner: CliRunner):
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "evaluate",
            "--benchmark",
            "mmlu",
            "--model",
            "instructlab/granite-7b-lab",
            "--backend",
            "vllm",
        ],
    )
    validate_model_mock.assert_called_once()
    assert (
        "Evaluate is currently not configured to use GPUs. If you are on a GPU-enabled system edit your config or pass the number of GPUs you would like to use with '--gpus'"
        in result.output
    )
