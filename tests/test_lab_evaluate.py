# Standard
from unittest import mock
from unittest.mock import patch
import os
import textwrap

# Third Party
from click.testing import CliRunner
from git import Repo

# First Party
from instructlab import lab


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
    assert result.exit_code == 0
    expected = textwrap.dedent(
        """\
        Generating answers...
        Evaluating answers...
        # SKILL EVALUATION REPORT

        ## MODEL
        instructlab/granite-7b-lab

        ### AVERAGE:
        1.5 (across 2)

        ### TURN ONE:
        1.0

        ### TURN TWO:
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
    assert result.output == expected


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
    assert result.exit_code == 0
    expected = textwrap.dedent(
        """\
        Generating questions and reference answers from qna files for branch rc...
        Generating questions and reference answers from qna files for branch main...
        Evaluating answers for branch rc...
        Evaluating answers for branch main...
        # SKILL EVALUATION REPORT

        ## BASE MODEL
        instructlab/granite-7b-lab

        ## MODEL
        instructlab/granite-7b-lab

        ### IMPROVEMENTS:
        1. category1/qna.yaml (+0.1)
        2. category3/qna.yaml (+0.1)

        ### REGRESSIONS:
        1. category2/qna.yaml (-0.1)
        2. category4/qna.yaml (-0.1)

        ### NO CHANGE:
        1. category5/qna.yaml

        ### NEW:
        1. category6/qna.yaml
        """
    )
    if error_rate > 0:
        expected += textwrap.dedent(
            f"""\

            ### ERROR RATE:
            {round(error_rate, 2)}
            """
        )
    assert result.output == expected


@patch(
    "instructlab.model.evaluate.launch_server",
    return_value=(mock.MagicMock(), "http://127.0.0.1:8000/v1"),
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
    cli_runner: CliRunner,
):
    run_mt_bench(cli_runner, 0)
    judge_answers_mock.assert_called_once()
    gen_answers_mock.assert_called_once()
    assert launch_server_mock.call_count == 2
    run_mt_bench(cli_runner, 0.768)
    assert judge_answers_mock.call_count == 2
    assert gen_answers_mock.call_count == 2
    assert launch_server_mock.call_count == 4


@patch(
    "instructlab.model.evaluate.launch_server",
    return_value=(mock.MagicMock(), "http://127.0.0.1:8000/v1"),
)
@patch("instructlab.eval.mt_bench.MTBenchBranchEvaluator.gen_answers")
@patch(
    "instructlab.eval.mt_bench.MTBenchBranchEvaluator.judge_answers",
    side_effect=[
        (gen_qa_pairs(True), 0),
        (gen_qa_pairs(False), 0),
        (gen_qa_pairs(True), 0.4567),
        (gen_qa_pairs(False), 0.4567),
    ],
)
def test_evaluate_mt_bench_branch(
    judge_answers_mock,
    gen_answers_mock,
    launch_server_mock,
    cli_runner: CliRunner,
):
    run_mt_bench_branch(cli_runner, 0)
    assert judge_answers_mock.call_count == 2
    assert gen_answers_mock.call_count == 2
    assert launch_server_mock.call_count == 3
    run_mt_bench_branch(cli_runner, 0.4567)
    assert judge_answers_mock.call_count == 4
    assert gen_answers_mock.call_count == 4
    assert launch_server_mock.call_count == 6


@patch(
    "instructlab.eval.mmlu.MMLUEvaluator.run",
    return_value=(0.5, {"task1": {"score": 0.1}, "task2": {"score": 0.9}}),
)
def test_evaluate_mmlu(run_mock, cli_runner: CliRunner):
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
    run_mock.assert_called_once()
    assert result.exit_code == 0
    expected = textwrap.dedent(
        """\
        Using safetensors from Hugging Face repo 'instructlab/granite-7b-lab' for '--model'
        # KNOWLEDGE EVALUATION REPORT

        ## MODEL
        instructlab/granite-7b-lab

        ### AVERAGE:
        0.5 (across 2)

        ### SCORES:
        task1 - 0.1
        task2 - 0.9
        """
    )
    assert result.output == expected


@patch(
    "instructlab.eval.mmlu.MMLUBranchEvaluator.run",
    side_effect=[
        (0.5, gen_individual_scores(True)),
        (0.6, gen_individual_scores(False)),
    ],
)
def test_evaluate_mmlu_branch(run_mock, cli_runner: CliRunner):
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
    assert run_mock.call_count == 2
    assert result.exit_code == 0
    expected = textwrap.dedent(
        """\
        Using safetensors from Hugging Face repo 'instructlab/granite-7b-lab' for '--model'
        # KNOWLEDGE EVALUATION REPORT

        ## BASE MODEL
        instructlab/granite-7b-lab

        ## MODEL
        instructlab/granite-7b-lab

        ### AVERAGE:
        -0.1 (across 5)

        ### IMPROVEMENTS:
        1. task1 (+0.1)
        2. task3 (+0.1)

        ### REGRESSIONS:
        1. task2 (-0.1)
        2. task4 (-0.1)

        ### NO CHANGE:
        1. task5
        """
    )
    assert result.output == expected


def test_no_model_mt_bench(cli_runner: CliRunner):
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
    assert (
        result.output
        == "Benchmark mt_bench requires the following args to be set: ['model', 'judge-model']\n"
    )
    assert result.exit_code != 0


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
    assert "Failed to determine backend:" in result.output
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
    assert "Model could not be found" in result.output
    assert result.exit_code != 0


@patch(
    "instructlab.model.evaluate.launch_server",
    return_value=(mock.MagicMock(), "http://127.0.0.1:8000/v1"),
)
def test_invalid_taxonomy_mt_bench_branch(launch_server_mock, cli_runner: CliRunner):
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


@patch(
    "instructlab.model.evaluate.launch_server",
    return_value=(mock.MagicMock(), "http://127.0.0.1:8000/v1"),
)
def test_invalid_branch_mt_bench_branch(launch_server_mock, cli_runner: CliRunner):
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


def test_invalid_tasks_dir(cli_runner: CliRunner):
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
        "MMLU and MMLUBranch can currently only be used with a safetensors directory"
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
        "MTBench and MTBenchBranch need to be passed either a safetensors directory or a GGUF file"
        in result.output
    )
    assert result.exit_code != 0
