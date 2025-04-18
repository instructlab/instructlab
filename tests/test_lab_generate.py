# Standard
from pathlib import Path
from unittest.mock import patch

# Third Party
from click.testing import CliRunner

# First Party
from instructlab import lab

# Local
from . import common


def test_vllm_args_null(cli_runner: CliRunner):
    fname = common.setup_gpus_config(
        section_path="generate.teacher", vllm_args=lambda: None
    )
    args = common.vllm_setup_test(
        cli_runner,
        [
            f"--config={fname}",
            "data",
            "generate",
            "--gpus",
            "4",
        ],
    )
    common.assert_tps(args, "4")


@patch("instructlab.cli.data.generate.gen_data")
@patch("instructlab.cli.data.data.storage_dirs_exist", return_value=True)
def test_generate_with_teacher_model_id(
    mock_gen_data, _, cli_runner: CliRunner, tmp_path: Path
):
    fname = common.setup_test_models_config(
        models_list=[
            {
                "id": "teacher_model",
                "path": "teacher/model/path",
                "family": "llama",
                "system_prompt": "system prompt",
            }
        ],
        dest=tmp_path,
    )

    result = cli_runner.invoke(
        lab.ilab,
        [
            f"--config={tmp_path}/{fname}",
            "data",
            "generate",
            "--teacher-model-id",
            "teacher_model",
            "--pipeline",
            "simple",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, f"Command failed with output: {result.output}"
    assert mock_gen_data.call_count == 1


@patch("instructlab.cli.data.generate.gen_data")
@patch("instructlab.cli.data.data.storage_dirs_exist", return_value=True)
def test_generate_with_global_teacher_model_id(
    mock_gen_data, _, cli_runner: CliRunner, tmp_path: Path
):
    fname = common.setup_test_models_config(
        models_list=[
            {
                "id": "teacher_model",
                "path": "teacher/model/path",
                "family": "llama",
                "system_prompt": "system prompt",
            }
        ],
        dest=tmp_path,
        global_teacher_id="teacher_model",
    )

    result = cli_runner.invoke(
        lab.ilab,
        [
            f"--config={tmp_path}/{fname}",
            "data",
            "generate",
            "--pipeline",
            "simple",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, f"Command failed with output: {result.output}"
    assert mock_gen_data.call_count == 1
