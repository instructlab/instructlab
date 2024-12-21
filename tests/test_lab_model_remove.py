# Standard
from pathlib import Path

# Third Party
from click.testing import CliRunner

# First Party
from instructlab import lab
from tests.common import create_gguf_file, create_safetensors_model_directory


def test_remove_dir_model(cli_runner: CliRunner, tmp_path):
    temp_model_dir = tmp_path / "test-models"
    temp_model_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(temp_model_dir)
    create_safetensors_model_directory(model_dir)
    model_dir_name = "test_namespace/testlab_model"
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "remove",
            "--model",
            model_dir_name,
            "--force",
            "--model-dir",
            temp_model_dir,
        ],
    )
    assert "Model test_namespace/testlab_model has been removed." in result.output
    assert result.exit_code == 0

    list_result = cli_runner.invoke(
        lab.ilab,
        ["--config=DEFAULT", "model", "list", "--model-dirs", temp_model_dir],
    )
    assert model_dir_name not in list_result.output
    assert list_result.exit_code == 0


def test_remove_gguf_model(cli_runner: CliRunner, tmp_path):
    temp_model_dir = tmp_path / "test-models"
    temp_model_dir.mkdir(parents=True, exist_ok=True)
    gguf_file_path = Path(temp_model_dir)
    create_gguf_file(gguf_file_path)
    gguf_file_name = "test-model.gguf"
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "remove",
            "--model",
            gguf_file_name,
            "--model-dir",
            temp_model_dir,
        ],
        input="yes\n",
    )
    assert "Model test-model.gguf has been removed." in result.output
    assert result.exit_code == 0

    list_result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "list",
            "--model-dirs",
            temp_model_dir,
        ],
    )
    assert gguf_file_name not in list_result.output
    assert list_result.exit_code == 0


def test_remove_not_exist_model(cli_runner: CliRunner, tmp_path):
    temp_model_dir = tmp_path / "test-models"
    temp_model_dir.mkdir(parents=True, exist_ok=True)
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "remove",
            "--model",
            "test-not-exist",
            "--model-dir",
            temp_model_dir,
            "--force",
        ],
    )
    assert "Model test-not-exist does not exist" in result.output
    assert result.exit_code != 0
