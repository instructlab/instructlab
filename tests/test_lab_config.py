# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import pathlib

# Third Party
from click.testing import CliRunner
from ruamel.yaml import YAML
import ruamel.yaml
import yaml

# First Party
from instructlab import configuration, lab


def remove_comment_indentation(yaml_output: str) -> str:
    return "\n".join(
        line.lstrip() if line.lstrip().startswith("#") else line
        for line in yaml_output.splitlines()
    )


def strip_comments(yaml_output: str) -> str:
    return "\n".join(
        line for line in yaml_output.splitlines() if not line.lstrip().startswith("#")
    )


def get_expected_yaml_output() -> str:
    cfg = configuration.get_default_config()
    general_config_commented_map = configuration.config_to_commented_map(cfg.general)

    yaml_stream = ruamel.yaml.compat.StringIO()
    YAML().dump(general_config_commented_map, yaml_stream)
    return yaml_stream.getvalue()


def test_ilab_config_show(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(lab.ilab, ["--config", "DEFAULT", "config", "show"])
    assert result.exit_code == 0, result.stdout

    parsed = yaml.safe_load(result.stdout_bytes)
    assert parsed

    assert configuration.Config(**parsed)


def test_ilab_config_show_key_general_yaml(cli_runner: CliRunner) -> None:
    expected_yaml_output = get_expected_yaml_output()

    # Test --key output for general in YAML
    result_key_general_yaml = cli_runner.invoke(
        lab.ilab,
        ["--config", "DEFAULT", "config", "show", "--key", "general"],
    )

    assert result_key_general_yaml.exit_code == 0
    actual_output = remove_comment_indentation(result_key_general_yaml.stdout)
    expected_yaml_output = remove_comment_indentation(expected_yaml_output)
    assert actual_output == expected_yaml_output


def test_ilab_config_show_without_comments(cli_runner: CliRunner) -> None:
    expected_yaml_output = get_expected_yaml_output()

    # Test --key output for general in YAML with --remove-comments
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config",
            "DEFAULT",
            "config",
            "show",
            "--key",
            "general",
            "--without-comments",
        ],
    )

    assert result.exit_code == 0
    actual_output = result.stdout
    expected_output_without_comments = strip_comments(expected_yaml_output)
    assert actual_output.strip() == expected_output_without_comments.strip()


def test_ilab_config_init_with_env_var_config(
    cli_runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    # Common setup code
    cfg = configuration.get_default_config()
    assert cfg.general.log_level == logging.getLevelName(logging.INFO)
    cfg.general.log_level = logging.getLevelName(logging.DEBUG)
    cfg_file = tmp_path / "config-gold.yaml"
    with cfg_file.open("w") as f:
        yaml.dump(cfg.model_dump(), f)

    # Invoke the CLI command
    command = ["config", "init", "--non-interactive"]
    result = cli_runner.invoke(
        lab.ilab, command, env={"ILAB_GLOBAL_CONFIG": cfg_file.as_posix()}
    )
    assert result.exit_code == 0, result.stdout

    # Load and check the generated config file
    config_path = pathlib.Path(configuration.DEFAULTS.CONFIG_FILE)
    assert config_path.exists()
    with config_path.open(encoding="utf-8") as f:
        parsed = yaml.safe_load(f)
    assert parsed
    assert configuration.Config(**parsed).general.log_level == logging.getLevelName(
        logging.DEBUG
    )


def test_ilab_config_init_with_model_path(cli_runner: CliRunner) -> None:
    # Common setup code
    model_path = "path/to/model"
    command = ["config", "init", "--model-path", model_path]

    # Invoke the CLI command
    result = cli_runner.invoke(lab.ilab, command)
    assert result.exit_code == 0, result.stdout

    # Load and check the generated config file
    config_path = pathlib.Path(configuration.DEFAULTS.CONFIG_FILE)
    assert config_path.exists()
    with config_path.open(encoding="utf-8") as f:
        parsed = yaml.safe_load(f)
    assert parsed
    # the generate config should NOT use the same model as the chat/serve model
    assert configuration.Config(**parsed).generate.model != "path/to/model"
    assert configuration.Config(**parsed).generate.teacher.model_path != "path/to/model"
    assert configuration.Config(**parsed).serve.model_path == "path/to/model"
