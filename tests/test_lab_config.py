# SPDX-License-Identifier: Apache-2.0

# Standard
import json
import logging
import pathlib

# Third Party
from click.testing import CliRunner
from ruamel.yaml import YAML
import ruamel.yaml
import yaml

# First Party
from instructlab import configuration, lab

test_source_config_data = """
# Chat configuration section.
chat:
  # Predefined setting or environment that influences the behavior and responses of
  # the chat assistant. Each context is associated with a specific prompt that
  # guides the assistant on how to respond to user inputs. Available contexts:
  # default, cli_helper.
  # Default: default
  context: default
  # Sets temperature to 0 if enabled, leading to more deterministic responses.
  # Default: False
  greedy_mode: false
  # Directory where chat logs are stored.
  # Default: /data/instructlab/chatlogs
  logs_dir: /data/instructlab/chatlogs
  # The maximum number of tokens that can be generated in the chat completion. Be
  # aware that larger values use more memory.
  # Default: None
  max_tokens:
  # Model to be used for chatting with.
  # Default: /cache/instructlab/models/merlinite-7b-lab-Q4_K_M.gguf
  model: /cache/instructlab/models/merlinite-7b-lab-Q4_K_M.gguf
  # Filepath of a dialog session file.
  # Default: None
  session:
  # Enable vim keybindings for chat.
  # Default: False
  vi_mode: false
  # Renders vertical overflow if enabled, displays ellipses otherwise.
  # Default: True
  visible_overflow: true
# General configuration section.
general:
  # Debug level for logging.
  # Default: 0
  debug_level: 0
  # Log format. https://docs.python.org/3/library/logging.html#logrecord-attributes
  # Default: %(levelname)s %(asctime)s %(name)s:%(lineno)d: %(message)s
  log_format: '%(levelname)s %(asctime)s %(name)s:%(lineno)d: %(message)s'
  # Log level for logging.
  # Default: INFO
  log_level: INFO
"""


def remove_comment_indentation(yaml_output: str) -> str:
    return "\n".join(
        line.lstrip() if line.lstrip().startswith("#") else line
        for line in yaml_output.splitlines()
    )


def test_ilab_config_show(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(lab.ilab, ["--config", "DEFAULT", "config", "show"])
    assert result.exit_code == 0, result.stdout

    parsed = yaml.safe_load(result.stdout_bytes)
    assert parsed

    assert configuration.Config(**parsed)


def test_ilab_config_show_json_output(
    cli_runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    test_config_path = tmp_path / "test_config.yaml"
    test_config_path.write_text(test_source_config_data)

    logging.disable(logging.CRITICAL)
    result_json = cli_runner.invoke(
        lab.ilab, ["--config", str(test_config_path), "config", "show", "--json"]
    )
    logging.disable(logging.NOTSET)

    assert result_json.exit_code == 0
    parsed_json = json.loads(result_json.stdout)
    assert "chat" in parsed_json
    assert "general" in parsed_json


def test_ilab_config_show_key_general(
    cli_runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    # Expected output as a JSON string for --key general
    test_config_path = tmp_path / "test_config.yaml"
    test_config_path.write_text(test_source_config_data)

    cfg = configuration.get_default_config()
    expected_general_output = cfg.general.model_dump()

    logging.disable(logging.CRITICAL)
    # Test --key output for general
    result_key_general = cli_runner.invoke(
        lab.ilab,
        [
            "--config",
            str(test_config_path),
            "config",
            "show",
            "--key",
            "general",
            "--json",
        ],
    )
    logging.disable(logging.NOTSET)

    assert result_key_general.exit_code == 0
    parsed_output = json.loads(result_key_general.stdout)
    assert parsed_output == expected_general_output


def test_ilab_config_show_key_general_yaml(
    cli_runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    test_config_path = tmp_path / "test_config.yaml"
    test_config_path.write_text(test_source_config_data)

    cfg = configuration.read_config(str(test_config_path))
    general_config_commented_map = configuration.config_to_commented_map(cfg.general)

    yaml_stream = ruamel.yaml.compat.StringIO()
    yaml_instance = YAML()
    yaml_instance.indent(mapping=2, sequence=4, offset=0)
    yaml_instance.dump(general_config_commented_map, yaml_stream)
    expected_yaml_output = yaml_stream.getvalue()

    # Test --key output for general in YAML
    logging.disable(logging.CRITICAL)
    result_key_general_yaml = cli_runner.invoke(
        lab.ilab,
        ["--config", str(test_config_path), "config", "show", "--key", "general"],
    )
    logging.disable(logging.NOTSET)

    assert result_key_general_yaml.exit_code == 0
    actual_output = remove_comment_indentation(result_key_general_yaml.stdout.strip())
    expected_yaml_output = remove_comment_indentation(expected_yaml_output.strip())
    assert actual_output == expected_yaml_output


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
