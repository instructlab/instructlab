# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import pathlib

# Third Party
from click.testing import CliRunner
import pytest
import yaml

# First Party
from instructlab import configuration, lab


def test_ilab_config_show(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(lab.ilab, ["--config", "DEFAULT", "config", "show"])
    assert result.exit_code == 0, result.stdout

    parsed = yaml.safe_load(result.stdout_bytes)
    assert parsed

    assert configuration.Config(**parsed)


@pytest.mark.parametrize(
    "command",
    [
        (["config", "init"]),
        (
            ["init"]
        ),  # TODO: remove this test once the deprecated alias 'ilab init' is removed
    ],
)
def test_ilab_config_init_with_env_var_config(
    cli_runner: CliRunner, tmp_path: pathlib.Path, command: list
) -> None:
    # Common setup code
    cfg = configuration.get_default_config()
    assert cfg.general.log_level == logging.getLevelName(logging.INFO)
    cfg.general.log_level = logging.getLevelName(logging.DEBUG)
    cfg_file = tmp_path / "config-gold.yaml"
    with cfg_file.open("w") as f:
        yaml.dump(cfg.model_dump(), f)

    # Invoke the CLI command
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
