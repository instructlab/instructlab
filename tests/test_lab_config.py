# SPDX-License-Identifier: Apache-2.0

# Third Party
from click.testing import CliRunner
import yaml

# First Party
from instructlab import configuration, lab


def test_ilab_config_show(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(lab.ilab, ["--config", "DEFAULT", "config", "show"])
    assert result.exit_code == 0, result.stderr

    parsed = yaml.safe_load(result.stdout_bytes)
    assert parsed

    assert configuration.Config(**parsed)
