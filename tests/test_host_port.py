# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code

from click.testing import CliRunner
from instructlab import lab
import pytest

@pytest.mark.parametrize(
    "args, expected_host, expected_port",
    [
        (["model", "serve"], "127.0.0.1", "8000"),
        (["model", "serve", "--host-port", "192.168.1.1:8080"], "192.168.1.1", "8080"),
        (["model", "serve", "--host-port", "192.168.1.2:8000"], "192.168.1.2", "8000"),
        (["model", "serve", "--host-port", "127.0.0.1:9090"], "127.0.0.1", "9090"),
    ],
)
def test_serve(args, expected_host, expected_port):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(lab.ilab, args)
        assert result.exit_code == 0
        assert expected_host in result.output
        assert expected_port in result.output
