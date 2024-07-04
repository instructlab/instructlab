# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code

from click.testing import CliRunner
from instructlab import lab


class TestServe:
    def test_serve_with_defaults(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(lab.ilab, ["model", "serve"])
            assert result.exit_code == 0
            assert "127.0.0.1" in result.output
            assert "8000" in result.output 

    def test_serve_with_custom_host_and_port(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(lab.ilab, ["model", "serve", "--host", "192.168.1.1", "--port", "8080"])
            assert result.exit_code == 0
            assert "192.168.1.1" in result.output
            assert "8080" in result.output 

    def test_serve_with_custom_host(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(lab.ilab, ["model", "serve", "--host", "192.168.1.2"])
            assert result.exit_code == 0
            assert "192.168.1.2" in result.output
            assert "8000" in result.output 

    def test_serve_with_custom_port(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(lab.ilab, ["model", "serve", "--port", "9090"])
            assert result.exit_code == 0
            assert "127.0.0.1" in result.output
            assert "9090" in result.output