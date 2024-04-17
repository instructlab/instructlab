# SPDX-License-Identifier: Apache-2.0

# Standard
import re
import unittest

# Third Party
import click
import pytest

# First Party
from cli import lab
from cli.utils import is_macos_with_m_chip


class TestConfig(unittest.TestCase):
    def test_cli_params_hyphenated(self):
        flag_pattern = re.compile("-{1,2}[0-9a-z-]+")
        invalid_flags = []
        for name, cmd in lab.cli.commands.items():
            for param in cmd.params:
                if not isinstance(param, click.Option):
                    continue
                for opt in param.opts:
                    if not flag_pattern.fullmatch(opt):
                        invalid_flags.append(f"{name} {opt}")
        self.assertFalse(
            invalid_flags, "<- these commands are using non-hyphenated params"
        )


def test_import_mlx():
    # smoke test to verify that mlx is always available on Apple Silicon
    # but never on Linux and Intel macOS.
    if is_macos_with_m_chip():
        assert __import__("mlx")
    else:
        with pytest.raises(ModuleNotFoundError):
            __import__("mlx")
