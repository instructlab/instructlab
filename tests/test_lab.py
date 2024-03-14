# Standard
import re
import unittest

# Third Party
import click

# First Party
from cli import lab


class TestConfig(unittest.TestCase):
    def test_cli_params_hyphenated(self):
        flag_pattern = re.compile("-{1,2}[a-z-]+")
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
