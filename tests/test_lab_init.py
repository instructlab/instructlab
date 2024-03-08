# Standard
import unittest

# Third Party
from click.testing import CliRunner
import pydantic
import pydantic_yaml

# First Party
from cli import lab
from tests.schema import Config


class TestLabInit(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_init(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = CliRunner().invoke(lab.init, ["--interactive"])
            assert result.exit_code == 0

            result = CliRunner().invoke(lab.init)
            assert result.exit_code == 0

    def test_config_pydantic(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            CliRunner().invoke(lab.init, ["--interactive"])
            try:
                pydantic_yaml.parse_yaml_file_as(model_type=Config, file="config.yaml")
                self.assertTrue
            except TypeError as e:
                print(e)
                self.assertFalse
            except pydantic.ValidationError as e:
                print(e)
                assert self.assertFalse
