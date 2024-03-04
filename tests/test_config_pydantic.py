# Standard
import pydantic
import pydantic_yaml

# 3rd party
from click.testing import CliRunner

# Local
from tests.schema import Config
import cli.lab as lab


def test_config_pydantic():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = CliRunner().invoke(lab.init, ["--interactive", "--taxonomy-path", "taxonomy"])
        try:
            pydantic_yaml.parse_yaml_file_as(model_type=Config, file="config.yaml")
            # If config.yaml parses to Config pydantic model, success.
            assert True
        except TypeError as e:
            print(e)
            assert False
        except pydantic.ValidationError as e:
            print(e)
            assert False
        except:
            assert False
