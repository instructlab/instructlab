# local
# Third Party
# 3rd party
import pydantic
import pydantic_yaml  # YAML parsing utility

# First Party
from tests.schema import Config
import cli.lab as lab


def test_config_pydantic():
    """
    Assumes for the moment:
        (1) config.yaml is at /cli/config.yaml of project
        (2) `pytest` is run from root of project.
    """

    # check if config.yaml is here

    try:
        parsed = pydantic_yaml.parse_yaml_file_as(model_type=Config, file="config.yaml")
        # If config.yaml parses to Config pydantic model, success.
        assert True
    except TypeError as e:
        print(e)
        assert False
    except pydantic.ValidationError as e:
        print(e)
        assert False
