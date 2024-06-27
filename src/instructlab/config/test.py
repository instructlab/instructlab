# from instructlab.configuration import _general, _generate, _serve, _train
# Standard
import typing

# Third Party
from pydantic import BaseModel, ConfigDict
import yaml

# First Party
from instructlab.configuration import get_default_config


class _chat(BaseModel):
    """Class describing configuration of the chat sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore")

    # required fields
    model: str

    # additional fields with defaults
    vi_mode: bool = False
    visible_overflow: bool = True
    context: str = "default"
    session: typing.Optional[str] = None
    logs_dir: str = "data/chatlogs"
    greedy_mode: bool = False
    max_tokens: typing.Optional[int] = None

    # def __repr__(self):
    #     return yaml.dump(self.model_dump_json())

    # def __str__(self):
    #     return self.__repr__()


if __name__ == "__main__":
    # Standard
    import json

    # Third Party
    import yaml

    cfg = get_default_config()
    test = _chat(
        model="test",
        # model_config={"test": "test"},
    )
    # test.model_config = {"test": "test"}
    print(cfg.model_dump_json())
    yaml_obj = yaml.load(cfg.model_dump_json(), Loader=yaml.SafeLoader)
    print(yaml_obj)
    # now add a print statement that pretty prints the YAML
    # Pretty print the YAML
    print(yaml.dump(yaml_obj, indent=4))
