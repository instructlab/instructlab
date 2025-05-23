# SPDX-License-Identifier: Apache-2.0
"""Click extensions for InstructLab"""

# Standard
from importlib import metadata
import enum
import functools
import json
import logging
import os
import sys
import typing

# Third Party
from click.core import ParameterSource
from click_didyoumean import DYMGroup
import click

# First Party
from instructlab.configuration import DEFAULTS, BaseModel, get_dict, storage_dirs_exist

logger = logging.getLogger(__name__)


class LazyEntryPointGroup(DYMGroup):
    """Lazy load commands from an entry point group

    Entry points are defined in `pyproject.toml`. Example:

        [project.entry-points."instructlab.command.config"]
        "init" = "instructlab.config.init:init"

    This defines the command `ilab config init` with the function
    `from instructlab.config.init import init` as click command.
    """

    def __init__(self, *args, ep_group: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps: metadata.EntryPoints = metadata.entry_points(group=ep_group)
        if not self.eps.names:
            raise ValueError(f"{ep_group} is empty")

    def list_commands(self, ctx: click.Context) -> list[str]:
        result = list(super().list_commands(ctx))
        result.extend(sorted(self.eps.names))
        return result

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name in self.eps.names:
            cmd = self.eps[cmd_name].load()
        else:
            cmd = super().get_command(ctx, cmd_name)
        if typing.TYPE_CHECKING:
            assert isinstance(cmd, click.Command) or cmd is None
        return cmd


class ExpandAliasesGroup(LazyEntryPointGroup):
    """Lazy load commands and aliases from entry point groups"""

    def __init__(self, *args, alias_ep_group: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alias_eps: metadata.EntryPoints = metadata.entry_points(
            group=alias_ep_group
        )

    def get_alias_info(self, cmd_name: str) -> tuple[str, str]:
        ep: metadata.EntryPoint = self.alias_eps[cmd_name]
        module_parts = ep.module.split(".", 3)
        # TODO: @aliryan revert the if else once chat split gets in bc
        # the alias range will be consistent again
        if len(module_parts) > 3:
            return module_parts[2], module_parts[3]
        return module_parts[1], module_parts[2]

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name in self.alias_eps.names:
            cmd = self.alias_eps[cmd_name].load()
            if typing.TYPE_CHECKING:
                assert isinstance(cmd, click.Command)
            # if some storage dirs do not exist
            # AND the --config flag is not customized, then we error
            if not storage_dirs_exist() and (
                ctx.params["config_file"] == DEFAULTS.CONFIG_FILE
            ):
                click.secho(
                    "Some ilab storage directories do not exist yet. Please run `ilab config init` before continuing.",
                    fg="red",
                )
                raise click.exceptions.Exit(1)
            return cmd
        return super().get_command(ctx, cmd_name)

    def format_epilog(self, ctx: click.Context, formatter):
        """Inject our aliases into the help string"""
        if self.alias_eps.names:
            formatter.write_paragraph()
            formatter.write_text("Aliases:")
            with formatter.indentation():
                cmd_names = sorted(self.alias_eps.names)
                pad = max(len(cmd_name) for cmd_name in cmd_names)
                for cmd_name in cmd_names:
                    group, primary = self.get_alias_info(cmd_name)
                    formatter.write_text(f"{cmd_name:<{pad}}  {group} {primary}\n")

        super().format_epilog(ctx, formatter)


class ConfigOption(click.Option):
    """A click.Option with extended default lookup help

    The defaults and description can be looked up in a subsection of the context's
    config object.

    The help output of ConfigOption class reads defaults from the context's
    config object field and shows the name of the config settings.

    Example:
        Options:
        --model-path PATH           Path to the model used during generation.
                                    [default: models/granite-7b-lab-Q4_K_M.gguf;
                                    config: 'serve.model_path']
    """

    def __init__(
        self,
        *args: typing.Any,
        config_sections: str | None = None,
        config_class: str | None = None,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.show_default:
            raise ValueError("show_default must be False")
        if self.help:
            raise ValueError(
                f"help must not be set for '{self.name}', it is derived from the Field description in the Config class"
            )
        if config_class:
            self.config_class = config_class
        else:
            self.config_class = ""
        if config_sections:
            self.config_sections = list(config_sections.split("."))
        else:
            self.config_sections = []

    def get_help_record(self, ctx: click.Context) -> tuple[str, str] | None:
        result = super().get_help_record(ctx)
        if not self.config_class and (result is None or ctx.default_map is None):
            # hidden is True
            # missing default map (clickman, sphinx)
            return result
        if result is None:
            result = ("", "")

        if self.config_class:
            cmd = self.config_class
        else:
            cmd = str(ctx.command.name)
        assert isinstance(cmd, str)
        name = self.name

        if self.config_sections:
            config_identifier = (
                [str(cmd)]
                + [
                    str(section)
                    for section in self.config_sections
                    if section is not None
                ]
                + [str(name)]
            )
        else:
            config_identifier = [str(cmd), str(name)]

        if typing.TYPE_CHECKING:
            assert name is not None

        # create help extra
        description, default_value = get_default_and_description(
            ctx.obj.config, config_identifier
        )
        if default_value is None:
            default_string = "<None>"
        elif isinstance(default_value, (list, tuple)):
            default_string = ", ".join(str(d) for d in default_value)
        elif isinstance(default_value, enum.Enum):
            default_string = default_value.value
        else:
            default_string = str(default_value)

        default_msg = (
            f"default: {default_string}; config: '{'.'.join(config_identifier)}'"
        )

        # extend message
        prefix, msg = result
        # If the 'help' field on click.Option is not set, then 'msg' will be None
        # unless options like 'required' are set
        if msg:
            # Needed to handle [required] option for example
            if msg.endswith("]"):
                msg = f"{description} {msg[:-1]}; {default_msg}]"
            else:
                msg = f"{description} {msg} [{default_msg}]"
        else:
            # do not append description if msg is None to avoid space
            msg = f"{description} [{default_msg}]"  # type: ignore
        return prefix, msg

    def consume_value(
        self, ctx: click.Context, opts: typing.Mapping[str, typing.Any]
    ) -> tuple[typing.Any, ParameterSource]:
        value, source = super().consume_value(ctx, opts)

        # overwrite the value if a userpassed a config_class=
        # this indicates they want to inherit the value from a different config class than their commands. This is a common practice with complex commands
        default_or_map = source in (
            ParameterSource.DEFAULT,
            ParameterSource.DEFAULT_MAP,
        )
        if default_or_map and self.config_class:
            section = get_dict(ctx.obj.config).get(self.config_class)
            assert isinstance(section, dict)
            # we need to overwrite the value
            for secname in self.config_sections:
                section = section.get(secname, {})
            value = section.get(self.name)
        # fix parameter source for config section that are mis-reported
        # as DEFAULT source instead of DEFAULT_MAP source.
        if (
            source == ParameterSource.DEFAULT
            and (self.config_sections or self.config_class)
            and ctx.default_map is not None
            and self.name is not None
        ):
            if self.config_class:
                section = get_dict(ctx.obj.config).get(self.config_class)
            else:
                section = ctx.default_map
            assert isinstance(section, dict)
            for secname in self.config_sections:
                section = section.get(secname, {})
            if self.name in section:
                source = ParameterSource.DEFAULT_MAP
        return value, source

    def get_default(
        self, ctx: click.Context, call: bool = True
    ) -> typing.Any | typing.Callable[[], typing.Any] | None:
        """Lookup default in config subsection

        Used so "serve" option "gpu_layers" looks up its default in
        "serve.llama_cpp.gpu_layers" instead of "serve.gpu_layers".
        """
        if not self.config_sections:
            # no config subsection
            return super().get_default(ctx, call=call)
        if ctx.default_map is not None and self.name is not None:
            cfg_dict = get_dict(ctx.obj.config)
            if self.config_class:
                section = cfg_dict.get(self.config_class, {})
            else:
                section = ctx.default_map
            for secname in self.config_sections:
                section = section.get(secname, {})
            value = section.get(self.name)
            if call and callable(value):
                return value()
            return value
        return None


def _get_param_info(
    ctx: click.Context, **kwargs: dict
) -> typing.Generator[tuple[str, typing.Any, str, str], None, None]:
    """Get click parameter information

    Returns name, value, type name, and parameter source
    """
    for key, value in kwargs.items():
        src = ctx.get_parameter_source(key)
        param_src: str = src.name.lower() if src is not None else "unknown"
        type_name: str
        if value is None:
            type_name = "None"
        else:
            type_: type = type(value)
            type_name = type_.__name__
            mod_name: str | None = getattr(type_, "__module__", None)
            if mod_name and mod_name != "builtins":
                type_name = f"{mod_name}.{type_name}"
        yield key, value, type_name, param_src
    # additional args
    if ctx.args:
        yield "args", ctx.args, "list", "args"


class _ParamEncoder(json.JSONEncoder):
    """Custom encoder for additional parameter types"""

    def default(self, o: typing.Any) -> typing.Any:
        if isinstance(o, os.PathLike):
            return os.fsdecode(o)
        return super().default(o)


def display_params(f: typing.Callable) -> typing.Callable:
    """Display command parameters decorator

    The function decorator adds two hidden parameters to a click Command.

    - `--debug-params` dumps parameter info for humans and exits immediately.
    - `--display-param-json` dumps parameter info as JSON and exits immediately.

    When DEBUG logging is enabled, it also dumps parameter information for
    humans, but does not exit immediately. The output contains parameter
    name, value, value type (as qualified dotted string), and the source of
    the value (commandline, environment, default, default_map, args,
    unknown).
    """

    @functools.wraps(f)
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        # some callbacks don't need ctx
        ctx: click.Context | None = kwargs.get("ctx")
        if ctx is None:
            ctx = click.get_current_context()
        display: str = kwargs.pop("_debug_params")

        # also display params when debug logging is enabled
        if (
            display is None
            and logger.isEnabledFor(logging.DEBUG)
            and not kwargs.get("quiet", False)
        ):
            display = "logger"

        if kwargs and display in {"logger", "human"}:
            pad = min(max(len(name) for name in kwargs), 24)
            print("Parameters:")
            for name, value, typ, src in _get_param_info(ctx, **kwargs):
                print(f"  {name:>{pad}}: {value!r:<8} \t[type: {typ}, src: {src}]")
        elif display == "json":
            params = {
                name: (value, type_name, param_src)
                for name, value, type_name, param_src in _get_param_info(ctx, **kwargs)
            }
            json.dump({"params": params}, sys.stdout, cls=_ParamEncoder)

        # --debug-params* exit
        if display in {"json", "human"}:
            ctx.exit()

        return f(*args, **kwargs)

    human_option = click.option(
        "--debug-params",
        "_debug_params",
        flag_value="human",
        hidden=True,
        help="display parameter info and exit.",
    )
    json_option = click.option(
        "--debug-params-json",
        "_debug_params",
        flag_value="json",
        hidden=True,
        help="display parameter as JSON and exit.",
    )

    return human_option(json_option(wrapper))


def get_default_and_description(
    cfg: BaseModel, config_identifier: list[str]
) -> typing.Tuple[str | None, typing.Any]:
    """
    Retrieve the default value and description for a given configuration field name.

    This function searches through the fields of a Pydantic BaseModel to find a field
    that matches the provided configuration name. If the field is found, it returns
    the field's description and default value. If the field is a nested model, the
    function is called recursively to search within the nested model.

    Args:
        cfg (BaseModel): The Pydantic model instance containing the configuration fields.
        config_identifier (list[str]): A list of field names to search for in the model.

    Returns:
        typing.Tuple[str | None, typing.Any]: A tuple containing the field's description
        and default value. If the field is not found, a ValueError is raised.

    Raises:
        ValueError: If the specified config_identifier is not found in the model.
    """
    # Loop through the fields of the model
    for field_name, field in cfg.model_fields.items():
        if field_name == config_identifier[0]:
            value = getattr(cfg, field_name)
            description = field.description
            default_value = field.get_default(call_default_factory=True)

            # If the value is a nested model and there are more names to check, recurse
            # Slice the config_identifier list to remove the current field name
            if isinstance(value, BaseModel) and len(config_identifier) > 1:
                return get_default_and_description(value, config_identifier[1:])

            return description, default_value

    # If no match is found, raise an exception
    raise ValueError(f"{config_identifier} not in Config object")
