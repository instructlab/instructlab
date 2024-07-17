# SPDX-License-Identifier: Apache-2.0
"""Click extensions for InstructLab"""

# Standard
import functools
import json
import logging
import os
import sys
import typing

# Third Party
from click.core import ParameterSource
import click

logger = logging.getLogger(__name__)


class ConfigOption(click.Option):
    """A click.Option with extended default lookup help

    The defaults can be looked up in a subsection of the context's
    default_map.

    The help output of ConfigOption class reads defaults from the context's
    default_map and shows the name of the config settings.

    Example:
        Options:
        --model-path PATH           Path to the model used during generation.
                                    [default: models/merlinite-7b-lab-Q4_K_M.gguf;
                                    config: 'serve.model_path']
    """

    def __init__(
        self,
        *args: typing.Any,
        config_sections: str | None = None,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.show_default:
            raise ValueError("show_default must be False")
        if config_sections:
            self.config_sections = tuple(config_sections.split("."))
        else:
            self.config_sections = ()

    def get_help_record(self, ctx: click.Context) -> tuple[str, str] | None:
        result = super().get_help_record(ctx)
        if result is None or ctx.default_map is None:
            # hidden is True
            # missing default map (clickman, sphinx)
            return result

        # get default from default_map
        cmd = ctx.command.name
        name = self.name
        if self.config_sections:
            config_name = f"{cmd}.{'.'.join(self.config_sections)}.{name}"
        else:
            config_name = f"{cmd}.{name}"

        if typing.TYPE_CHECKING:
            assert name is not None

        section = ctx.default_map
        for secname in self.config_sections:
            section = section.get(secname, {})
        if self.name not in section:
            raise ValueError(f"{config_name} not in default_map {ctx.default_map}")

        # create help extra
        default_value = self.get_default(ctx)
        if default_value is None:
            default_string = "<None>"
        elif isinstance(default_value, (list, tuple)):
            default_string = ", ".join(str(d) for d in default_value)
        else:
            default_string = str(default_value)

        default_msg = f"default: {default_string}; config: '{config_name}'"

        # extend message
        prefix, msg = result
        if msg.endswith("]"):
            msg = f"{msg[:-1]}; {default_msg}]"
        else:
            msg = f"{msg}  [{default_msg}]"
        return prefix, msg

    def consume_value(
        self, ctx: click.Context, opts: typing.Mapping[str, typing.Any]
    ) -> tuple[typing.Any, ParameterSource]:
        value, source = super().consume_value(ctx, opts)
        # fix parameter source for config section that are mis-reported
        # as DEFAULT source instead of DEFAULT_MAP source.
        if (
            source == ParameterSource.DEFAULT
            and self.config_sections
            and ctx.default_map is not None
            and self.name is not None
        ):
            section = ctx.default_map
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
