# SPDX-License-Identifier: Apache-2.0
"""Click extensions for InstructLab"""

# Standard
import typing

# Third Party
import click


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
        self, *args: typing.Any, config_section: str | None = None, **kwargs: typing.Any
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.show_default:
            raise ValueError("show_default must be False")
        self.config_section: str | None = config_section

    def get_help_record(self, ctx: click.Context) -> tuple[str, str] | None:
        result = super().get_help_record(ctx)
        if result is None or ctx.default_map is None:
            # hidden is True
            # missing default map (clickman, sphinx)
            return result

        # get default from default_map
        cmd = ctx.command.name
        name = self.name
        if typing.TYPE_CHECKING:
            assert name is not None
        if self.config_section is None:
            if self.name not in ctx.default_map:
                raise ValueError(f"{cmd}.{name} not in default_map {ctx.default_map}")
        else:
            section = ctx.default_map.get(self.config_section, {})
            if self.name not in section:
                raise ValueError(
                    f"{cmd}.{self.config_section}.{name} not in default_map {ctx.default_map}"
                )

        # create help extra
        default_value = self.get_default(ctx)
        if default_value is None:
            default_string = "<None>"
        elif isinstance(default_value, (list, tuple)):
            default_string = ", ".join(str(d) for d in default_value)
        else:
            default_string = str(default_value)

        if self.config_section is None:
            config_name = f"{cmd}.{name}"
        else:
            config_name = f"{cmd}.{self.config_section}.{name}"

        default_msg = f"default: {default_string}; config: '{config_name}'"

        # extend message
        prefix, msg = result
        if msg.endswith("]"):
            msg = f"{msg[:-1]}; {default_msg}]"
        else:
            msg = f"{msg}  [{default_msg}]"
        return prefix, msg

    def get_default(
        self, ctx: click.Context, call: bool = True
    ) -> typing.Any | typing.Callable[[], typing.Any] | None:
        """Lookup default in config subsection

        Used so "serve" option "gpu_layers" looks up its default in
        "serve.llama_cpp.gpu_layers" instead of "serve.gpu_layers".
        """
        if self.config_section is None:
            # no config subsection
            return super().get_default(ctx, call=call)
        if ctx.default_map is not None:
            section = ctx.default_map.get(self.config_section, {})
            value = section.get(self.name)
            if call and callable(value):
                return value()
            return value
        return None
