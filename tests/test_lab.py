# SPDX-License-Identifier: Apache-2.0

# Standard
from importlib import metadata
import json
import pathlib
import re
import subprocess
import sys
import typing

# Third Party
from click.testing import CliRunner
import click
import pytest

# First Party
from instructlab import configuration as config
from instructlab import lab
from instructlab.clickext import ConfigOption, get_default_and_description
from instructlab.utils import is_macos_with_m_chip


def get_commands():
    commands: typing.Dict[str, typing.Set[str]] = {}
    for primary in metadata.entry_points(group="instructlab.command"):
        sub = commands.setdefault(primary.name, set())
        sub.add(
            ""
        )  # Add empty string to handle primary command without secondary command
        for secondary in metadata.entry_points(
            group=f"instructlab.command.{primary.name}"
        ):
            sub.add(secondary.name)
    return commands


class TestConfig:
    def test_cli_params_hyphenated(self):
        flag_pattern = re.compile("-{1,2}[0-9a-z-]+")
        invalid_flags = []
        commands = get_commands()
        for prim, secondaries in commands.items():
            entry_points = metadata.entry_points(group=f"instructlab.command.{prim}")
            for sec in secondaries:
                for entry_point in entry_points:
                    if entry_point.name != sec:
                        continue
                    command = entry_point.load()
                    if not isinstance(command, click.Command):
                        continue
                    for param in command.params:
                        if not isinstance(param, ConfigOption):
                            continue
                        # Do this long one-liner to avoid linter complaint:
                        # Too many nested blocks (6/5) (too-many-nested-blocks)
                        invalid_flags.extend(
                            f"{prim} {opt}"
                            for opt in param.opts
                            if not flag_pattern.fullmatch(opt)
                        )
        assert not invalid_flags, "<- these commands are using non-hyphenated params"


def test_llamap_cpp_import():
    # Third Party
    import llama_cpp

    llama_cpp.llama_backend_init()


def test_import_mlx():
    # smoke test to verify that mlx is always available on Apple Silicon
    # but never on Linux and Intel macOS.
    if is_macos_with_m_chip():
        assert __import__("mlx")
    else:
        with pytest.raises(ModuleNotFoundError):
            __import__("mlx")


def test_ilab_cli_imports(testdata_path: pathlib.Path):
    script = testdata_path / "leanimports.py"
    subprocess.check_call([sys.executable, str(script)], text=True)


class Command(typing.NamedTuple):
    args: tuple[str, ...]
    extra_args: tuple[str, ...] = ()
    needs_config: bool = True
    should_fail: bool = True

    def get_args(self, *extra, default_config: bool = True) -> list[str]:
        args = []
        if self.needs_config and default_config:
            args.extend(["--config", "DEFAULT"])
        args.extend(self.args)
        args.extend(self.extra_args)
        args.extend(extra)
        return args

    @property
    def has_debug_params(self) -> bool:
        # only second level subcommands have a --debug-params option
        return len(self.args) > 1


subcommands: list[Command] = [
    # split first and second level for nicer pytest output
    # first, second, extra args
    Command((), needs_config=False, should_fail=False),
    Command(("config",), needs_config=False, should_fail=False),
    Command(("config", "edit")),
    Command(("config", "init"), needs_config=False, should_fail=False),
    Command(("config", "show")),
    Command(("model",), needs_config=False, should_fail=False),
    Command(("model", "chat")),
    Command(("model", "convert"), ("--model-dir", "test")),
    Command(("model", "download")),
    Command(("model", "evaluate"), ("--benchmark", "mmlu")),
    Command(("model", "serve")),
    Command(("model", "test")),
    Command(("model", "train")),
    Command(("model", "list")),
    Command(
        ("model", "upload"),
        ("--model", "foo", "--destination", "bar"),
    ),
    Command(("data",), needs_config=False, should_fail=False),
    Command(("data", "generate")),
    Command(("data", "list")),
    Command(("system",), needs_config=False, should_fail=False),
    Command(("system", "info"), needs_config=False, should_fail=False),
    Command(("taxonomy",), needs_config=False, should_fail=False),
    Command(("taxonomy", "diff")),
    Command(("process",), needs_config=False, should_fail=False),
    Command(("process", "list")),
    Command(("process", "attach")),
]

aliases = [
    "serve",
    "train",
    "chat",
    "generate",
]


@pytest.mark.parametrize("command", subcommands, ids=lambda sc: repr(sc.args))
def test_ilab_cli_help(command: Command, cli_runner: CliRunner):
    cmd = command.get_args("--help")
    assert "--help" in cmd
    result = cli_runner.invoke(lab.ilab, cmd)
    assert result.exit_code == 0, result.stdout


def test_ilab_alias_output(cli_runner: CliRunner):
    expected_output = """Aliases:
  chat      model chat
  generate  data generate
  serve     model serve
  train     model train"""
    result = cli_runner.invoke(lab.ilab)
    assert result.exit_code == 0, result.stdout

    alias_section = False
    actual_output = []

    for line in result.stdout.splitlines():
        if line.strip() == "Aliases:":
            alias_section = True
        if alias_section:
            actual_output.append(line)

    actual_output_str = "\n".join(actual_output).strip()
    assert (
        expected_output == actual_output_str
    ), f"Expected aliases output:\n{expected_output}\n\nBut got:\n{actual_output_str}"


def test_cli_help_matches_field_description(cli_runner: CliRunner):
    commands = get_commands()
    for prim, secondaries in commands.items():
        if prim == "config":
            continue
        for sec in secondaries:
            command_name = f"{prim} {sec}".strip()
            entry_points = metadata.entry_points(group=f"instructlab.command.{prim}")
            for entry_point in entry_points:
                if entry_point.name != sec:
                    continue
                command = entry_point.load()
                if not isinstance(command, click.Command):
                    continue
                command_name_to_list = command_name.split()
                result = cli_runner.invoke(
                    lab.ilab,
                    ["--config", "DEFAULT"] + command_name_to_list + ["--help"],
                )
                assert "Usage:" in result.output
                assert result.exit_code == 0, result.output

                for param in command.params:
                    if not isinstance(param, ConfigOption):
                        continue
                    cfg = config.get_default_config()
                    # Build the config name to retrieve the description
                    # We only want the current command not the parent command
                    # for instance command_name_to_list is "model generate", "model" is
                    # not known in the Pydantic Config so we only want "generate". We
                    # use the last element in case we add more positional arguments in
                    # the future.
                    config_identifier = (
                        [command_name_to_list[-1]] if command_name_to_list[-1] else []
                    )
                    if hasattr(param, "config_sections") and param.config_sections:
                        config_identifier += param.config_sections
                    config_identifier.append(str(param.name))

                    # Fetch the description
                    description, _ = get_default_and_description(cfg, config_identifier)

                    # This is annoying but for additional_args, the description
                    # string is "Additional arguments to pass to the training
                    # script. These arguments are passed as key-value pairs to the
                    # training script." but when the help is generated, the text is
                    # truncated at the key-value word and we end up with
                    # "key-\nvalue", so when we normalize the output and remove the
                    # newlines this gives us "key- value" which is NOT what we want.
                    # So we need to replace "- " with "-" to make sure we match.
                    normalize_output = " ".join(result.stdout.split()).replace(
                        "- ", "-"
                    )
                    assert str(description) in normalize_output, normalize_output


@pytest.mark.parametrize(
    "command",
    [sc for sc in subcommands if sc.has_debug_params],
    ids=lambda sc: repr(sc.args),
)
def test_ilab_cli_debug_params(command: Command, cli_runner: CliRunner):
    result = cli_runner.invoke(lab.ilab, command.get_args("--debug-params"))
    assert result.exit_code == 0, result.stdout

    result = cli_runner.invoke(lab.ilab, command.get_args("--debug-params-json"))
    assert result.exit_code == 0, result.stdout
    j = json.loads(result.stdout)
    assert isinstance(j, dict)


def test_ilab_commands_tested():
    ilab_commands = {None: set([""])}
    for primary in metadata.entry_points(group="instructlab.command"):
        sub = ilab_commands.setdefault(primary.name, set())
        sub.add("")
        for secondary in metadata.entry_points(
            group=f"instructlab.command.{primary.name}"
        ):
            sub.add(secondary.name)

    tested = {None: set([""])}
    for command in subcommands:
        if not command.args:
            continue
        sub = tested.setdefault(command.args[0], set())
        if len(command.args) == 2:
            sub.add(command.args[1])
        else:
            sub.add("")

    assert ilab_commands == tested

    ep_aliases = metadata.entry_points(group="instructlab.command.alias")
    assert set(aliases) == ep_aliases.names


@pytest.mark.parametrize(
    "command",
    subcommands,
    ids=lambda sc: repr(sc.args),
)
def test_ilab_missing_config(command: Command, cli_runner: CliRunner) -> None:
    cmd = command.get_args(default_config=False)
    assert "--config" not in cmd
    result = cli_runner.invoke(lab.ilab, cmd)

    if command.needs_config:
        assert result.exit_code == 2, result
        assert "does not exist or is not a readable file" in result.stdout
    else:
        # should fail due to missing dirs
        if command.should_fail:
            assert result.exit_code == 1, result
            assert (
                "Some ilab storage directories do not exist yet. Please run `ilab config init` before continuing."
                in result.stdout
            )
        else:
            assert result.exit_code == 0, result
