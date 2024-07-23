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
from instructlab import lab
from instructlab.utils import is_macos_with_m_chip


class TestConfig:
    def test_cli_params_hyphenated(self):
        flag_pattern = re.compile("-{1,2}[0-9a-z-]+")
        invalid_flags = []
        for name, cmd in lab.ilab.commands.items():
            for param in cmd.params:
                if not isinstance(param, click.Option):
                    continue
                for opt in param.opts:
                    if not flag_pattern.fullmatch(opt):
                        invalid_flags.append(f"{name} {opt}")
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
    Command((), needs_config=False),
    Command(("config",), needs_config=False),
    Command(("config", "init"), needs_config=False),
    Command(("config", "show")),
    Command(("model",), needs_config=False),
    Command(("model", "chat")),
    Command(("model", "convert"), ("--model-dir", "test")),
    Command(("model", "download")),
    Command(("model", "evaluate"), ("--benchmark", "mmlu")),
    Command(("model", "serve")),
    Command(("model", "test")),
    Command(("model", "train")),
    Command(("model", "list")),
    Command(("data",), needs_config=False),
    Command(("data", "generate")),
    Command(("system",), needs_config=False),
    Command(("system", "info"), needs_config=False),
    Command(("taxonomy",), needs_config=False),
    Command(("taxonomy", "diff")),
]

aliases = [
    "serve",
    "train",
    "convert",
    "chat",
    "test",
    "evaluate",
    "init",
    "download",
    "diff",
    "generate",
    "sysinfo",
    "list",
]


@pytest.mark.parametrize("command", subcommands, ids=lambda sc: repr(sc.args))
def test_ilab_cli_help(command: Command, cli_runner: CliRunner):
    cmd = command.get_args("--help")
    assert "--help" in cmd
    result = cli_runner.invoke(lab.ilab, cmd)
    assert result.exit_code == 0, result.stdout


@pytest.mark.parametrize("alias", aliases)
def test_ilab_cli_deprecated_help(alias: str, cli_runner):
    cmd = ["--config", "DEFAULT", alias, "--help"]
    result = cli_runner.invoke(lab.ilab, cmd)
    assert result.exit_code == 0, result.stdout
    assert "this will be deprecated in a future release" in result.stdout


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
        assert result.exit_code == 0, result
