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


subcommands = [
    # split first and second level for nicer pytest output
    # first, second, extra args
    (None, None, ()),
    ("config", None, ()),
    ("config", "init", ()),
    ("config", "show", ()),
    ("model", None, ()),
    ("model", "chat", ()),
    ("model", "convert", ("--model-dir", "test")),
    ("model", "download", ()),
    ("model", "evaluate", ("--benchmark", "mmlu")),
    ("model", "serve", ()),
    ("model", "test", ()),
    ("model", "train", ()),
    ("data", None, ()),
    ("data", "generate", ()),
    ("system", None, ()),
    ("system", "info", ()),
    ("taxonomy", None, ()),
    ("taxonomy", "diff", ()),
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
]


@pytest.mark.parametrize("first,second", [sc[:2] for sc in subcommands])
def test_ilab_cli_help(first: str | None, second: str | None, cli_runner):
    cmd = ["--config", "DEFAULT"]
    if first is not None:
        cmd.append(first)
    if second is not None:
        cmd.append(second)
    cmd.append("--help")
    result = cli_runner.invoke(lab.ilab, cmd)
    assert result.exit_code == 0, result.stdout


@pytest.mark.parametrize("alias", aliases)
def test_ilab_cli_deprecated_help(alias: str, cli_runner):
    cmd = ["--config", "DEFAULT", alias, "--help"]
    result = cli_runner.invoke(lab.ilab, cmd)
    assert result.exit_code == 0, result.stdout
    assert "this will be deprecated in a future release" in result.stdout


@pytest.mark.parametrize(
    "first,second,extra",
    # only second level subcommands have a --debug-params option
    [sc for sc in subcommands if sc[1] is not None],
)
def test_ilab_cli_debug_params(
    first: str, second: str, extra: typing.Sequence[str], cli_runner: CliRunner
):
    cmd = ["--config", "DEFAULT", first, second]
    cmd.extend(extra)

    result = cli_runner.invoke(lab.ilab, cmd + ["--debug-params"])
    assert result.exit_code == 0, result.stdout

    result = cli_runner.invoke(lab.ilab, cmd + ["--debug-params-json"])
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
    for primary, secondary, _ in subcommands:
        sub = tested.setdefault(primary, set())
        sub.add(secondary or "")

    assert ilab_commands == tested

    ep_aliases = metadata.entry_points(group="instructlab.command.alias")
    assert set(aliases) == ep_aliases.names
