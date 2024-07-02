# SPDX-License-Identifier: Apache-2.0

# Standard
import re
import subprocess
import sys

# Third Party
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


def test_ilab_cli_imports():
    # ensure that `ilab` CLI startup is not slowed down by heavy packages
    unwanted = ["deepspeed", "llama_cpp", "torch", "vllm"]
    code = ["import json, sys"]
    for modname in unwanted:
        # block unwanted imports
        code.append(f"sys.modules['{modname}'] = None")
    # import CLI last
    code.append("import instructlab.lab")

    subprocess.check_call([sys.executable, "-c", "; ".join(code)], text=True)
