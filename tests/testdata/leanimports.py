"""Helper for test_ilab_cli_imports"""

# Standard
from importlib import metadata
import sys

# block slow imports
for unwanted in ["deepspeed", "llama_cpp", "torch", "vllm"]:
    # importlib raises ModuleNotFound when sys.modules value is None.
    assert unwanted not in sys.modules
    sys.modules[unwanted] = None  # type: ignore[assignment]

for command in metadata.entry_points(group="instructlab.command"):
    print(f"{command.name}")
    command.load()
    for subcommand in metadata.entry_points(
        group=f"instructlab.command.{command.name}"
    ):
        print(f"  {command.name}.{subcommand.name}")
        subcommand.load()
