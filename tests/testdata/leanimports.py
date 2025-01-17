"""Helper for test_ilab_cli_imports.
For the unit tests, leanimports.py gets used from tests/test_lab.py in the test_ilab_cli_imports method in the following code:

def test_ilab_cli_imports(testdata_path: pathlib.Path):
    script = testdata_path / "leanimports.py"
    subprocess.check_call([sys.executable, str(script)], text=True)

What this test does is start a new Python process that runs leanimports.py. All that spawned process does is go through and
load each command/subcommand in the ilab CLI. It doesn't actually run any commands - just loads them. And, it ensures that
slow imports - such as torch - don't load when running any of the commands because if they do, it slows down the entire ilab
CLI even for simple operations that for example would not need to use torch.

Unit tests that need things that depend on torch should make sure those things load torch only when needed instead of just
importing them at the top of the file.  For example, see the call to _load_converter_and_format_options() call in
tests/test_lab_rag_convert.py and the code it calls.
"""

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
