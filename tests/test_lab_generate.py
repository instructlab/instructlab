# Standard

# Third Party
from click.testing import CliRunner

# Local
from . import common


def test_vllm_args_null(cli_runner: CliRunner):
    fname = common.setup_gpus_config(
        section_path="generate.teacher", vllm_args=lambda: None
    )
    args = common.vllm_setup_test(
        cli_runner,
        [
            f"--config={fname}",
            "data",
            "generate",
            "--gpus",
            "4",
        ],
    )
    common.assert_tps(args, "4")
