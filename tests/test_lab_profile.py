# Third Party
from click.testing import CliRunner

# First Party
from instructlab import lab


def test_profile_list():
    runner = CliRunner()
    result = runner.invoke(
        lab.ilab,
        [
            "profile",
            "list",
        ],
    )
    assert "apple" in result.output
    assert "intel" in result.output
    assert "amd" in result.output
    assert "nvidia" in result.output
