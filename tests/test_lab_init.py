from click.testing import CliRunner
import cli.lab as lab

def test_init():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = CliRunner().invoke(lab.init, ["--interactive", "--taxonomy-path", "taxonomy"])
        assert result.exit_code == 0   

        result = CliRunner().invoke(lab.init)
        assert result.exit_code == 0 