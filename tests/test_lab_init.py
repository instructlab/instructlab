from click.testing import CliRunner
import cli.lab as lab

def test_init():
    runner = CliRunner()
    lab_test = lab
    with runner.isolated_filesystem():
        result = CliRunner().invoke(lab_test.init, ["--interactive", "--taxonomy-path", "taxonomy"])
        assert result.exit_code == 0   

        result = CliRunner().invoke(lab.init)
        assert result.exit_code == 0 