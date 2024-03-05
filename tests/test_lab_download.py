from click.testing import CliRunner
import cli.lab as lab

def test_lab_download():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = CliRunner().invoke(lab.init, ["--interactive"])
        assert result.exit_code == 0  
        result = CliRunner().invoke(lab.download, ["--hf-endpoint", "http://localhost:8080/"])
        assert result.exit_code == 0   
    