# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock, patch

# Third Party
from click.testing import CliRunner
from huggingface_hub.errors import GatedRepoError
from huggingface_hub.utils import HfHubHTTPError

# First Party
from instructlab import lab


class TestLabDownload:
<<<<<<< HEAD
=======
    @pytest.mark.parametrize(
        "repo,provided_token,expected_token,expect_failure",
        [
            ("instructlab/any", "", None, False),
            ("instructlab/any", "mytoken", "mytoken", False),
            ("any/any", "mytoken", "mytoken", False),
            ("any/any", "", None, True),
        ],
    )
    @patch("instructlab.model.download.list_repo_files")
    def test_downloader_handles_token(
        self,
        mock_list_repo_files,
        repo: str,
        provided_token: str,
        expected_token: Union[str, bool, None],
        expect_failure: bool,
    ):
        if expect_failure:
            mock_list_repo_files.side_effect = GatedRepoError("invalid token")

        downloader = HFDownloader(
            repository=repo,
            hf_token=provided_token,
            release="",
            download_dest=Path(""),
            filename="",
            log_level="INFO",
        )

        assert downloader.hf_token == expected_token
        if expect_failure:
            with pytest.raises(ValueError, match="HF_TOKEN"):
                downloader.download()

>>>>>>> d34ba76 (fix: Only require HF_TOKEN when actually needed)
    # When using `from X import Y` you need to understand that Y becomes part
    # of your module, so you should use `my_module.Y`` to patch.
    # When using `import X`, you should use `X.Y` to patch.
    # https://docs.python.org/3/library/unittest.mock.html#where-to-patch?
    @patch("instructlab.model.download.hf_hub_download")
    def test_download(self, mock_hf_hub_download, cli_runner: CliRunner):
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "download",
                "--hf-token=foo",
            ],
        )
        assert (
            result.exit_code == 0
        ), f"command finished with an unexpected exit code. {result.stdout}"
        assert mock_hf_hub_download.call_count == 3

    @patch(
        "instructlab.model.download.hf_hub_download",
        MagicMock(side_effect=HfHubHTTPError("Could not reach hugging face server")),
    )
    def test_download_error(self, cli_runner: CliRunner):
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "download",
            ],
        )
        assert result.exit_code == 1, "command finished with an unexpected exit code"
        assert "Could not reach hugging face server" in result.output

    @patch("instructlab.model.download.OCIDownloader.download")
    def test_oci_download(self, mock_oci_download, cli_runner: CliRunner):
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "download",
                "--repository=docker://quay.io/ai-lab/models/granite-7b-lab",
                "--release=latest",
            ],
        )
        assert result.exit_code == 0
        mock_oci_download.assert_called_once()
        assert "model download completed successfully!" in result.output
        assert "Available models (`ilab model list`):" in result.output

    def test_oci_download_repository_error(self, cli_runner: CliRunner):
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "download",
                "--repository=docker://quay.io/ai-lab/models/granite-7b-lab:latest",
            ],
        )
        assert result.exit_code == 1
        assert (
            "\nInvalid repository supplied:\n    Please specify tag/version 'latest' via --release"
            in result.output
        )
