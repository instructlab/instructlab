# Standard
from unittest import mock

# Third Party
from click.testing import CliRunner

# First Party
from instructlab import lab
from instructlab.configuration import DEFAULTS


@mock.patch("openai.OpenAI")
@mock.patch("httpx.Client")
@mock.patch("instructlab.model.backends.backends.select_backend")
@mock.patch("instructlab.sdg.generate_data.generate_data")
def test_generate_cert_params(
    m_generate_data: mock.Mock,
    m_select_backend: mock.Mock,
    m_client: mock.Mock,
    m_openai: mock.Mock,
    cli_runner: CliRunner,
) -> None:
    endpoint = "https://endpoint.test/v1"
    m_select_backend().run_detached.return_value = endpoint

    args = [
        "--config=DEFAULT",
        "data",
        "generate",
    ]
    result = cli_runner.invoke(lab.ilab, args)
    assert result.exit_code == 0, result.stdout
    m_client.assert_called_once_with(cert=None, verify=True)
    m_openai.assert_called_once_with(
        base_url="https://endpoint.test/v1",
        api_key=DEFAULTS.API_KEY,
        http_client=m_client(),
    )
    m_generate_data.assert_called()

    insecure_args = args + [
        "--endpoint-insecure",
        "--endpoint-client-cert=/path/to/cert.pem",
        "--endpoint-client-key=/path/to/cert.key",
    ]
    result = cli_runner.invoke(lab.ilab, insecure_args)
    assert result.exit_code == 0, result.stdout
    m_client.assert_called_with(
        cert=("/path/to/cert.pem", "/path/to/cert.key", None),
        verify=False,
    )

    cacert_args = args + [
        "--endpoint-ca-cert=/path/to/ca.pem",
    ]
    result = cli_runner.invoke(lab.ilab, cacert_args)
    assert result.exit_code == 0, result.stdout
    m_client.assert_called_with(
        cert=None,
        verify="/path/to/ca.pem",
    )
