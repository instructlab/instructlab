# Standard
from unittest import mock
from unittest.mock import patch
import pathlib
import socket
import sys

# Third Party
from click.testing import CliRunner
import pytest

# First Party
from instructlab import lab
from instructlab.model.backends import backends
from instructlab.model.backends.vllm import build_vllm_cmd


def test_free_port():
    host = "localhost"
    port = backends.free_tcp_ipv4_port(host)
    # check that port is bindable
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))


# the test fails because the model_path is not a valid GGUF file
def test_get_backend_auto_detection_fail_not_gguf(tmp_path: pathlib.Path):
    tmp_gguf = tmp_path / "test.gguf"
    # Write a known invalid header
    invalid_header = (
        b"\x00\x00\x00\x00"  # Use a header that will definitely not match "GGUF_MAGIC"
    )
    tmp_gguf.write_bytes(
        invalid_header + bytes([0] * 4093)
    )  # Fill the rest of the file with zeros
    with pytest.raises(ValueError) as exc_info:
        backends.get(tmp_gguf, None)
    assert "is not a GGUF format" in str(exc_info.value)


# this test succeeds because the model_path is a valid GGUF file (is_model_gguf mocked to returns True)
@patch("instructlab.model.backends.backends.is_model_gguf", return_value=True)
def test_get_backend_auto_detection_success_gguf(
    m_is_model_gguf, tmp_path: pathlib.Path
):
    tmp_gguf = tmp_path / "test.gguf"
    backend = backends.get(tmp_gguf, None)
    assert backend == "llama-cpp"
    m_is_model_gguf.assert_called_once_with(tmp_gguf)


# this test fails because the model_path is a directory but the platform is not linux
@patch("sys.platform", "darwin")
def test_get_backend_auto_detection_failure_vllm_dir_but_not_linux(
    tmp_path: pathlib.Path,
):
    with pytest.raises(ValueError) as exc_info:
        backends.get(tmp_path, None)
    assert "Cannot determine which backend to use" in str(exc_info.value)


# this test succeeds because the model_path is a directory and the platform is linux so vllm is
# picked
@patch("sys.platform", "linux")
def test_get_backend_auto_detection_success_vllm_dir(tmp_path: pathlib.Path):
    backend = backends.get(tmp_path, None)
    assert backend == "vllm"


# this test fails because the OS is darwin and a directory is passed, only works on Linux
@patch("sys.platform", "darwin")
def test_get_backend_auto_detection_failed_vllm_dir_darwin(tmp_path: pathlib.Path):
    with pytest.raises(ValueError) as exc_info:
        backends.get(tmp_path, None)
    assert (
        "Model is a directory containing huggingface safetensors files but the system is not Linux."
        in str(exc_info.value)
    )


# this test succeeds even if the auto-detection picked a different backend, we continue with what
# the user requested
@patch(
    "instructlab.model.backends.backends.determine_backend",
    return_value=("vllm", "reason for selection"),
)
def test_get_forced_backend_fails_autodetection(m_determine_backend):
    backend = backends.get("", "llama-cpp")
    assert backend == "llama-cpp"
    m_determine_backend.assert_called_once()


@mock.patch("instructlab.model.backends.vllm.Server")
@mock.patch("instructlab.model.backends.backends.get", return_value=backends.VLLM)
def test_ilab_vllm_args(
    m_backends_get: mock.Mock,
    m_server: mock.Mock,
    tmp_path: pathlib.Path,
    cli_runner: CliRunner,
):
    cmd = [
        "--config",
        "DEFAULT",
        "model",
        "serve",
        "--model-path",
        str(tmp_path),
        "--backend",
        backends.VLLM,
        "--",
        "--enable_lora",
    ]
    result = cli_runner.invoke(lab.ilab, cmd)
    assert result.exit_code == 0, result.stdout
    m_backends_get.assert_called_once_with(tmp_path, backends.VLLM)
    m_server.assert_called_once_with(
        api_base="http://127.0.0.1:8000/v1",
        chat_template=None,
        model_family=None,
        model_path=tmp_path,
        vllm_args=["--enable_lora"],
        host="127.0.0.1",
        port=8000,
    )


@mock.patch("instructlab.model.backends.llama_cpp.Server")
@mock.patch("instructlab.model.backends.backends.get", return_value=backends.LLAMA_CPP)
def test_ilab_llama_cpp_args(
    m_backends_get: mock.Mock, m_server: mock.Mock, cli_runner: CliRunner
):
    gguf = pathlib.Path("test.gguf")
    cmd = [
        "--config",
        "DEFAULT",
        "model",
        "serve",
        "--model-path",
        str(gguf),
        "--backend",
        backends.LLAMA_CPP,
    ]
    result = cli_runner.invoke(lab.ilab, cmd)
    assert result.exit_code == 0, result.stdout
    m_backends_get.assert_called_once_with(gguf, backends.LLAMA_CPP)
    m_server.assert_called_once_with(
        api_base="http://127.0.0.1:8000/v1",
        model_path=gguf,
        model_family=None,
        host="127.0.0.1",
        port=8000,
        gpu_layers=-1,
        max_ctx_size=4096,
        num_threads=None,
        chat_template=None,
    )


def test_build_vllm_cmd_with_defaults(tmp_path: pathlib.Path):
    host = "localhost"
    port = 8080
    model_path = pathlib.Path("/path/to/model")
    model_family = ""
    chat_template = tmp_path / "chat_template.jinja2"
    chat_template.touch()
    vllm_args: list[str] = []
    expected_cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        str(model_path),
        "--chat-template",
        str(chat_template),
        "--distributed-executor-backend",
        "mp",
    ]
    cmd, _ = build_vllm_cmd(
        host, port, model_family, model_path, str(chat_template), vllm_args
    )
    assert cmd == expected_cmd


def test_build_vllm_cmd_with_args_provided(tmp_path: pathlib.Path):
    host = "localhost"
    port = 8080
    model_path = pathlib.Path("/path/to/model")
    model_family = ""
    chat_template = tmp_path / "chat_template.jinja2"
    chat_template.touch()
    vllm_args = [
        "--port",
        str(8001),
        "--model=/path/to/other/model",
        "--distributed-executor-backend",
        "ray",
    ]
    expected_cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        host,
        "--chat-template",
        str(chat_template),
    ] + vllm_args

    cmd, _ = build_vllm_cmd(
        host, port, model_family, model_path, str(chat_template), vllm_args
    )
    assert cmd == expected_cmd
