# Standard
from unittest.mock import patch
import pathlib
import socket

# Third Party
import pytest

# First Party
from instructlab.model.backends import backends


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
