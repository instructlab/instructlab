# Standard
import socket

# Third Party
import pytest

# First Party
from instructlab.model.backends import backends

supported_backends = ["llama-cpp", "vllm"]  # Example supported backends


@pytest.fixture
def mock_supported_backends(monkeypatch):
    monkeypatch.setattr(
        "instructlab.model.backends.backends.SUPPORTED_BACKENDS", supported_backends
    )


def test_free_port():
    host = "localhost"
    port = backends.free_tcp_ipv4_port(host)
    # check that port is bindable
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
