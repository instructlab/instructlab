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


@pytest.mark.usefixtures("mock_supported_backends")
class TestValidateBackend:
    def test_validate_backend_valid(self):
        # Test with a valid backend
        try:
            backends.validate_backend("llama-cpp")
        except ValueError:
            pytest.fail("validate_backend raised ValueError unexpectedly!")
        # Test with a valid backend
        try:
            backends.validate_backend("LLAMA-CPP")
        except ValueError:
            pytest.fail("validate_backend raised ValueError unexpectedly!")
        # Test with a valid backend
        try:
            backends.validate_backend("vllm")
        except ValueError:
            pytest.fail("validate_backend raised ValueError unexpectedly!")

    def test_validate_backend_invalid(self):
        # Test with an invalid backend
        with pytest.raises(ValueError):
            backends.validate_backend("foo")
