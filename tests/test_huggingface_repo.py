# SPDX-License-Identifier: Apache-2.0

# Standard
import pytest

# First Party
from instructlab.utils import is_huggingface_repo, extract_huggingface_repo_name


@pytest.mark.parametrize(
    "repo_name,expected",
    [
        # Valid cases
        ("user/model", True),
        ("org/model-name", True),
        ("user123/model_v1", True),
        ("user-name/model.v1", True),
        ("user_name/model-v1.2", True),
        ("user.name/model_v1-2", True),
        # URL cases - these should now be handled correctly
        ("https://huggingface.co/user/model", True),
        ("http://huggingface.co/org/model-name", True),
        ("https://www.huggingface.co/user123/model_v1", True),
        
        # Invalid cases
        (None, False),
        ("", False),
        ("model", False),  # Missing owner part
        ("/model", False),  # Empty owner part
        ("user/", False),  # Empty model part
        ("user//model", False),  # Double slash
        ("user model", False),  # Contains space
        ("user@name/model", False),  # Invalid character in owner
        ("user/model@v1", False),  # Invalid character in model
        (123, False),  # Non-string input
        ("https://example.com/user/model", False),  # Not a huggingface URL
        ("huggingface.co/user/model", False),  # Missing http/https protocol
    ],
)
def test_is_huggingface_repo(repo_name, expected):
    """Test is_huggingface_repo function with various valid and invalid inputs."""
    assert is_huggingface_repo(repo_name) == expected


@pytest.mark.parametrize(
    "input_value,expected",
    [
        # URL inputs
        ("https://huggingface.co/user/model", "user/model"),
        ("http://huggingface.co/org/model-name", "org/model-name"),
        ("https://www.huggingface.co/user123/model_v1", "user123/model_v1"),
        ("https://huggingface.co/org/model/tree/main", "org/model/tree/main"),
        
        # Non-URL inputs (should return original)
        ("user/model", "user/model"),
        ("", ""),
        (None, None),
        (123, 123),
        
        # Non-huggingface URLs (should return original)
        ("https://example.com/user/model", "https://example.com/user/model"),
        ("huggingface.co/user/model", "huggingface.co/user/model"),  # Missing protocol
    ],
)
def test_extract_huggingface_repo_name(input_value, expected):
    """Test extract_huggingface_repo_name function for both URL and non-URL inputs."""
    assert extract_huggingface_repo_name(input_value) == expected


def test_is_huggingface_repo_integration():
    """Integration test with a real-world example from the download module."""
    # Common real-world examples
    assert is_huggingface_repo("instructlab/granite-7b-lab-GGUF")
    assert is_huggingface_repo("TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
    assert is_huggingface_repo("prometheus-eval/prometheus-8x7b-v2.0")
    
    # With full URLs
    assert is_huggingface_repo("https://huggingface.co/instructlab/granite-7b-lab-GGUF")
    assert is_huggingface_repo("https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
    assert is_huggingface_repo("https://huggingface.co/prometheus-eval/prometheus-8x7b-v2.0") 