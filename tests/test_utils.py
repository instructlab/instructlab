# Standard
from pathlib import Path

# Third Party
import pytest

# First Party
from cli.utils import get_skills_dict, skill_path_to_tags


@pytest.fixture
def taxonomy_path(tmp_path):
    p1 = Path(tmp_path / "taxonomy" / "tag1" / "tag2" / "skill1" / "qna.yml")
    p1.parent.mkdir(parents=True)
    p1.touch()
    p2 = Path(tmp_path / "taxonomy" / "tag1" / "tag2" / "skill2" / "qna.yml")
    p2.parent.mkdir(parents=True)
    p2.touch()
    return tmp_path / "taxonomy", p1, p2


def test_skill_path_to_tags():
    tax_path = Path("/home/user/taxonomy")
    skill_path = Path("/home/user/taxonomy/tag1/tag2/skill/qna.yml")

    assert skill_path_to_tags(skill_path, tax_path) == ("tag1", "tag2")

    skill_path = Path("/home/user/taxonomy/tag3/tag4/tag5/skill/qna.yml")
    assert skill_path_to_tags(skill_path, tax_path) == ("tag3", "tag4", "tag5")


def test_get_skills_dict(taxonomy_path):
    # pylint: disable=W0621  # Erroneous detection of redefined-outer-name
    tax_path, p1, p2 = taxonomy_path
    expected_output = {
        "skill1": {
            "path": p1,
            "tags": ("tag1", "tag2"),
        },
        "skill2": {
            "path": p2,
            "tags": ("tag1", "tag2"),
        },
    }
    assert get_skills_dict(tax_path) == expected_output
