# pylint: disable=W0621 redefined-outer-name
# First party
# Standard
import pathlib
import random
import typing

# Third Party
import click

# Third party
import pytest

# First Party
import instructlab.model.train as model_train

FakeEvalFunc = typing.Callable[..., float]


@pytest.fixture
def dir_of_checkpoints(tmp_path: pathlib.Path) -> pathlib.Path:
    for dir_name in ["fakedir1", "fakedir2"]:
        (tmp_path / dir_name).mkdir()

    return tmp_path


@pytest.fixture
def dir_of_checkpoints_and_a_file(
    dir_of_checkpoints: pathlib.Path,
) -> pathlib.Path:
    (dir_of_checkpoints / "file.txt").write_text("something")
    return dir_of_checkpoints


@pytest.fixture
def fake_eval_func() -> FakeEvalFunc:
    return lambda model: random.randrange(10) * 1.0


def test_clickpath_setup() -> None:
    """State of output click.Path object should be consistent"""

    path: click.Path = model_train.clickpath_setup(is_dir=False)
    assert not path.dir_okay
    assert path.file_okay
    assert path.exists
    assert path.resolve_path

    path = model_train.clickpath_setup(is_dir=True)
    assert path.dir_okay
    assert not path.file_okay
    assert path.exists
    assert path.resolve_path


def test_evaluate_dir_of_checkpoints(
    dir_of_checkpoints: pathlib.Path, fake_eval_func: FakeEvalFunc
) -> None:
    """Iterate through fake checkpoints dir. Should return a float and Path"""

    score = model_train._evaluate_dir_of_checkpoints(
        checkpoints_dir=dir_of_checkpoints, eval_func=fake_eval_func
    )

    assert isinstance(score[0], float)
    assert isinstance(score[1], pathlib.Path)


def test_evaluate_empty_dir_of_checkpoints(
    tmp_path: pathlib.Path, fake_eval_func: FakeEvalFunc
) -> None:
    """Iterate through empty dir. Should raise RuntimeError"""

    with pytest.raises(RuntimeError):
        _ = model_train._evaluate_dir_of_checkpoints(
            checkpoints_dir=tmp_path, eval_func=fake_eval_func
        )


def test_evaluate_empty_dir_of_checkpoints_and_a_file(
    dir_of_checkpoints_and_a_file: pathlib.Path,
    fake_eval_func: FakeEvalFunc,
) -> None:
    """Iterate through fake dir of checkpoints and a file. Should succeed without problem."""

    score = model_train._evaluate_dir_of_checkpoints(
        checkpoints_dir=dir_of_checkpoints_and_a_file, eval_func=fake_eval_func
    )

    assert isinstance(score[0], float)
    assert isinstance(score[1], pathlib.Path)


@pytest.fixture
def phase_dir_directory_names() -> list[str]:
    return ["checkpoints", "eval_cache"]


@pytest.fixture
def phases_directory_names() -> list[str]:
    return ["phase1", "phase2"]


@pytest.fixture
def phasedir_already_initialized(
    tmp_path: pathlib.Path, phase_dir_directory_names: list[str]
) -> pathlib.Path:
    for dirname in phase_dir_directory_names:
        (tmp_path / dirname).mkdir()

    return tmp_path


def test_setup_phase_dirs(
    tmp_path: pathlib.Path, phase_dir_directory_names: list[str]
) -> None:
    """_setup_phase_dirs should create two dirs, checkpoints and eval_cache"""
    model_train._setup_phase_dirs(path=tmp_path)

    for dirname in phase_dir_directory_names:
        assert (tmp_path / dirname).exists()
        assert (tmp_path / dirname).is_dir()


def test_setup_phasedirs_already_initialized(
    phasedir_already_initialized: pathlib.Path,
):
    """if phasedir already has directories in it, should fail. _setup_phase_dirs expects empty dir right now"""
    with pytest.raises(FileExistsError):
        model_train._setup_phase_dirs(phasedir_already_initialized)


def test_prepare_phased_base_dir(
    tmp_path: pathlib.Path, phases_directory_names: list[str]
) -> None:
    """_prepare_phased_base_dir should create two dirs, phase1 and phase2"""
    model_train._prepare_phased_base_dir(phased_base_dir=tmp_path)

    for dirname in phases_directory_names:
        assert (tmp_path / dirname).exists()
        assert (tmp_path / dirname).is_dir()
        assert len(list(tmp_path.iterdir())) == len(phases_directory_names)
