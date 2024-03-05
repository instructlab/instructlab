# Standard
import subprocess


class DownloadException(Exception):
    """An exception raised during downloading artifacts necessary to run lab."""


def clone_taxonomy(
    gh_repo="https://github.com/instruct-lab/taxonomy.git",
    gh_branch="main",
    directory="taxonomy",
    min_taxonomy=False,
):
    """
    Clone the taxonomy repository from a Git repository source.

    Parameters:
    - repository (str): URL of the taxonomy git repository.
        Default is the Intruct Lab taxonomy repository.
    - gh_branch (str): The GitHub branch of the taxonomy repository. Default is main
    - directory (str): Target directory where to clone the repository. Default is taxonomy.
    - min_taxonomy(bool): Shallow clone the taxonomy repository with minimum size.

    Returns:
    - None
    """
    # Clone taxonomy repo
    git_clone_commands = ["git", "clone", gh_repo, "--branch", gh_branch]
    if min_taxonomy:
        git_clone_commands.append("--depth=1")
    git_clone_commands.extend([directory])

    try:
        create_subprocess(git_clone_commands)
    except FileNotFoundError as exc:
        raise DownloadException("`git` binary not found") from exc
    except subprocess.CalledProcessError as exc:
        raise DownloadException("error cloning {repository}@{branch}: {exc}") from exc


def create_subprocess(commands):
    return subprocess.run(
        commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
