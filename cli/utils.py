# Standard
from typing import Dict, List, Union
import copy
import functools
import glob
import logging
import os
import platform
import re
import shutil
import subprocess
import sys

# Third Party
from langchain.text_splitter import RecursiveCharacterTextSplitter
import click
import git
import gitdb
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s",
)

logger = logging.getLogger(__name__)


def macos_requirement(echo_func, exit_exception):
    """Adds a check for MacOS before running a method.

    :param echo_func: Echo function accepting message and fg parameters to print the error.
    :param exit_exception: Exit exception to raise in case the MacOS requirement is not fulfilled.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_macos_with_m_chip():
                echo_func(
                    message=f"`ilab {func.__name__}` is only implemented for macOS with M-series chips for now",
                    fg="red",
                )
                raise exit_exception(1)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def is_macos_with_m_chip():
    """Checks if the OS is MacOS"""
    if platform.system() != "Darwin":
        return False

    # Check for Apple Silicon (M1, M2, etc.)
    try:
        # Running 'sysctl -a' and searching for a specific line that indicates ARM architecture
        result = subprocess.check_output(["sysctl", "-a"], text=True)
        is_m_chip = "machdep.cpu.brand_string: Apple" in result
        return is_m_chip
    except subprocess.SubprocessError:
        return False


def expand_path(path):
    """Expands ~ and environment variables from the given path"""
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    return path


def make_lab_diff_aliases(cli, diff):
    lab_list = copy.deepcopy(diff)
    lab_list.name = "list"
    lab_list.help = """
    Lists taxonomy files that have changed since <taxonomy-base>.
    Similar to 'git diff <ref>'
    """
    lab_list.deprecated = True

    def lab_list_callback(*args, **kwargs):
        click.secho(
            "DeprecationWarning: Use `ilab diff` instead.",
            fg="red",
        )
        retval = diff.callback(*args, **kwargs)
        return retval

    lab_list.callback = lab_list_callback
    cli.add_command(lab_list)

    lab_check = copy.deepcopy(diff)
    lab_check.name = "check"
    lab_check.help = "Check that taxonomy is valid"
    lab_check.deprecated = True
    # use `--quiet` for current `lab check` behavior
    lab_check.params = lab_check.params[:3]

    def lab_check_callback(*args, **kwargs):
        click.secho(
            "DeprecationWarning: Use `ilab diff --quiet` instead.",
            fg="red",
        )
        retval = diff.callback(*args, **kwargs, quiet=True)
        return retval

    lab_check.callback = lab_check_callback
    cli.add_command(lab_check)


def istaxonomyfile(fn):
    topleveldir = fn.split("/")[0]
    if fn.endswith(".yaml") and topleveldir in ["compositional_skills", "knowledge"]:
        return True
    return False


def get_taxonomy_diff(repo="taxonomy", base="origin/main"):
    repo = git.Repo(repo)
    untracked_files = [u for u in repo.untracked_files if istaxonomyfile(u)]

    branches = [b.name for b in repo.branches]

    head_commit = None
    if "/" in base:
        re_git_branch = re.compile(f"remotes/{base}$", re.MULTILINE)
    elif base in branches:
        re_git_branch = re.compile(f"{base}$", re.MULTILINE)
    else:
        try:
            head_commit = repo.commit(base)
        except gitdb.exc.BadName as exc:
            raise SystemExit(
                yaml.YAMLError(
                    f'Couldn\'t find the taxonomy git ref "{base}" from the current HEAD'
                )
            ) from exc

    # Move backwards from HEAD until we find the first commit that is part of base
    # then we can take our diff from there
    current_commit = repo.commit("HEAD")
    while not head_commit:
        branches = repo.git.branch("-a", "--contains", current_commit.hexsha)
        if re_git_branch.findall(branches):
            head_commit = current_commit
            break
        try:
            current_commit = current_commit.parents[0]
        except IndexError as exc:
            raise SystemExit(
                yaml.YAMLError(
                    f'Couldn\'t find the taxonomy base branch "{base}" from the current HEAD'
                )
            ) from exc

    modified_files = [
        d.b_path
        for d in head_commit.diff(None)
        if not d.deleted_file and istaxonomyfile(d.b_path)
    ]

    updated_taxonomy_files = list(set(untracked_files + modified_files))
    return updated_taxonomy_files


def get_documents(input_pattern: Dict[str, Union[str, List[str]]]) -> List[str]:
    """
    Retrieve the content of files from a Git repository.

    Args:
        input_pattern (dict): Input dictionary containing repository URL, commit hash, and list of file patterns.

    Returns:
         List[str]: List of document contents.
    """ ""

    # Extract input parameters

    repo_url = input_pattern.get("repo")
    commit_hash = input_pattern.get("commit")
    file_patterns = input_pattern.get("pattern")
    temp_dir = os.path.join(os.getcwd(), "temp_repo")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    try:
        # Create a temporary directory to clone the repository
        os.makedirs(temp_dir, exist_ok=True)

        # Clone the repository to the temporary directory
        repo = git.Repo.clone_from(repo_url, temp_dir)

        # Checkout the specified commit
        repo.git.checkout(commit_hash)

        file_contents = []

        # Adjust pattern_path based on file_patterns
        if file_patterns:
            if file_patterns.endswith("/"):  # If file_patterns is just a folder name
                pattern_path = os.path.join(temp_dir, file_patterns, "*.md")
            else:
                pattern_path = os.path.join(temp_dir, file_patterns)
        else:  # If file_patterns is None, consider all .md files from root folder
            pattern_path = os.path.join(temp_dir, "*.md")

        logger.info("Processing files...")
        for file_path in glob.glob(pattern_path):
            if os.path.isfile(file_path) and file_path.endswith(".md"):
                with open(file_path, "r", encoding="utf-8") as file:
                    file_contents.append(file.read())
        repo.close()
        shutil.rmtree(temp_dir)
        return file_contents

    except (OSError, git.exc.GitCommandError, FileNotFoundError) as e:
        logger.error("Error: {}".format(str(e)))
        return [f"Error: {str(e)}"]

    finally:
        # Cleanup: Remove the temporary directory if it exists
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def split_knowledge_docs(documents: List, max_context_size, split_kd_wc) -> List[str]:
    """
    Iterates over large documents and splits them into smaller ones.
    Args:
        document (dict): List of large documents
        max_context_size (int): Defaults to 4096
        split_kd_wc (int): Maximum number of words to include in each document split from a large one.
    Returns:
         List[str]: List of split documents.
    """

    ## We ask the user to input word count, but the actually value we
    ## use is Tokens. Converting the value input by the user, to tokens,
    ## and using that value.

    no_of_tokens = int(split_kd_wc * 1.3)  # 1 word ~ 1.3 token
    if no_of_tokens >= (max_context_size - 1024):
        logger.error("Error: Word count for each doc cannot be more than 2400")
        sys.exit(1)

    split_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=int(no_of_tokens * 4),  # 1 token ~ 4 English character
        chunk_overlap=100,
    )

    for docs in documents:
        temp = text_splitter.create_documents([docs])
        split_docs.extend([item.page_content for item in temp])

    return split_docs
