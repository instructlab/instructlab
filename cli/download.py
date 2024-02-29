# Standard
import os
import re
import subprocess


class DownloadException(Exception):
    """An exception raised during downloading artifacts necessary to run lab."""


def download_model(
    gh_repo="https://github.com/open-labrador/cli.git",
    gh_release="latest",
    model_dir=".",
    pattern="",
):
    """
    Download and combine the model file from a GitHub repository source.

    Parameters:
    - gh_repo (str): The URL of the GitHub repository containing the model.
        Default: Open Labrador CLI repository.
    - gh_release (str): The GitHub release version of the model to download.
        Default is 'latest'.
    - model_dir(str): The local directory to download the model files into
    - pattern(str): Download only assets that match a glob pattern

    Returns:
    - a list of downloaded models
    """

    model_file_split_keyword = ".split."

    # Download GitHub release
    download_commands = [
        "gh",
        "release",
        "download",
        gh_release,
        "--repo",
        gh_repo,
        "--dir",
        model_dir,
    ]
    if pattern != "":
        download_commands.extend(["--pattern", pattern])
    if gh_release == "latest":
        download_commands.pop(3)  # remove gh_release arg to download the latest version
        if (
            pattern == ""
        ):  # Latest release needs to specify the pattern argument to download files
            download_commands.extend(["--pattern", "*"])
    try:
        create_subprocess(download_commands)
    except FileNotFoundError as exc:
        raise DownloadException("`gh` binary not found") from exc
    except subprocess.CalledProcessError as exc:
        raise DownloadException(f"error invoking `gh` command: {exc}") from exc

    # Get the list of local files
    ls_commands = ["ls", model_dir]
    ls_result = create_subprocess(ls_commands)
    file_list = ls_result.stdout.decode("utf-8").split("\n")
    file_list = [os.path.join(model_dir, f) for f in file_list]

    splitted_models = {}

    # Use Regex to capture model and files names that need to combine into one file.
    for file in file_list:
        if model_file_split_keyword in file:
            model_name_regex = "".join(
                ["([^ \t\n,]+)", model_file_split_keyword, "[^ \t\n,]+"]
            )
            regex_result = re.findall(model_name_regex, file)
            if regex_result:
                model_name = regex_result[0]
                if splitted_models.get(model_name):
                    splitted_models[model_name].append(file)
                else:
                    splitted_models[model_name] = [file]

    # Use the model and file names we captured to minimize the number of bash subprocess creations.
    combined_model_list = []
    for key, value in splitted_models.items():
        cat_commands = ["cat"]
        splitted_model_files = list(value)
        cat_commands.extend(splitted_model_files)
        cat_result = create_subprocess(cat_commands)
        if cat_result.stdout not in (None, b""):
            with open(key, "wb") as model_file:
                model_file.write(cat_result.stdout)
            rm_commands = ["rm"]
            rm_commands.extend(splitted_model_files)
            create_subprocess(rm_commands)
            combined_model_list.append(key)

    return combined_model_list


def clone_taxonomy(
    repository="https://github.com/open-labrador/taxonomy.git",
    branch="main",
    filter_spec="",
):
    """
    Clone the taxonomy repository from a Git repository source.

    Parameters:
    - gh_repo (str): The URL of the taxonomy Git repository.
        Default is the Open Labrador taxonomy repository.
    - gh_branch (str): The GitHub branch of the taxonomy repository. Default is main
    - filter_spec(str): Optional path to the git filter spec for git partial clone

    Returns:
    - None
    """
    # Clone taxonomy repo
    git_clone_commands = ["git", "clone", repository]
    if filter_spec != "" and os.path.exists(filter_spec):
        # TODO: Add gitfilterspec to sparse clone GitHub repo
        git_filter_arg = "".join(["--filter=sparse:oid=", branch, ":", filter_spec])
        git_sparse_clone_flags = ["--sparse", git_filter_arg]
        git_clone_commands.extend(git_sparse_clone_flags)
    else:
        git_clone_commands.extend(["--branch", branch])
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
