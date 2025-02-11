# SPDX-License-Identifier: Apache-2.0

# Standard
import argparse
import sys

# Local
from .custom_exceptions import (
    GitWorkflowFilesNotFoundError,
    ExposedSecretsError,
)
from .utils import (
    find_all_yaml_files_in_directory,
    find_exposed_github_secrets_in_env_block,
    find_exposed_github_secrets_in_job,
    get_workflow_trigger_conditions,
    is_git_workflow_file,
    load_workflow_file,
    workflow_auto_triggers_on_pull_request,
)


def create_parser() -> argparse.ArgumentParser:
    """
    Generates a parser for the CLI.

    Returns
    -------
    parser: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="Exposed-Secrets-Detection",
        description="Attempts to find any potentially exposed secrets, and reports them",
    )

    # User can pass in a directory or file, hence they both have "required=False", but one must be chosen
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        help="Directory to read Git workflow files from",
        required=False,
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to the Git workflow file to read",
        required=False,
    )

    return parser


def run_validation(args: dict):
    """
    Runs validation steps to detect potentially-exposed secrets. In the process, this function also validates
    that the user has provided at least "--dir" or "--file" as inputs

    Inputs
    ------
    args: dict
        Dictionary of the user's arguments
    """
    read_dir = args.get("dir")
    file_path = args.get("file")

    if not read_dir and not file_path:
        print(
            "This tool requires either: a path to a valid Git workflow file or a directory that contains Git workflow files. Detected neither."
        )
        sys.exit(1)

    if read_dir and file_path:
        raise NotImplementedError(
            "This tool currently does not support both --dir and --file being passed in simultaneously."
        )

    if read_dir:
        print(
            f"Finding GitHub workflow files under {read_dir} and its sub-directories ..."
        )
        git_workflow_files = {}
        yaml_files = find_all_yaml_files_in_directory(read_dir)

        # For each YAML file we've found, store it in a dictionary for later parsing. We need the yaml file *path* as the
        # key so we can provide actionable error messages if an exception is caught.
        workflow_files_found = 0
        for yaml_file in yaml_files:
            loaded_file = load_workflow_file(yaml_file)

            # Only parse the file if we've confirmed it's a valid Git workflow file and not a random YAML
            if is_git_workflow_file(loaded_file):
                git_workflow_files.update({yaml_file: loaded_file})
                workflow_files_found += 1
                print(f"  - Found and loaded workflow file: {yaml_file}")

        if workflow_files_found == 0:
            raise GitWorkflowFilesNotFoundError(
                f"YAML files were found under directory '{read_dir}', but none were identified as Git workflow files"
            )

    if file_path:
        print(f"Reading Git workflow file {file_path}...")
        loaded_file = load_workflow_file(file_path)

        # Only parse the file if we've confirmed it's a valid Git workflow file and not a random YAML
        if not is_git_workflow_file(loaded_file):
            raise GitWorkflowFilesNotFoundError(
                f"YAML file '{file_path}' is not recognized as a Git workflow file."
            )
        print(f"  - Successfully found and loaded workflow file: {file_path}")
        git_workflow_files = {file_path: loaded_file}

    # For each workflow file, detect if there are exposed secrets
    exposed_secrets_by_file = {}
    print("Reviewing workflow file(s) to find any exposed GitHub secrets...")

    for file_path, file_contents in git_workflow_files.items():
        # This will find any workflow trigger conditions. If none exist, an error is raised.
        trigger_conditions = get_workflow_trigger_conditions(file_path, file_contents)

        # If the workflow doesn't automatically trigger on pull requests, then org/repo maintainers don't have to
        # worry about a bad actor editing the repo contents to retrieve and/or use our GitHub secrets without their
        # knowledge or consent.
        if workflow_auto_triggers_on_pull_request(trigger_conditions) is False:
            filename_without_path = file_path.split("/")[-1]
            print(
                f"[NOTE] The following workflow does not automatically trigger on pull requests, so ignoring: {filename_without_path}"
            )
            continue

        # Check if "env" exists at the top level and find any secrets
        top_level_env = file_contents.get("env", {})
        exposed_secrets = find_exposed_github_secrets_in_env_block(top_level_env)

        # If we've found any exposed secrets, keep track, but don't throw an error. Let's find all errors for
        # the user first.
        if len(exposed_secrets) > 0:
            print(
                f"[WARNING] Detected exposed top-level env secrets: {exposed_secrets}"
            )
            exposed_secrets_by_file.update(
                {
                    "filename": file_path,
                    "job_name": "n/a",
                    "exposed_secrets": exposed_secrets,
                }
            )

        # It's possible that someone's workflow file is a WIP and doesn't have "jobs" defined yet, so don't throw
        # an error in this case!
        jobs = file_contents.get("jobs", {})
        for job_name, job_def in jobs.items():
            exposed_secrets = find_exposed_github_secrets_in_job(job_def)

            # Same as above. If we find an exposed secret, keep track, but don't fail until the end.
            if len(exposed_secrets) > 0:
                print(
                    f"[WARNING] Detected exposed secrets for job '{job_name}': {exposed_secrets}"
                )
                exposed_secrets_by_file.update(
                    {
                        "filename": file_path,
                        "job_name": job_name,
                        "exposed_secrets": exposed_secrets,
                    }
                )

        if len(exposed_secrets_by_file) > 0:
            raise ExposedSecretsError(
                f"Detected one or more exposed secrets. Please review the findings below, and if you "
                f"feel they have been made in error, please file a GitHub issue and link to this job "
                f"URL. Findings: {exposed_secrets_by_file}"
            )

    print(
        "Hooray! There is no concern about exposed secrets in the environment! Carry on..."
    )
