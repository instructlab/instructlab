# SPDX-License-Identifier: Apache-2.0

# Standard
import os
import re

# Third-Party
import yaml

# Local
from .custom_exceptions import (
    GitWorkflowFilesNotFoundError,
    GitWorkflowFilesSearchError,
    MissingTriggerConditionsError,
)


def load_workflow_file(workflow_filename: str) -> dict:
    """
    Loads `workflow_filename` into a dictionary object

    Inputs
    ------
    workflow_filename: str
        The name of the YAML workflow file to load

    Returns
    -------
    workflow_file: dict
        The loaded workflow file, in dictionary format
    """
    workflow_file = None
    try:
        with open(workflow_filename, "r") as file:
            workflow_file = yaml.safe_load(file)

    except FileNotFoundError as file_nf_err:
        print(
            f"Failed to find workflow file: {workflow_filename}. This is an internal error. "
            "Please file a GitHub issue if you encounter this error and link to this failed "
            "job. Most likely, an incorrect filename was referenced due to a typo."
        )
        raise file_nf_err

    except OSError as os_err:
        print(
            f"Unable to open/read workflow file: {workflow_filename}. This is an internal error. "
            "Please file a GitHub issue if you encounter this error and link to this failed "
            "job. It is possible the workflow file is corrupted or a RAM issue was encountered."
        )
        raise os_err

    except yaml.YAMLError as yml_err:
        print(
            f"Unable to open/read workflow file: {workflow_filename}. Most likely, this error "
            "occurred because the format of your YAML is invalid. Please ensure your YAML file is "
            "properly formatted and try again. However, if you have not modified a YAML file in "
            "your pull request, please file a GitHub issue and link to this failed job."
        )
        raise yml_err

    except Exception as ex:
        print(
            f"An unknown exception occurred when attempting to load {workflow_filename}. "
            "This is an internal error. Please file a GitHub issue if you encounter this error, "
            "link to this failed job, and provide the error message printed to the console below."
        )
        raise ex

    return workflow_file


def workflow_auto_triggers_on_pull_request(trigger_conditions: list) -> bool:
    """
    Detects if a workflow automatically runs on a pull request by default, or requires
    a maintainer to manually run the job. Some jobs may allow for a combination of manual
    and automatic triggers, but if any automatic triggers exist, then we return False.

    Inputs
    ------
    trigger_conditions: list
        List of workflow triggers. For example: ["schedule", "workflow_dispatch", "pull_request"]

    Returns
    -------
    True/False: bool
       Returns "True" if the GitHub workflow file can automatically trigger on PR builds
       without any form of pre-approval from a maintainer.
    """
    # These triggers are often used in workflow files to automatically trigger on certain pull
    # request/issue actions.
    automatic_triggers = {
        "issue_comment",  # this can sometimes require a GitHub token stored in an env var
        "pull_request",
        "pull_request_comment",  # this is deprecated and replaced by "issue_comment", but GitHub still allows it to be used
        "pull_request_review",
        "pull_request_review_comment",
        "pull_request_target",
    }

    # If even one of the above triggers is found in the given workflow file, then we must review the
    # whole workflow to find potential exposed secrets.
    return bool(set(trigger_conditions) & automatic_triggers)


def find_exposed_github_secrets_in_env_block(env_block: dict) -> dict:
    """
    Finds any exposed GitHub secrets in an `env` block

    Inputs
    ------
    env_block: dict
        The `env` block to parse

    Returns
    -------
    exposed_secrets: dict
        A dictionary which contains the env var names that have secrets, but does
        not contain the secrets themselves.
    """
    # GitHub secrets follow the form: "${{ secrets.SOME_NAME }}"
    secrets_regexp = re.compile(r"\$\{\{.*secrets.*\}\}")

    # Contains all detected exposed GitHub secrets
    exposed_secrets = {}

    for var, value in env_block.items():
        if secrets_regexp.search(value):
            print(f"WARNING: Detected exposed GitHub secret: {var}={value}")
            exposed_secrets.update({var: value})

    return exposed_secrets


def find_exposed_github_secrets_in_job(job: dict) -> dict:
    """
    Finds all `env` block definitions in a job within a Git workflow file, then
    determines if any secrets exist.

    Inputs
    ------
    job: dict
        The job configuration/definition

    Returns
    -------
    exposed_secrets: dict
        A dictionary which contains the env var names that have secrets, but does
        not contain the secrets themselves.
    """
    # Contains all detected exposed GitHub secrets
    exposed_secrets = {}

    steps = job.get("steps", [])
    for job_step in steps:
        env_block = job_step.get("env", {})
        env_block_exposed_secrets = find_exposed_github_secrets_in_env_block(env_block)
        exposed_secrets.update(env_block_exposed_secrets)

    return exposed_secrets


def find_all_yaml_files_in_directory(dirname: str) -> list[str]:
    """
    Finds all yaml files in a given directory and its sub-directories

    Inputs
    ------
    dirname: str
        Directory to search through

    Returns
    -------
    yaml_files: list
        List of the full paths to the YAML files found (if any)
    """
    yaml_files = []

    # This theoretically should never throw an exception, even if the directory to search
    # through does not exist, but a try-except block is added for safety.
    try:
        for root, _, files in os.walk(dirname):
            for filename in files:
                if filename.endswith((".yaml", ".yml")):
                    yaml_files.append(root + os.sep + filename)
    except Exception as ex:
        raise GitWorkflowFilesSearchError(
            f"An unknown error occurred when trying to walk through directory: {dirname}. If "
            f"you think this is error is incorrect, please file a GitHub issue and provide a "
            f"link to this failing job alongside a screenshot. Error message returned: {ex}"
        )

    if len(yaml_files) == 0:
        raise GitWorkflowFilesNotFoundError(
            f"No YAML files were found under '{dirname}' or any of its sub-directories."
        )

    return yaml_files


def is_git_workflow_file(yaml_file: dict) -> bool:
    """
    Determines if the provided YAML file is a workflow file or not

    Inputs
    ------
    yaml_file: dict
        YAML file that has been loaded and stored as a dictionary

    Returns
    -------
    True/False
    """
    # All workflow files need the "name" and "jobs" field at a bare minimum
    return bool("name" in yaml_file and "jobs" in yaml_file)


def get_workflow_trigger_conditions(
    workflow_file_path: str, workflow_file_contents: dict
) -> list[str]:
    """
    Gets the list of trigger conditions for a given workflow.

    Inputs
    ------
    workflow_file_path: str
        Path to the workflow file name. (Mainly used for error handling.)
    workflow_file_contents: dict
        Git workflow file contents that were loaded from a YAML file

    Returns
    -------
    trigger_conditions: list
        List of the trigger conditions, but not their configs.
    """
    # Note that because GitHub workflow files use the key "on" to denote trigger conditions,
    # sometimes YAML parsing automatically converts "on" to a boolean value (i.e., "True"
    # in this case). So we have to check for both conditions.
    trigger_conditions_dict = workflow_file_contents.get("on", {})
    if len(trigger_conditions_dict) == 0:
        trigger_conditions_dict = workflow_file_contents.get(True, {})

    # Sometimes, users may make pull requests that are a WIP and presently missing trigger conditions,
    # but we should still flag it anyway. We don't want anything flying under the radar.
    if len(trigger_conditions_dict) == 0:
        raise MissingTriggerConditionsError(
            f"Could not identify trigger conditions for Git workflow file: {workflow_file_path}. If you "
            f"are currently working on updating this workflow in a pull request, then please add one or "
            f"more valid trigger conditions to your workflow file. If you are NOT updating this "
            f"workflow file and you feel this error message is incorrect, please file a GitHub issue "
            f"and link to this job URL, as well as provide a screenshot."
        )

    trigger_conditions = list(trigger_conditions_dict.keys())
    return trigger_conditions
