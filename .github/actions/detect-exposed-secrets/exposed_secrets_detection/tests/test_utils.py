# SPDX-License-Identifier: Apache-2.0

# Third-Party
import pytest
from yaml import YAMLError

# Local
from src.utils import (
    find_all_yaml_files_in_directory,
    find_exposed_github_secrets_in_env_block,
    find_exposed_github_secrets_in_job,
    get_workflow_trigger_conditions,
    is_git_workflow_file,
    load_workflow_file,
    workflow_auto_triggers_on_pull_request,
)
from src.custom_exceptions import (
    GitWorkflowFilesNotFoundError,
    MissingTriggerConditionsError,
)


def test_load_workflow_file_fails_because_file_does_not_exist():
    with pytest.raises(
        FileNotFoundError,
        match=r"No such file or directory: 'non-existent-file'",
    ):
        load_workflow_file("non-existent-file")


def test_load_workflow_file_fails_for_malformed_file():
    with pytest.raises(YAMLError):
        load_workflow_file("tests/test_data/git_workflow_invalid_yaml.yml")


def test_load_workflow_file_for_valid_workflow():
    load_workflow_file("tests/test_data/git_workflow_manual_trigger.yml")


def test_get_workflow_trigger_conditions_for_valid_workflow_file():
    test_workflow_file_path = "some-workflow.yml"
    valid_workflow_file = {
        "name": "Valid fake E2E Job",
        "on": {
            "schedule": [{"cron": "0 11 * * *"}],
            "workflow_dispatch": {
                "inputs": {
                    "pr_or_branch": {
                        "description": "pull request number or branch name",
                        "required": True,
                        "default": "main",
                    },
                },
            },
        },
        "jobs": {},  # doesn't matter if it's empty for this test
    }
    trigger_conditions = get_workflow_trigger_conditions(
        test_workflow_file_path, valid_workflow_file
    )
    expected_trigger_conditions = ["schedule", "workflow_dispatch"]

    assert len(trigger_conditions) == 2
    assert sorted(trigger_conditions) == sorted(expected_trigger_conditions)


def test_get_workflow_trigger_conditions_for_valid_workflow_file_with_yaml_parse_conversion_to_True():
    test_workflow_file_path = "some-workflow.yml"
    valid_workflow_file = {
        "name": "Valid fake E2E Job",
        True: {
            "schedule": [{"cron": "0 11 * * *"}],
            "workflow_dispatch": {
                "inputs": {
                    "pr_or_branch": {
                        "description": "pull request number or branch name",
                        "required": True,
                        "default": "main",
                    },
                },
            },
        },
        "jobs": {},  # doesn't matter if it's empty for this test
    }
    trigger_conditions = get_workflow_trigger_conditions(
        test_workflow_file_path, valid_workflow_file
    )
    expected_trigger_conditions = ["schedule", "workflow_dispatch"]

    assert len(trigger_conditions) == 2
    assert sorted(trigger_conditions) == sorted(expected_trigger_conditions)


def test_get_workflow_trigger_conditions_from_workflow_missing_trigger_conditions():
    test_workflow_file_path = "some-workflow.yml"
    invalid_workflow_file = {
        "name": "Invalid E2E job missing trigger conditions",
        "jobs": {},  # doesn't matter if it's empty for this test
    }
    with pytest.raises(MissingTriggerConditionsError):
        get_workflow_trigger_conditions(test_workflow_file_path, invalid_workflow_file)


def test_workflow_auto_triggers_on_pull_request_no_automatic_triggers_exist():
    trigger_conditions = [
        "schedule",
        "workflow_dispatch",
    ]
    workflow_contains_auto_triggers = workflow_auto_triggers_on_pull_request(
        trigger_conditions
    )
    assert workflow_contains_auto_triggers == False  # noqa: E712


def test_workflow_auto_triggers_on_pull_request_one_automatic_trigger_exists():
    trigger_conditions = [
        "schedule",
        "workflow_dispatch",
        "pull_request",
    ]
    workflow_contains_auto_triggers = workflow_auto_triggers_on_pull_request(
        trigger_conditions
    )
    assert workflow_contains_auto_triggers == True  # noqa: E712


def test_find_exposed_secrets_in_env_block_empty():
    env_block = {}
    exposed_secrets = find_exposed_github_secrets_in_env_block(env_block)
    assert len(exposed_secrets) == 0


def test_find_exposed_secrets_in_env_block_no_github_secrets():
    env_block = {"VAR1": "value1", "VAR2": "value2"}
    exposed_secrets = find_exposed_github_secrets_in_env_block(env_block)
    assert len(exposed_secrets) == 0


def test_find_exposed_secrets_in_env_block_found_one_secret():
    env_block = {
        "VAR1": "value1",
        "VAR2": "value2",
        "SECRET_TOKEN": "${{ secrets.SECRET_TOKEN }}",
    }
    exposed_secrets = find_exposed_github_secrets_in_env_block(env_block)
    expected_exposed_secrets = {
        "SECRET_TOKEN": "${{ secrets.SECRET_TOKEN }}",
    }

    assert len(exposed_secrets) == 1
    assert exposed_secrets == expected_exposed_secrets


def test_find_exposed_secrets_in_env_block_found_mulitple_secrets():
    env_block = {
        "VAR1": "value1",
        "VAR2": "value2",
        "SECRET_TOKEN": "${{ secrets.SECRET_TOKEN }}",
        "SECRET_KEY": "${{ secrets.SECRET_KEY }}",
        "SECRET_PASSWORD": "${{ secrets['SECRET_PASSWORD'] }}",
    }
    exposed_secrets = find_exposed_github_secrets_in_env_block(env_block)
    expected_exposed_secrets = {
        "SECRET_TOKEN": "${{ secrets.SECRET_TOKEN }}",
        "SECRET_KEY": "${{ secrets.SECRET_KEY }}",
        "SECRET_PASSWORD": "${{ secrets['SECRET_PASSWORD'] }}",
    }

    assert len(exposed_secrets) == 3
    assert exposed_secrets == expected_exposed_secrets


def test_find_exposed_github_secrets_in_job_no_env_vars():
    job_with_no_env_vars = {
        "start-fake-ec2-runner": {
            "outputs": {
                "label": "${{ steps.start-ec2-runner.outputs.label }}",
                "ec2-instance-id": "${{ steps.start-ec2-runner.outputs.ec2-instance-id }}",
            },
            "runs-on": "ubuntu-latest",
            "steps": [
                {
                    "name": "Step 1",
                    "uses": "aws-actions/configure-aws-credentials@4fc4975a852c8cd99761e2de1f4ba73402e44dd9",
                    "with": {
                        "aws-access-key-id": "${{ secrets.AWS_ACCESS_KEY_ID }}",
                        "aws-secret-access-key": "${{ secrets.AWS_SECRET_ACCESS_KEY }}",
                        "aws-region": "us-east-1",
                    },
                },
            ],
        },
    }
    exposed_secrets = find_exposed_github_secrets_in_job(job_with_no_env_vars)
    assert len(exposed_secrets) == 0


def test_find_exposed_github_secrets_in_job_env_vars_exist_but_no_secrets_exposed():
    job_with_env_vars_but_no_exposed_secrets = {
        "needs": [
            "start-fake-ec2-runner",
        ],
        "runs-on": "${{ needs.start-fake-ec2-runner.outputs.label }}",
        "steps": [
            {
                "name": "Step 1",
                "run": "echo 'hello'",
                "env": {"VAR1": "value1", "VAR2": "value2"},
            },
        ],
    }
    exposed_secrets = find_exposed_github_secrets_in_job(
        job_with_env_vars_but_no_exposed_secrets
    )
    assert len(exposed_secrets) == 0


def test_find_exposed_github_secrets_in_job_env_vars_exist_two_secrets_exposed():
    job_with_exposed_secrets_in_env = {
        "needs": [
            "start-fake-ec2-runner",
        ],
        "runs-on": "${{ needs.start-fake-ec2-runner.outputs.label }}",
        "steps": [
            {
                "name": "Step 1",
                "run": "echo 'hello'",
                "env": {
                    "SECRET_TOKEN": "${{ secrets.SECRET_TOKEN }}",
                    "SECRET_KEY": "${{ secrets.SECRET_KEY }}",
                    "SOME_VARIABLE": "test-value",
                },
            },
        ],
    }
    exposed_secrets = find_exposed_github_secrets_in_job(
        job_with_exposed_secrets_in_env
    )
    expected_exposed_secrets = {
        "SECRET_TOKEN": "${{ secrets.SECRET_TOKEN }}",
        "SECRET_KEY": "${{ secrets.SECRET_KEY }}",
    }
    assert len(exposed_secrets) == 2
    assert exposed_secrets == expected_exposed_secrets


def test_find_all_yaml_files_in_directory_no_yaml_files():
    with pytest.raises(
        GitWorkflowFilesNotFoundError,
        match=r"No YAML files were found under 'src' or any of its sub-directories",
    ):
        find_all_yaml_files_in_directory("src")


def test_find_all_yaml_files_in_directory_but_directory_does_not_exist():
    with pytest.raises(
        GitWorkflowFilesNotFoundError,
        match=r"No YAML files were found under 'nonexistent-dir' or any of its sub-directories",
    ):
        find_all_yaml_files_in_directory("nonexistent-dir")


# Note: We presently have three test YAML files under `tests/test_data`
def test_find_all_yaml_files_in_directory_with_no_trailing_slash():
    yaml_files = find_all_yaml_files_in_directory("tests")
    assert len(yaml_files) > 0


# Note: We presently have three test YAML files under `tests/test_data`
def test_find_all_yaml_files_in_directory_with_trailing_slash():
    yaml_files = find_all_yaml_files_in_directory("tests/")
    assert len(yaml_files) > 0


def test_is_git_workflow_file_not_a_workflow_file():
    non_workflow_file = {
        "some-key": {
            "sub-key-1": [
                "something",
                "something-else",
            ],
            "sub-key-2": "some-val",
        },
        "jobs": {},
    }
    is_valid_workflow = is_git_workflow_file(non_workflow_file)
    assert is_valid_workflow == False  # noqa: E712


def test_is_git_workflow_file_is_valid_file():
    valid_workflow_file = {
        "name": "Valid fake E2E Job",
        "on": {
            "schedule": [{"cron": "0 11 * * *"}],
            "workflow_dispatch": {
                "inputs": {
                    "pr_or_branch": {
                        "description": "pull request number or branch name",
                        "required": True,
                        "default": "main",
                    },
                },
            },
        },
        "jobs": {},  # doesn't matter if it's empty for this test
    }
    is_valid_workflow = is_git_workflow_file(valid_workflow_file)
    assert is_valid_workflow == True  # noqa: E712
