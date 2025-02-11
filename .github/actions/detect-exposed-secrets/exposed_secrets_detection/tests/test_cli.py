# SPDX-License-Identifier: Apache-2.0

# Third-Party
import pytest
import yaml

# Local
from src.custom_exceptions import (
    ExposedSecretsError,
    GitWorkflowFilesNotFoundError,
    MissingTriggerConditionsError,
)
from src.cli import (
    create_parser,
    run_validation,
)


def test_create_parser_succeeds():
    create_parser()


def test_run_validation_cli_arguments_are_missing():
    args = {}
    with pytest.raises(SystemExit):
        run_validation(args)


def test_run_validation_combination_of_cli_arguments_is_unsupported():
    args = {
        "file": "some-filename.yml",
        "dir": "some-directory",
    }
    with pytest.raises(NotImplementedError):
        run_validation(args)


def test_run_validation_workflow_file_does_not_exist():
    args = {
        "file": "some-nonexistent-file.yml",
    }
    with pytest.raises(FileNotFoundError):
        run_validation(args)


def test_run_validation_workflow_directory_does_not_exist():
    args = {
        "file": "some-nonexistent-directory/fake-dir",
    }
    with pytest.raises(FileNotFoundError):
        run_validation(args)


def test_run_validation_workflow_file_exists_but_has_no_trigger_conditions():
    args = {
        "file": "tests/test_data/git_workflow_missing_triggers.yml",
    }
    with pytest.raises(MissingTriggerConditionsError):
        run_validation(args)


def test_run_validation_workflow_directory_exists_but_one_file_has_a_yaml_parsing_error():
    args = {
        "dir": "tests",
    }
    with pytest.raises(yaml.YAMLError):
        run_validation(args)


def test_run_validation_workflow_but_yaml_is_not_a_workflow():
    args = {
        "file": "tests/test_data/random_yaml_file.yml",
    }
    with pytest.raises(
        GitWorkflowFilesNotFoundError,
        match=r"YAML file 'tests/test_data/random_yaml_file.yml' is not recognized as a Git workflow file.",
    ):
        run_validation(args)


def test_run_validation_workflow_is_manual_and_secrets_are_not_exposed():
    args = {
        "file": "tests/test_data/git_workflow_manual_trigger.yml",
    }
    # We don't need to assert anything here. Any secrets exposed will throw an exception.
    run_validation(args)


def test_run_validation_workflow_has_top_level_secrets():
    args = {
        "file": "tests/test_data/git_workflow_automatic_trigger_with_top_level_env_secrets.yaml",
    }
    # Assert we find a 'SECRET_TOKEN' and 'SECRET_KEY' loaded into the env, and see 'n/a' as the job name
    with pytest.raises(
        ExposedSecretsError,
        match=r"Detected one or more exposed secrets.*Findings:.*'job_name': 'n/a'.*secrets.SECRET_TOKEN.*secrets.SECRET_KEY.*",
    ):
        run_validation(args)


def test_run_validation_workflow_is_automatic_and_secrets_are_exposed():
    args = {
        "file": "tests/test_data/git_workflow_automatic_trigger.yaml",
    }
    # Assert we find a 'SECRET_TOKEN' and 'SECRET_KEY' loaded into the env
    with pytest.raises(
        ExposedSecretsError,
        match=r"Detected one or more exposed secrets.*Findings:.*secrets.SECRET_TOKEN.*secrets.SECRET_KEY.*",
    ):
        run_validation(args)
