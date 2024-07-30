# SPDX-License-Identifier: Apache-2.0

# Standard
from functools import cache, wraps
from importlib import resources
from importlib.abc import Traversable
from pathlib import Path
from typing import Any, List, Mapping, Optional, TypedDict
from urllib.parse import urlparse
import copy
import glob
import json
import logging
import os
import platform
import re
import subprocess
import tempfile

# Third Party
from git import Repo, exc
from referencing import Resource
import click
import git
import gitdb
import httpx
import yaml

# Local
from . import common

logger = logging.getLogger(__name__)

MIN_KNOWLEDGE_VERSION = 3

DEFAULT_YAML_RULES = """\
extends: relaxed

rules:
  line-length:
    max: 120
"""


class TaxonomyReadingException(Exception):
    """An exception raised during reading of the taxonomy."""


class Message(TypedDict):
    """
    Represents a message within an AI conversation.
    """

    content: str
    # one of: "user", "assistant", or "system"
    role: str


class MessageSample(TypedDict):
    """
    Represents a sample datapoint for a dataset using the HuggingFace messages format.
    """

    messages: List[Message]
    group: str
    dataset: str
    metadata: str


class LegacyMessageSample(TypedDict):
    """
    Represents a legacy message sample within an AI conversation.
    This is what is currently used by the legacy training methods such as Linux training and MacOS training.
    """

    system: str
    user: str
    assistant: str


def macos_requirement(echo_func, exit_exception):
    """Adds a check for MacOS before running a method.

    :param echo_func: Echo function accepting message and fg parameters to print the error.
    :param exit_exception: Exit exception to raise in case the MacOS requirement is not fulfilled.
    """

    def decorator(func):
        @wraps(func)
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
    except subprocess.CalledProcessError:
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
            "DeprecationWarning: Use `ilab taxonomy diff` instead.",
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
            "DeprecationWarning: Use `ilab taxonomy diff --quiet` instead.",
            fg="red",
        )
        retval = diff.callback(*args, **kwargs, quiet=True)
        return retval

    lab_check.callback = lab_check_callback
    cli.add_command(lab_check)


TAXONOMY_FOLDERS: List[str] = ["compositional_skills", "knowledge"]
"""Taxonomy folders which are also the schema names"""


def istaxonomyfile(fn):
    path = Path(fn)
    return path.suffix == ".yaml" and path.parts[0] in TAXONOMY_FOLDERS


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
        except gitdb.exc.BadName as e:
            raise TaxonomyReadingException(
                yaml.YAMLError(
                    f'Couldn\'t find the taxonomy git ref "{base}" from the current HEAD'
                )
            ) from e

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
        except IndexError as e:
            raise TaxonomyReadingException(
                yaml.YAMLError(
                    f'Couldn\'t find the taxonomy base branch "{base}" from the current HEAD'
                )
            ) from e

    modified_files = [
        d.b_path
        for d in head_commit.diff(None)
        if not d.deleted_file and istaxonomyfile(d.b_path)
    ]

    updated_taxonomy_files = list(set(untracked_files + modified_files))
    return updated_taxonomy_files


class SourceDict(TypedDict):
    repo: str
    commit: str
    patterns: List[str]


def _validate_documents(
    source: SourceDict,
    skip_checkout: bool = False,
) -> None:
    """
    Validate that we can retrieve the content of files from a Git repository.

    Args:
        source (dict): Source info containing repository URL, commit hash, and list of file patterns.

    Returns:
         None
    """ ""
    repo_url = source.get("repo", "")
    commit_hash = source.get("commit", "")
    file_patterns = source.get("patterns", [])
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            repo = git_clone_checkout(
                repo_url=repo_url,
                commit_hash=commit_hash,
                temp_dir=temp_dir,
                skip_checkout=skip_checkout,
            )

            logger.debug("Processing files...")
            opened_files = False
            for pattern in file_patterns:
                for file_path in glob.glob(os.path.join(repo.working_dir, pattern)):
                    if os.path.isfile(file_path) and file_path.endswith(".md"):
                        with open(file_path, "r", encoding="utf-8"):
                            # Success! we could open the file.
                            # We don't actually care about the contents, though.
                            opened_files = True

            if not opened_files:
                raise TaxonomyReadingException("Couldn't find knowledge documents")
        except (OSError, exc.GitCommandError, FileNotFoundError) as e:
            raise e


def git_clone_checkout(
    repo_url: str, temp_dir: str, commit_hash: str, skip_checkout: bool
) -> Repo:
    repo = Repo.clone_from(repo_url, temp_dir)
    if not skip_checkout:
        repo.git.checkout(commit_hash)
    return repo


# pylint: disable=unused-argument
def get_sysprompt(model=None):
    """
    Gets a system prompt specific to a model
    Args:
        model (str): currently not implemented
    Returns:
        str: The system prompt for the model being used
    """
    return common.SYS_PROMPT


@cache
def _load_schema(path: Traversable) -> Resource:
    """Load the schema from the path into a Resource object.

    Args:
        path (Traversable): Path to the schema to be loaded.

    Raises:
        NoSuchResource: If the resource cannot be loaded.

    Returns:
        Resource: A Resource containing the requested schema.
    """
    # pylint: disable=import-outside-toplevel
    # Third Party
    from referencing.exceptions import NoSuchResource
    from referencing.jsonschema import DRAFT202012

    try:
        contents = json.loads(path.read_text(encoding="utf-8"))
        resource = Resource.from_contents(
            contents=contents, default_specification=DRAFT202012
        )
    except Exception as e:
        raise NoSuchResource(str(path)) from e
    return resource


def validate_yaml(contents: Mapping[str, Any], taxonomy_path: Path) -> int:
    """Validate the parsed yaml document using the taxonomy path to
    determine the proper schema.

    Args:
        contents (Mapping): The parsed yaml document to validate against the schema.
        taxonomy_path (Path): Relative path of the taxonomy yaml document where the
            first element is the schema to use.

    Returns:
        int: The number of errors found during validation.
        Messages for each error have been logged.
    """
    # Third Party
    from jsonschema.protocols import Validator
    from jsonschema.validators import validator_for
    from referencing import Registry
    from referencing.exceptions import NoSuchResource
    from referencing.typing import URI

    errors = 0
    version = get_version(contents)
    schemas_path = resources.files("instructlab.schema").joinpath(f"v{version}")

    def retrieve(uri: URI) -> Resource:
        path = schemas_path.joinpath(uri)
        # This mypy violation will be fixed in:
        # https://github.com/instructlab/schema/pull/33
        return _load_schema(path)  # type: ignore[arg-type]

    schema_name = taxonomy_path.parts[0]
    if schema_name not in TAXONOMY_FOLDERS:
        schema_name = "knowledge" if "document" in contents else "compositional_skills"
        logger.info(
            f"Cannot determine schema name from path {taxonomy_path}. Using {schema_name} schema."
        )

    if schema_name == "knowledge" and version < MIN_KNOWLEDGE_VERSION:
        logger.error(
            f"Version {version} is not supported for knowledge taxonomy. Minimum supported version is {MIN_KNOWLEDGE_VERSION}."
        )
        errors += 1
        return errors

    try:
        schema_resource = retrieve(f"{schema_name}.json")
        schema = schema_resource.contents
        validator_cls = validator_for(schema)
        validator: Validator = validator_cls(
            # mypy doesn't understand attrs classes fields, see:
            # https://github.com/python/mypy/issues/5406
            schema,
            registry=Registry(retrieve=retrieve),  # type: ignore[call-arg]
        )

        for validation_error in validator.iter_errors(contents):
            errors += 1
            yaml_path = validation_error.json_path[1:]
            if not yaml_path:
                yaml_path = "."
            if validation_error.validator == "minItems":
                # Special handling for minItems which can have a long message for seed_examples
                message = (
                    f"Value must have at least {validation_error.validator_value} items"
                )
            else:
                message = validation_error.message[-200:]
            logger.error(
                f"Validation error in {taxonomy_path}: [{yaml_path}] {message}"
            )
    except NoSuchResource as e:
        cause = e.__cause__ if e.__cause__ is not None else e
        errors += 1
        logger.error(f"Cannot load schema file {e.ref}. {cause}")

    return errors


def get_version(contents: Mapping) -> int:
    try:
        return int(contents.get("version", 1))
    except (ValueError, TypeError):
        return 1  # fallback to version 1


# pylint: disable=broad-exception-caught
def validate_taxonomy_file(file_path: str, yaml_rules: Optional[str] = None):
    warnings = 0
    errors = 0
    file_path_p = Path(file_path).resolve()
    # file should end with ".yaml" explicitly
    if file_path_p.suffix != ".yaml":
        logger.warning(
            f"Skipping {file_path_p}! Use lowercase '.yaml' extension instead."
        )
        warnings += 1
        return warnings, errors
    for i in range(len(file_path_p.parts) - 1, -1, -1):
        if file_path_p.parts[i] in TAXONOMY_FOLDERS:
            taxonomy_path = Path(*file_path_p.parts[i:])
            break
    else:
        taxonomy_path = file_path_p
    # read file if extension is correct
    try:
        with file_path_p.open(encoding="utf-8") as file:
            contents = yaml.safe_load(file)
        if not contents:
            logger.warning(f"Skipping {file_path_p} because it is empty!")
            warnings += 1
            return warnings, errors
        if not isinstance(contents, Mapping):
            logger.error(
                f"{file_path_p} is not valid. The top-level element is not an object with key-value pairs."
            )
            errors += 1
            return warnings, errors

        # do general YAML linting if specified
        version = get_version(contents)
        if version > 1:  # no linting for version 1 yaml
            if yaml_rules is not None:
                if os.path.isfile(yaml_rules):
                    logger.debug(f"Using YAML rules from {yaml_rules}")
                    yamllint_cmd = [
                        "yamllint",
                        "-f",
                        "parsable",
                        "-c",
                        yaml_rules,
                        str(file_path_p),
                        "-s",
                    ]
                else:
                    logger.debug(f"Cannot find {yaml_rules}. Using default rules.")
                    yamllint_cmd = [
                        "yamllint",
                        "-f",
                        "parsable",
                        "-d",
                        DEFAULT_YAML_RULES,
                        str(file_path_p),
                        "-s",
                    ]
            else:
                yamllint_cmd = [
                    "yamllint",
                    "-f",
                    "parsable",
                    "-d",
                    DEFAULT_YAML_RULES,
                    str(file_path_p),
                    "-s",
                ]
            try:
                subprocess.check_output(yamllint_cmd, text=True)
            except subprocess.CalledProcessError as e:
                lint_messages = [f"Problems found in file {file_path_p}"]
                parsed_output = e.output.splitlines()
                for p in parsed_output:
                    errors += 1
                    delim = str(file_path_p) + ":"
                    parsed_p = p.split(delim)[1]
                    lint_messages.append(parsed_p)
                logger.error("\n".join(lint_messages))
                return warnings, errors

        validation_errors = validate_yaml(contents, taxonomy_path)
        if validation_errors:
            errors += validation_errors
            return warnings, errors

        # If the taxonomy file includes a document reference, validate that
        # we can retrieve the content of the document
        if "document" in contents:
            try:
                _validate_documents(contents["document"])
            except Exception as e:
                errors += 1
                logger.error(f"Failed to load document content for {file_path_p}: {e}")

    except Exception as e:
        errors += 1
        raise TaxonomyReadingException(f"Exception {e} raised in {file_path_p}") from e

    return warnings, errors


# TODO: remove `_logger` parameter after instructlab.sdg is fixed.
def validate_taxonomy(_logger, taxonomy, taxonomy_base, yaml_rules):
    if os.path.isfile(taxonomy):
        warnings, errors = validate_taxonomy_file(taxonomy, yaml_rules)
        if warnings:
            logger.warning(
                f"{warnings} warnings (see above) due to taxonomy file not (fully) usable."
            )
        if errors:
            raise TaxonomyReadingException(yaml.YAMLError("Taxonomy file with errors!"))
    else:  # taxonomy is dir
        # Gather the new or changed YAMLs using git diff
        updated_taxonomy_files = get_taxonomy_diff(taxonomy, taxonomy_base)
        total_errors = 0
        total_warnings = 0
        if updated_taxonomy_files:
            logger.debug("Found new taxonomy files:")
            for e in updated_taxonomy_files:
                logger.debug(f"* {e}")
        for f in updated_taxonomy_files:
            file_path = os.path.join(taxonomy, f)
            warnings, errors = validate_taxonomy_file(file_path, yaml_rules)
            total_warnings += warnings
            total_errors += errors
        if total_warnings:
            logger.warning(
                f"{total_warnings} warnings (see above) due to taxonomy files that were not (fully) usable."
            )
        if total_errors:
            raise TaxonomyReadingException(
                yaml.YAMLError(
                    f"{total_errors} total errors found across {len(updated_taxonomy_files)} updated taxonomy files!"
                )
            )


def get_ssl_cert_config(tls_client_cert, tls_client_key, tls_client_passwd):
    if tls_client_cert:
        return tls_client_cert, tls_client_key, tls_client_passwd


def http_client(params):
    return httpx.Client(
        cert=get_ssl_cert_config(
            params.get("tls_client_cert", None),
            params.get("tls_client_key", None),
            params.get("tls_client_passwd", None),
        ),
        verify=not params.get("tls_insecure", True),
    )


def split_hostport(hostport: str) -> tuple[str, int]:
    """Split server:port into server and port (IPv6 safe)"""
    if "//" not in hostport:
        # urlparse expects an URL-like input like '//host:port'
        hostport = f"//{hostport}"
    parsed = urlparse(hostport)
    hostname = parsed.hostname
    port = parsed.port
    if not hostname or not port:
        raise ValueError(f"Invalid host-port string: '{hostport}'")
    return hostname, port


def convert_messages_to_legacy_dataset(
    dataset: List[MessageSample],
) -> List[LegacyMessageSample]:
    """
    Converts the new HuggingFace messages dataset format to the legacy format.
    For reference, see: https://huggingface.co/docs/transformers/en/chat_templating#templates-for-chat-modelos

    **Note**: The legacy dataset format assumes only a turn of 1. All extra turns will be dropped.
    This means that we only look at the first 3 messages, and everything afterwards will be ignored.
    """
    converted_dataset: List[LegacyMessageSample] = []
    for dp in dataset:
        # in new dataset, the roles of a message will be determined.
        if len(dp["messages"]) < 3:
            raise ValueError(
                "The dataset is expecting a minimum of 3 messages in each sample."
            )

        converted: LegacyMessageSample = {  # type: ignore
            m["role"]: m["content"] for m in dp["messages"][:3]
        }
        converted_dataset.append(converted)
    return converted_dataset


def is_messages_dataset(
    dataset: List[MessageSample] | List[LegacyMessageSample],
) -> bool:
    """
    Indicates whether or not the provided dataset is using the newer "messages" format
    or the legacy format used by the old linux training script.
    """
    if not dataset:
        raise ValueError("dataset is empty")

    return "messages" in dataset[0]


def ensure_legacy_dataset(
    dataset: List[MessageSample] | List[LegacyMessageSample],
) -> List[LegacyMessageSample]:
    """
    Given a dataset that's either in the HF messages format or the legacy ilab train format,
    ensure that the returned dataset is always in the legacy ilab train format.
    """
    if not is_messages_dataset(dataset):
        return dataset  # type: ignore

    return convert_messages_to_legacy_dataset(dataset)  # type: ignore


def is_oci_repo(repo_url: str) -> bool:
    """
    Checks if a provided repository follows the OCI registry URL syntax
    """

    # TODO: flesh this out and make it a more robust check
    oci_url_prefix = "docker://"
    return repo_url.startswith(oci_url_prefix)


def is_huggingface_repo(repo_name: str) -> bool:
    # allow alphanumerics, underscores, hyphens and periods in huggingface repo names
    # repo name should be of the format <owner>/<model>
    pattern = r"^[\w.-]+\/[\w.-]+$"
    return re.match(pattern, repo_name) is not None


def load_json(file_path: Path):
    try:
        with open(file_path, encoding="UTF-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise ValueError(f"file not found: {file_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"could not read JSON file: {file_path}") from e
    except Exception as e:
        raise ValueError("unexpected error occurred") from e
