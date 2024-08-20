# SPDX-License-Identifier: Apache-2.0

# Standard
from functools import wraps
from pathlib import Path
from typing import List, Tuple, TypedDict
from urllib.parse import urlparse
import copy
import glob
import json
import logging
import os
import pathlib
import platform
import re
import shutil
import subprocess
import tempfile
import nvidia_smi

# Third Party
from git import Repo, exc
from instructlab.schema.taxonomy import DEFAULT_TAXONOMY_FOLDERS as TAXONOMY_FOLDERS
from instructlab.schema.taxonomy import (
    TaxonomyMessageFormat,
    TaxonomyParser,
    TaxonomyReadingException,
)
import click
import git
import gitdb
import httpx
import yaml

# Local
from . import common

logger = logging.getLogger(__name__)


class SDGTokens:
    """
    Provides constant definitions for the tokens used by SDG.
    """

    USER = "<|user|>"
    ASSISTANT = "<|assistant|>"
    PRETRAINING = "<|pretraining|>"


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


class HttpClientParams(TypedDict):
    """
    Types the parameters used when initializing the HTTP client.
    """

    tls_client_cert: str | None
    tls_client_key: str | None
    tls_client_passwd: str | None
    tls_insecure: bool


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


def is_taxonomy_file(fn: str) -> bool:
    path = Path(fn)
    if path.parts[0] not in TAXONOMY_FOLDERS:
        return False
    if path.name == "qna.yaml":
        return True
    if path.name.casefold() in {"qna.yml", "qna.yaml"}:
        # warning for incorrect extension or case variants
        logger.warning(
            "Found a '%s' file: %s: taxonomy files must be named 'qna.yaml'. File will not be checked.",
            path.name,
            path,
        )
    return False


def get_taxonomy_diff(
    repo_path: str | Path = "taxonomy", base: str = "origin/main"
) -> list[str]:
    repo = git.Repo(repo_path)
    untracked_files = [u for u in repo.untracked_files if is_taxonomy_file(u)]

    branches = [b.name for b in repo.branches]  # type: ignore[attr-defined]

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
        contains = repo.git.branch("-a", "--contains", current_commit.hexsha)
        if re_git_branch.findall(contains):
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
        if not d.deleted_file and is_taxonomy_file(d.b_path)
    ]

    updated_taxonomy_files = list(set(untracked_files + modified_files))
    return updated_taxonomy_files


def get_taxonomy(repo="taxonomy"):
    repo = Path(repo)
    taxonomy_file_paths = []
    for root, _, files in os.walk(repo):
        for file in files:
            file_path = Path(root).joinpath(file).relative_to(repo)
            if is_taxonomy_file(file_path):
                taxonomy_file_paths.append(str(file_path))
    return taxonomy_file_paths


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


# pylint: disable=broad-exception-caught
def validate_taxonomy_file(
    file_path: str | Path, yamllint_config: str | None = None
) -> tuple[int, int]:
    parser = TaxonomyParser(
        schema_version=0,  # Use version value in yaml
        message_format=TaxonomyMessageFormat.LOGGING,  # Report warnings and errors to the logger
        yamllint_config=yamllint_config,
        yamllint_strict=True,  # Report yamllint warnings as errors
    )
    taxonomy = parser.parse(file_path)

    if taxonomy.warnings or taxonomy.errors:
        return taxonomy.warnings, taxonomy.errors

    # If the taxonomy file includes a document reference, validate that
    # we can retrieve the content of the document
    document = taxonomy.contents.get("document")
    if document:
        try:
            _validate_documents(document)
        except Exception:
            logger.error(
                "Failed to load document content for %s",
                taxonomy.path,
                exc_info=True,
            )
            taxonomy.errors += 1

    return taxonomy.warnings, taxonomy.errors


def validate_taxonomy(
    taxonomy: str | Path,
    taxonomy_base: str,
    yaml_rules: str | Path | None = None,
) -> None:
    yamllint_config = None  # If no custom rules file, use default config
    if yaml_rules is not None:  # user attempted to pass custom rules file
        yaml_rules_path = Path(yaml_rules)
        if yaml_rules_path.is_file():  # file was found, use specified config
            logger.debug("Using YAML rules from %s", yaml_rules)
            yamllint_config = yaml_rules_path.read_text(encoding="utf-8")
        else:
            logger.debug("Cannot find %s. Using default rules.", yaml_rules)

    if os.path.isfile(taxonomy):
        warnings, errors = validate_taxonomy_file(taxonomy, yamllint_config)
        if warnings:
            logger.warning(
                "%s warnings (see above) due to taxonomy file not (fully) usable.",
                warnings,
            )
        if errors:
            raise TaxonomyReadingException(yaml.YAMLError("Taxonomy file with errors!"))
    else:  # taxonomy is dir
        if taxonomy_base == "empty":
            # Gather all the yamls - equivalent to a diff against "the null tree"
            taxonomy_files = get_taxonomy(taxonomy)
        else:
            # Gather the new or changed YAMLs using git diff, including untracked files
            taxonomy_files = get_taxonomy_diff(taxonomy, taxonomy_base)
        total_errors = 0
        total_warnings = 0
        if taxonomy_files:
            logger.debug("Found new taxonomy files:")
            for e in taxonomy_files:
                logger.debug("* %s", e)
        for f in taxonomy_files:
            file_path = os.path.join(taxonomy, f)
            warnings, errors = validate_taxonomy_file(file_path, yamllint_config)
            total_warnings += warnings
            total_errors += errors
        if total_warnings:
            logger.warning(
                "%s warnings (see above) due to taxonomy files that were not (fully) usable.",
                total_warnings,
            )
        if total_errors:
            raise TaxonomyReadingException(
                yaml.YAMLError(
                    f"{total_errors} total errors found across {len(taxonomy_files)} taxonomy files!"
                )
            )


def get_ssl_cert_config(tls_client_cert, tls_client_key, tls_client_passwd):
    if tls_client_cert:
        return tls_client_cert, tls_client_key, tls_client_passwd


def http_client(params: HttpClientParams):
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


def is_pretraining_dataset(ds: List[MessageSample]) -> bool:
    """
    Determines whether or not the given dataset is the phase07 pretraining
    dataset.
    """
    if not ds:
        return False
    sample = ds[0]
    return any(m["role"] == "pretraining" for m in sample["messages"])


def get_user_assistant_from_pretraining(pretraining: str) -> Tuple[str, str]:
    """
    Given a pretraining message sample which contains the tokens '<|user|>' and '<|assistant|>',
    return the strings corresponding to the user & assistant inputs.

    This function assumes that a given message contains only a single instance of each
    `<|user|>` and `<|assistant|>` token, and that `<|user|>` precedes `<|assistant|>`.

    Raises ValueError if neither the '<|user|>' or '<|assistant|>' tokens exist
    within the given sample.
    """
    for token in (SDGTokens.ASSISTANT, SDGTokens.USER):
        if token not in pretraining:
            raise ValueError(
                f"pretraining sample doesn't contain the '{token}' token: {pretraining}"
            )
    first, assistant = pretraining.split(SDGTokens.ASSISTANT)
    _, user = first.split(SDGTokens.USER)
    return user, assistant


def convert_pretraining_messages_to_legacy_dataset(
    dataset: List[MessageSample],
) -> List[LegacyMessageSample]:
    """
    Given a Phase07 pretraining dataset that's in the messages format, returns
    a version that's been converted to be compatible with the legacy ilab data format.

    This function assumes that each sample contains at least a single message with the
    "pretraining" role. If a "system" role exists then it's also parsed out, otherwise
    an empty string is set for the system.
    """
    converted_dataset: List[LegacyMessageSample] = []
    for sample in dataset:
        pretraining = next(
            (
                msg["content"]
                for msg in sample["messages"]
                if msg["role"] == "pretraining"
            ),
            None,
        )
        if not pretraining:
            raise ValueError(
                "dataset contains sample which lacks a pretraining message"
            )

        # not sure if 'system' will exist in phase07 or not - leaving here just in case
        system = next(
            (msg["content"] for msg in sample["messages"] if msg["role"] == "system"),
            "",
        )
        user, assistant = get_user_assistant_from_pretraining(pretraining)
        converted_dataset.append(
            {"system": system, "user": user, "assistant": assistant}
        )
    return converted_dataset


def convert_standard_messages_to_legacy_dataset(
    dataset: List[MessageSample],
) -> List[LegacyMessageSample]:
    """
    This function converts a standard dataset that's in the HuggingFace
    messages format into the legacy ilab format.

    This function assumes that each sample in the dataset has at least 3 messages,
    of which the first 3 are: system, user, assistant. Any additional messages
    are ignored.
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


def convert_messages_to_legacy_dataset(
    dataset: List[MessageSample],
) -> List[LegacyMessageSample]:
    """
    Converts the new HuggingFace messages dataset format to the legacy format.
    For reference, see: https://huggingface.co/docs/transformers/en/chat_templating#templates-for-chat-modelos

    **Note**: The legacy dataset format assumes only a turn of 1. All extra turns will be dropped.
    This means that we only look at the first 3 messages, and everything afterwards will be ignored.
    """
    # return the converted dataset based on the type
    if is_pretraining_dataset(dataset):
        return convert_pretraining_messages_to_legacy_dataset(dataset)
    return convert_standard_messages_to_legacy_dataset(dataset)


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
    if not dataset:
        # base case - they are both equivalent
        return []

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


def print_table(headers: list[str], data: list[list[str]]):
    """
    Given a set of headers and corresponding dataset where the
    number of headers matches the length of each item in the given dataset.

    Prints out the dataset in the format:
    ```
    +--------+--------+-----+
    | Col 1  | Col 2  | ... |
    +--------+--------+-----+
    | Item 1 | Item 2 | ... |
    +--------+--------+-----+
    | .....  | .....  | ... |
    +--------+--------+-----+
    ```
    """
    column_widths = [
        max(len(str(row[i])) for row in data + [headers]) for i in range(len(headers))
    ]
    # Print separator line between headers and data
    horizontal_lines = ["-" * (width + 2) for width in column_widths]
    joining_line = "+" + "+".join(horizontal_lines) + "+"
    print(joining_line)
    outputs = []
    for header, width in zip(headers, column_widths, strict=False):
        outputs.append(f" {header:{width}} ")
    print("|" + "|".join(outputs) + "|")
    print(joining_line)
    for row in data:
        outputs = []
        for item, width in zip(row, column_widths, strict=False):
            outputs.append(f" {item:{width}} ")
        print("|" + "|".join(outputs) + "|")
    print(joining_line)


def convert_bytes_to_proper_mag(f_size: int) -> Tuple[float, str]:
    """
    Given an integer representing the filesize in bytes, returns
    a floating point representation of the size along with the associated
    magnitude of the size, e.g. 2000 would get converted into 1.95, "KB"
    """
    magnitudes = ["KB", "MB", "GB"]
    magnitude = "B"
    adjusted_fsize = float(f_size)
    for mag in magnitudes:
        if adjusted_fsize >= 1024:
            magnitude = mag
            adjusted_fsize /= 1024
        else:
            return adjusted_fsize, magnitude
    return adjusted_fsize, magnitude


def clear_directory(path: pathlib.Path) -> None:
    """Recursively deletes content below {path} and recreates directory."""
    if path.exists():
        shutil.rmtree(path)
    os.makedirs(path)


def are_nvidia_gpus_available() -> bool:
    """Check if NVIDIA GPUs are available"""
    try:
        nvidia_smi.nvmlInit()
        nvidia_smi.nvmlShutdown()
    except Exception:
        return False
    return True
