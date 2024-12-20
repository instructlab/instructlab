# SPDX-License-Identifier: Apache-2.0

# Standard
from functools import wraps
from pathlib import Path
from typing import List, Tuple, TypedDict
import copy
import glob
import json
import logging
import os
import pathlib
import platform
import re
import shutil
import struct
import subprocess
import tempfile
import time
import typing

# Third Party
from git import Repo, exc
from instructlab.schema.taxonomy import DEFAULT_TAXONOMY_FOLDERS as TAXONOMY_FOLDERS
from instructlab.schema.taxonomy import (
    TaxonomyMessageFormat,
    TaxonomyParser,
    TaxonomyReadingException,
)
from instructlab.training.chat_templates import (
    ibm_legacy_tmpl as granite_llama,  # type: ignore
)
from packaging import version
import click
import git
import gitdb
import yaml

# Local
from . import common
from .common import CLI_HELPER_SYS_PROMPT, SYSTEM_PROMPTS, SupportedModelArchitectures
from .defaults import DEFAULT_INDENT, DEFAULTS, RECOMMENDED_SCOPEO_VERSION

# mypy: disable_error_code="import-untyped"

logger = logging.getLogger(__name__)

AnalyzeModelResult = Tuple[str, str, str]


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


def is_valid_document(file_path: str, file_info: dict) -> bool:
    """Try to open a file with the provided file_info."""
    try:
        logger.debug("Opening %s file: %s", file_info["description"], file_path)
        with open(file_path, file_info["mode"], encoding=file_info["encoding"]):
            logger.debug("%s File opened successfully", file_info["description"])
        return True
    except Exception as e:
        logger.error(
            "Failed to open %s file: %s. Error %s",
            file_info["description"],
            file_path,
            e,
        )
        raise TaxonomyReadingException(
            f"Error reading {file_info['description']} file: {file_path}"
        ) from e


def _validate_documents(
    source: SourceDict,
    skip_checkout: bool = False,
) -> None:
    """
    Validate that we can retrieve the content of files from a Git repository specified in qna.yaml.

    Args:
        source (dict): Source info containing repository URL, commit hash, and list of file patterns.
        skip_checkout (bool, optional): If True, skips checking out the specific commit. Defaults to False.

    Raises:
        TaxonomyReadingException: If no knowledge documents could be opened.
        OSError, GitCommandError, FileNotFoundError: If an error occurs during Git operations or file access.

    Returns:
        None
    """
    repo_url = source.get("repo", "")
    commit_hash = source.get("commit", "")
    file_patterns = source.get("patterns", [])

    #  Supported file types and their respective open modes
    file_types = {
        ".md": {"mode": "r", "encoding": "utf-8", "description": "Markdown"},
        ".pdf": {"mode": "rb", "encoding": None, "description": "PDF"},
        # Add other file types when supported here.
    }

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
                    logger.debug("Checking file: %s", file_path)
                    if os.path.isfile(file_path):
                        file_extension = os.path.splitext(file_path)[1]
                        file_info = file_types.get(file_extension)

                        if not isinstance(file_info, dict):
                            click.secho(
                                "Unsupported file format for knowledge docs", fg="red"
                            )
                            raise click.exceptions.Exit(1)

                        # Attempt to open the file
                        if is_valid_document(file_path, file_info):
                            opened_files = True

            if not opened_files:
                raise TaxonomyReadingException(
                    "Couldn't find any valid knowledge documents."
                )

        except (OSError, exc.GitCommandError, FileNotFoundError) as e:
            click.secho(f"Error validating documents: {str(e)}", fg="red")
            raise click.exceptions.Exit(1)


def git_clone_checkout(
    repo_url: str, temp_dir: str, commit_hash: str, skip_checkout: bool
) -> Repo:
    repo = Repo.clone_from(repo_url, temp_dir)
    if not skip_checkout:
        repo.git.checkout(commit_hash)
    return repo


# pylint: disable=unused-argument
def get_sysprompt(arch: str):
    """
    Gets a system prompt specific to a model's architecture.
    For all unsupported architectures we return a default system prompt
    Args:
        arch (str): model architecture
    Returns:
        str: The system prompt for the model being used
    """
    return SYSTEM_PROMPTS.get(arch, common.DEFAULT_SYS_PROMPT)


def get_cli_helper_sysprompt() -> str:
    """
    Returns the system prompt to put the chatbot in CLI helper mode
    """
    return CLI_HELPER_SYS_PROMPT


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
    if not repo_url.startswith("docker://"):
        return False
    oci_url_regex = r"^docker://([a-zA-Z0-9\-_.]+(:[0-9]+)?)(/[a-zA-Z0-9\-_.]+)+(@[a-zA-Z0-9]+|:[a-zA-Z0-9\-_.]+)?$"
    if not re.match(oci_url_regex, repo_url):
        logger.warning(f"Invalid OCI repository URL: {repo_url}")
        return False
    return True


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


def print_table(headers: List[str], data: List[Tuple[str, ...]] | List[List[str]]):
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


def validate_safetensors_file(file_path: pathlib.Path) -> bool:
    """Validate the .safetensors file"""
    # Third Party
    from safetensors import safe_open

    try:
        with safe_open(file_path, framework="pt") as f:  # type: ignore
            # Check if at least one tensor exists
            tensor_list = f.keys()
            if not tensor_list:
                logger.debug(f"No tensors found in {file_path}")
                return False
    # pylint: disable=broad-exception-caught
    except Exception as e:
        logger.debug(f"Error while processing {file_path}: {e}")
        return False
    return True


def is_model_safetensors(model_path: pathlib.Path) -> bool:
    """Check if model_path is a valid safe tensors directory

    Check if provided path to model represents directory containing a safetensors representation
    of a model. Directory must contain a specific set of files to qualify as a safetensors model directory
    Args:
        model_path (Path): The path to the model directory
    Returns:
        bool: True if the model is a safetensors model, False otherwise.
    """
    try:
        files = list(model_path.iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
        logger.debug("Failed to read directory: %s", e)
        return False

    # directory should contain either .safetensors or .bin files to be considered valid
    has_bin_file = False
    safetensors_files: List[pathlib.Path] = []

    for file in files:
        if file.suffix == ".safetensors":
            safetensors_files.append(file)
        elif file.suffix == ".bin":
            has_bin_file = True

    if not safetensors_files and not has_bin_file:
        logger.debug("'%s' has no .safetensors or .bin files", model_path)
        return False

    if safetensors_files:
        for safetensors_file in safetensors_files:
            if not validate_safetensors_file(safetensors_file):
                return False

    basenames = {file.name for file in files}
    requires_files = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    diff = requires_files.difference(basenames)
    if diff:
        logger.debug("'%s' is missing %s", model_path, diff)
        return False

    for file in model_path.glob("*.json"):
        try:
            with file.open(encoding="utf-8") as f:
                json.load(f)
        except (PermissionError, json.JSONDecodeError) as e:
            logger.debug("'%s' is not a valid JSON file: e", file, e)
            return False

    return True


def is_model_gguf(model_path: pathlib.Path) -> bool:
    """
    Check if the file is a GGUF file.
    Args:
        model_path (Path): The path to the file.
    Returns:
        bool: True if the file is a GGUF file, False otherwise.
    """

    if os.path.isdir(model_path):
        logger.debug(f"GGUF Path {model_path} is a directory")
        return False

    # Third Party
    from gguf.constants import GGUF_MAGIC

    try:
        with model_path.open("rb") as f:
            first_four_bytes = f.read(4)

        # Convert the first four bytes to an integer
        first_four_bytes_int = int(struct.unpack("<I", first_four_bytes)[0])

        return first_four_bytes_int == GGUF_MAGIC
    except struct.error as e:
        logger.debug(
            f"Failed to unpack the first four bytes of {model_path}. "
            f"The file might not be a valid GGUF file or is corrupted: {e}"
        )
        return False
    except OSError as e:
        logger.debug(f"An unexpected error occurred while processing {model_path}: {e}")
        return False


def _analyze_gguf(entry: Path) -> AnalyzeModelResult:
    # stat the gguf, add it to the table
    stat = Path(entry.absolute()).stat(follow_symlinks=False)
    f_size = stat.st_size
    adjusted_size, magnitude = convert_bytes_to_proper_mag(f_size)
    # add to table
    modification_time = os.path.getmtime(entry.absolute())
    modification_time_readable = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(modification_time)
    )
    return (entry.name, modification_time_readable, f"{adjusted_size:.1f} {magnitude}")


def _analyze_dir(
    entry: Path, list_checkpoints: bool, directory: Path
) -> List[AnalyzeModelResult]:
    actual_model_name = ""
    all_files_sizes = 0
    add_model = False
    models: List[AnalyzeModelResult] = []
    # walk entire dir.
    for root, _, files in os.walk(entry.as_posix()):
        normalized_path = os.path.normpath(root)
        # Split the path into its components
        parts = normalized_path.split(os.sep)
        # Get the last two or three (for checkpoints) parts and join them back into a path
        printed_parts = os.path.join(parts[-2], parts[-1])
        if directory == Path(DEFAULTS.CHECKPOINTS_DIR):
            printed_parts = os.path.join(parts[-3], parts[-2], parts[-1])
        # if this is a dir it could be:
        # top level repo dir `instructlab/`
        # top level model dir `instructlab/granite-7b-lab`
        # checkpoint top level dir `step-19`
        # any lower level dir: `instructlab/granite-7b-lab/.huggingface/download.....`
        # so, check if model is valid Safetensor, GGUF, or list it regardless w/ `--list-checkpoints`
        # if --list-checkpoints is specified, we will list all checkpoints in the checkpoints dir regardless of the validity
        if is_model_safetensors(Path(normalized_path)) or is_model_gguf(
            Path(normalized_path)
        ):
            actual_model_name = printed_parts
            all_files_sizes = 0
            add_model = True
        else:
            if list_checkpoints and directory is DEFAULTS.CHECKPOINTS_DIR:
                logging.debug("Including model regardless of model validity")
            else:
                continue
        for f in files:
            # stat each file in the dir, add the size in Bytes, then convert to proper magnitude
            full_file = os.path.join(root, f)
            # do not follow symlinks, we only want to list the models dir ones
            stat = Path(full_file).stat(follow_symlinks=True)
            all_files_sizes += stat.st_size
        adjusted_all_sizes, magnitude = convert_bytes_to_proper_mag(all_files_sizes)
        if add_model:
            # add to table
            modification_time = os.path.getmtime(entry.absolute())
            modification_time_readable = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(modification_time)
            )
            models.append(
                (
                    actual_model_name,
                    modification_time_readable,
                    f"{adjusted_all_sizes:.1f} {magnitude}",
                )
            )
    return models


def list_models(
    model_dirs: List[Path], list_checkpoints: bool
) -> List[AnalyzeModelResult]:
    """Goes through a set of directories and returns all of the model
    found, including checkpoints if selected.


    Args:
        model_dirs (List[Path]): List of base directories to search through.
        list_checkpoints (bool): Whether or not we should include trained checkpoints.

    Returns:
        List[AnalyzeResult]: Results of the listing operation.
    """
    # if we want to list checkpoints, add that dir to our list
    if list_checkpoints:
        model_dirs.append(Path(DEFAULTS.CHECKPOINTS_DIR))
    data: List[AnalyzeModelResult] = []
    for directory in model_dirs:
        for entry in Path(directory).iterdir():
            # if file, just tally the size. This must be a GGUF.
            if entry.is_file() and is_model_gguf(entry):
                data.append(_analyze_gguf(entry))
            elif entry.is_dir():
                data.extend(_analyze_dir(entry, list_checkpoints, directory))
    return data


def contains_argument(prefix: str, args: typing.Iterable[str]) -> bool:
    # Either --foo value or --foo=value
    return any(s == prefix or s.startswith(prefix + "=") for s in args)


def get_model_arch(model_path: pathlib.Path) -> str:
    """
    Extract a given model's architecture from its config if available and return it

    args
        model_path (Path): Path to the model, used to read the model config if available
    returns
        model_arch (str): The architecture of the model as expressed in the config file
    """

    model_arch = "default"

    if is_model_gguf(model_path):
        # Third Party
        from gguf import GGUFReader
        from transformers.integrations.ggml import _gguf_parse_value

        reader = GGUFReader(model_path)
        value = reader.fields["general.architecture"]
        model_arch = [
            _gguf_parse_value(value.parts[_data_index], value.types)
            for _data_index in value.data
        ][0]

    elif is_model_safetensors(model_path):
        try:
            model_cfg = get_config_file_from_model(model_path, "config.json")
            model_arch = model_cfg["model_type"]
        except Exception as e:
            logger.debug(
                f"Unable to get model architecture from config for model: {model_path}: {e}. Falling back to default value"
            )

    return model_arch


def get_config_file_from_model(model_path: pathlib.Path, filename: str):
    """
    Reads a config file from a model's directory into memory

    args
        model_path (str): path to the model
        filename (str): name of config file to be read
    returns
        json loaded config dict
    """
    try:
        with open(
            pathlib.Path(model_path) / filename,
            "r",
            encoding="utf-8",
        ) as f:
            model_cfg = json.load(f)
        return model_cfg
    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
        raise ValueError(
            f"Unable to load file {filename} from {model_path}: {e}"
        ) from e


def get_model_template_from_tokenizer(model_path: pathlib.Path) -> Tuple[str, str, str]:
    """
    Attempts to read a model's tokenizer config file and extract the template and special
    tokens from it

    args
        model_path (str): path to the model
    returns
        template (str): resolved chat template
        bos_token (str): Beginning of sentence token
        eos_token (str): End of sentence token
    """
    template = eos_token = bos_token = None

    try:
        tcfg = get_config_file_from_model(model_path, "tokenizer_config.json")
        template = tcfg["chat_template"]
        bos_token = tcfg["bos_token"]
        eos_token = tcfg["eos_token"]
    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
        raise e

    return template, eos_token, bos_token


def use_legacy_pretraining_format(model_path: Path, model_arch: str) -> bool:
    """
    Checks if the provided model is carrying a tokenizer config that specifies
    legacy special tokens. If unable to find/load the tokenizer config, falls back
    to check the architecture of the model. This function determines if SDG and training
    need to be instructed to also work with legacy tokens and chat templates

    args
        model_path (str): path to the model
        model_arch (str): architecture of model
    returns
        (bool): True if we should use legacy pretraining format and template, false if not
    """
    try:
        _, eos_token, bos_token = get_model_template_from_tokenizer(model_path)
        if eos_token is not None and bos_token is not None:
            if (
                eos_token == granite_llama.SPECIAL_TOKENS.eos.token
                and bos_token == granite_llama.SPECIAL_TOKENS.bos.token
            ):
                return True
    except Exception:
        logger.debug(
            "unable to load tokenizer config to determine pretraining format. Checking model architecture"
        )
        return model_arch.lower() == SupportedModelArchitectures.LLAMA
    return False


def check_skopeo_version():
    """
    Check if skopeo is installed and the version is at least the recommended version
    This is required for downloading models from OCI registries.
    """
    # Run the 'skopeo --version' command and capture the output
    try:
        result = subprocess.run(
            ["skopeo", "--version"], capture_output=True, text=True, check=True
        )
    except FileNotFoundError as fnf_exc:
        logger.error(
            f"\nskopeo is not installed.\nPlease install recommended version {RECOMMENDED_SCOPEO_VERSION}",
        )
        raise fnf_exc

    logger.debug(f"'skopeo --version' output: {result.stdout}")

    # Extract the version number using a regular expression
    match = re.search(r"skopeo version (\d+\.\d+\.\d+)", result.stdout)
    if match:
        installed_version = match.group(1)
        logger.debug(f"detected skopeo version: {installed_version}")

        # Compare the extracted version with the required version
        if version.parse(installed_version) < version.parse(RECOMMENDED_SCOPEO_VERSION):
            raise ValueError(
                f"\n{DEFAULT_INDENT}skopeo version {installed_version} is lower than {RECOMMENDED_SCOPEO_VERSION}.\n{DEFAULT_INDENT}Please upgrade it."
            )
    else:
        logger.warning(
            f"Failed to determine skopeo version. Recommended version is {RECOMMENDED_SCOPEO_VERSION}. Downloading the model might fail."
        )


def get_cuda_device_properties(gpu_num: int) -> tuple[str, int]:
    """
    get_cuda_device_properties returns the name of the GPU and the amount of vRAM
    """
    # Third Party
    import torch

    properties = torch.cuda.get_device_properties(gpu_num)
    chip_name = properties.name.lower()
    return chip_name, int(properties.total_memory)
