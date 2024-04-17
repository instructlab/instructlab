# SPDX-License-Identifier: Apache-2.0

# Standard
from functools import cache, wraps
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
import click
import git
import gitdb
import yaml

# Local
from . import common

DEFAULT_YAML_RULES = """\
extends: relaxed

rules:
  line-length:
    max: 120
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s",
)


class TaxonomyReadingException(Exception):
    """An exception raised during reading of the taxonomy."""


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


TAXONOMY_FOLDERS: List[str] = ["compositional_skills", "knowledge"]
"""Taxonomy folders which are also the schema names"""


def istaxonomyfile(fn):
    path = Path(fn)
    if path.suffix == ".yaml" and path.parts[0] in TAXONOMY_FOLDERS:
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
        except gitdb.exc.BadName as e:
            raise SystemExit(
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
            raise SystemExit(
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


def get_documents(
    logger,
    source: Dict[str, Union[str, List[str]]],
    skip_checkout: bool = False,
) -> List[str]:
    """
    Retrieve the content of files from a Git repository.

    Args:
        source (dict): Source info containing repository URL, commit hash, and list of file patterns.

    Returns:
         List[str]: List of document contents.
    """ ""
    repo_url = source.get("repo")
    commit_hash = source.get("commit")
    file_patterns = source.get("patterns")
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            repo = git_clone_checkout(
                repo_url=repo_url,
                commit_hash=commit_hash,
                temp_dir=temp_dir,
                skip_checkout=skip_checkout,
            )
            file_contents = []

            logger.debug("Processing files...")
            for pattern in file_patterns:
                for file_path in glob.glob(os.path.join(repo.working_dir, pattern)):
                    if os.path.isfile(file_path) and file_path.endswith(".md"):
                        with open(file_path, "r", encoding="utf-8") as file:
                            file_contents.append(file.read())

            if file_contents:
                return file_contents
            raise SystemExit("Couldn't find knowledge documents")
        except (OSError, exc.GitCommandError, FileNotFoundError) as e:
            raise e


def git_clone_checkout(
    repo_url: str, temp_dir: str, commit_hash: str, skip_checkout: bool
) -> Repo:
    repo = Repo.clone_from(repo_url, temp_dir)
    if not skip_checkout:
        repo.git.checkout(commit_hash)
    return repo


def chunk_document(documents: List, server_ctx_size, chunk_word_count) -> List[str]:
    """
    Iterates over the documents and splits them into chunks based on the word count provided by the user.
    Args:
        documents (dict): List of documents retrieved from git (can also consist of a single document).
        server_ctx_size (int): Context window size of server.
        chunk_word_count (int): Maximum number of words to chunk a document.
    Returns:
         List[str]: List of chunked documents.
    """
    no_tokens_per_doc = int(chunk_word_count * 1.3)  # 1 word ~ 1.3 token
    if no_tokens_per_doc > int(server_ctx_size - 1024):
        raise ValueError(
            "Error: {}".format(
                str(
                    f"Given word count ({chunk_word_count}) per doc will exceed the server context window size ({server_ctx_size})"
                )
            )
        )
    content = []
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=int(no_tokens_per_doc * 4),  # 1 token ~ 4 English character
        chunk_overlap=100,
    )

    for docs in documents:
        temp = text_splitter.create_documents([docs])
        content.extend([item.page_content for item in temp])

    return content


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
def _load_schema(path: "importlib.resources.abc.Traversable") -> "referencing.Resource":
    """Load the schema from the path into a Resource object.

    Args:
        path (Traversable): Path to the schema to be loaded.

    Raises:
        NoSuchResource: If the resource cannot be loaded.

    Returns:
        Resource: A Resource containing the requested schema.
    """
    # pylint: disable=C0415
    # Third Party
    from referencing import Resource
    from referencing.exceptions import NoSuchResource
    from referencing.jsonschema import DRAFT202012

    try:
        contents = json.loads(path.read_text(encoding="utf-8"))
        resource = Resource.from_contents(
            contents=contents, default_specification=DRAFT202012
        )
    except Exception as e:
        raise NoSuchResource(ref=str(path)) from e
    return resource


def validate_yaml(
    logger: Logger, contents: Mapping[str, Any], taxonomy_path: Path
) -> int:
    """Validate the parsed yaml document using the taxonomy path to
    determine the proper schema.

    Args:
        logger (Logger): The logger for errors/warnings.
        contents (Mapping): The parsed yaml document to validate against the schema.
        taxonomy_path (Path): Relative path of the taxonomy yaml document where the
        first element is the schema to use.

    Returns:
        int: The number of errors found during validation.
        Messages for each error have been logged.
    """
    # pylint: disable=C0415
    # Standard
    from importlib import resources

    # Third Party
    from jsonschema.protocols import Validator
    from jsonschema.validators import validator_for
    from referencing import Registry, Resource
    from referencing.exceptions import NoSuchResource
    from referencing.typing import URI

    errors = 0
    version = get_version(contents)
    schemas_path = resources.files("cli").joinpath(f"schema/v{version}")

    def retrieve(uri: URI) -> Resource:
        path = schemas_path.joinpath(uri)
        return _load_schema(path)

    schema_name = taxonomy_path.parts[0]
    if schema_name not in TAXONOMY_FOLDERS:
        schema_name = "knowledge" if "document" in contents else "compositional_skills"
        logger.info(
            f"Cannot determine schema name from path {taxonomy_path}. Using {schema_name} schema."
        )

    try:
        schema_resource = retrieve(f"{schema_name}.json")
        schema = schema_resource.contents
        validator_cls = validator_for(schema)
        validator: Validator = validator_cls(
            schema, registry=Registry(retrieve=retrieve)
        )

        for error in validator.iter_errors(contents):
            errors += 1
            error_path = error.json_path[1:]
            if not error_path:
                error_path = "."
            logger.error(
                f"Validation error in {taxonomy_path} on {error_path}: {error.message}"
            )
    except NoSuchResource as e:
        cause = e.__cause__ if e.__cause__ is not None else e
        errors += 1
        logger.error(f"Cannot load schema file {e.ref}. {cause}")

    return errors


def get_version(contents: Mapping) -> int:
    version = contents.get("version", 1)
    if not isinstance(version, int):
        # schema validation will complain about the type
        try:
            version = int(version)
        except ValueError:
            version = 1  # fallback to version 1
    return version


# pylint: disable=broad-exception-caught
def read_taxonomy_file(
    logger: Logger, file_path: str, yaml_rules: Optional[str] = None
):
    seed_instruction_data = []
    warnings = 0
    errors = 0
    file_path = Path(file_path).resolve()
    # file should end with ".yaml" explicitly
    if file_path.suffix != ".yaml":
        logger.warn(f"Skipping {file_path}! Use lowercase '.yaml' extension instead.")
        warnings += 1
        return None, warnings, errors
    for i in range(len(file_path.parts) - 1, -1, -1):
        if file_path.parts[i] in TAXONOMY_FOLDERS:
            taxonomy_path = Path(*file_path.parts[i:])
            break
    else:
        taxonomy_path = file_path
    # read file if extension is correct
    try:
        # do general YAML linting if specified
        if yaml_rules is not None:
            is_file = os.path.isfile(yaml_rules)
            if is_file:
                logger.debug(f"Using YAML rules from {yaml_rules}")
                yamllint_cmd = [
                    "yamllint",
                    "-f",
                    "parsable",
                    "-c",
                    yaml_rules,
                    file_path,
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
                    file_path,
                    "-s",
                ]
        else:
            yamllint_cmd = [
                "yamllint",
                "-f",
                "parsable",
                "-d",
                DEFAULT_YAML_RULES,
                file_path,
                "-s",
            ]
        try:
            subprocess.check_output(yamllint_cmd, text=True)
        except subprocess.SubprocessError as e:
            lint_messages = [f"Problems found in file {file_path}"]
            parsed_output = e.output.splitlines()
            for p in parsed_output:
                errors += 1
                delim = str(file_path) + ":"
                parsed_p = p.split(delim)[1]
                lint_messages.append(parsed_p)
            logger.error("\n".join(lint_messages))
            return None, warnings, errors
        # do more explict checking of file contents
        with open(file_path, "r", encoding="utf-8") as file:
            contents = yaml.safe_load(file)
            if not contents:
                logger.warn(f"Skipping {file_path} because it is empty!")
                warnings += 1
                return None, warnings, errors
            validation_errors = validate_yaml(logger, contents, taxonomy_path)
            if validation_errors:
                errors += validation_errors
                return None, warnings, errors

            # get seed instruction data
            tax_path = "->".join(taxonomy_path.parent.parts)
            task_description = contents.get("task_description")
            documents = contents.get("document")
            if documents:
                documents = get_documents(source=documents, logger=logger)
                logger.debug("Content from git repo fetched")

            for seed_example in contents.get("seed_examples"):
                question = seed_example.get("question")
                answer = seed_example.get("answer")
                context = seed_example.get("context", "")
                seed_instruction_data.append(
                    {
                        "instruction": question,
                        "input": context,
                        "output": answer,
                        "taxonomy_path": tax_path,
                        "task_description": task_description,
                        "document": documents,
                    }
                )
    except Exception as e:
        errors += 1
        raise TaxonomyReadingException(f"Exception {e} raised in {file_path}") from e

    return seed_instruction_data, warnings, errors


def read_taxonomy(logger, taxonomy, taxonomy_base, yaml_rules):
    seed_instruction_data = []
    is_file = os.path.isfile(taxonomy)
    if is_file:  # taxonomy is file
        seed_instruction_data, warnings, errors = read_taxonomy_file(
            logger, taxonomy, yaml_rules
        )
        if warnings:
            logger.warn(
                f"{warnings} warnings (see above) due to taxonomy file not (fully) usable."
            )
        if errors:
            raise SystemExit(yaml.YAMLError("Taxonomy file with errors! Exiting."))
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
            data, warnings, errors = read_taxonomy_file(logger, file_path, yaml_rules)
            total_warnings += warnings
            total_errors += errors
            if data:
                seed_instruction_data.extend(data)
        if total_warnings:
            logger.warn(
                f"{total_warnings} warnings (see above) due to taxonomy files that were not (fully) usable."
            )
        if total_errors:
            raise SystemExit(
                yaml.YAMLError(f"{total_errors} taxonomy files with errors! Exiting.")
            )
    return seed_instruction_data
