# SPDX-License-Identifier: Apache-2.0

# Standard
from copy import copy
from os import listdir
from os.path import dirname, exists
import os
import pathlib
import re
import shutil
import typing

# Third Party
from git import GitError, Repo
import click

# First Party
from instructlab import clickext, utils
from instructlab.configuration import (
    DEFAULTS,
    Config,
    ensure_storage_directories_exist,
    get_default_config,
    read_config,
    recreate_system_profiles,
    write_config,
)


@click.command()
@click.option(
    "--interactive/--non-interactive",
    default=True,
    show_default=True,
    help="Initialize the environment assuming defaults.",
)
@click.option(
    "--model-path",
    type=click.Path(),
    default=lambda: DEFAULTS.DEFAULT_CHAT_MODEL,
    show_default="The instructlab data files location per the user's system.",
    help="Path to the model used during generation.",
)
@click.option(
    "--taxonomy-base",
    default=DEFAULTS.TAXONOMY_BASE,
    show_default=True,
    help="Base git-ref to use when listing/generating new taxonomy.",
)
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    default=lambda: DEFAULTS.TAXONOMY_DIR,
    show_default="The instructlab data files location per the user's system.",
    help="Path to where the taxonomy should be cloned.",
)
@click.option(
    "--repository",
    default=DEFAULTS.TAXONOMY_REPO,
    show_default=True,
    help="Taxonomy repository location.",
)
@click.option(
    "--min-taxonomy",
    is_flag=True,
    help="Shallow clone the taxonomy repository with minimum size. "
    "Please do not use this option if you are planning to contribute back "
    "using the same taxonomy repository. ",
)
@click.option(
    "--profile",
    type=click.Path(),
    default=None,
    help="Overwrite the default values in the generated config.yaml by passing in an existing configuration yaml.",
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(),
    default=None,
    envvar=DEFAULTS.ILAB_GLOBAL_CONFIG,
    show_default=True,
    help="Path to a configuration file.",
)
@clickext.display_params
@click.pass_context
def init(
    ctx,
    interactive,
    model_path: str,
    taxonomy_path,
    taxonomy_base,
    repository,
    min_taxonomy,
    profile,
    config_file,
):
    """Initializes environment for InstructLab"""
    fresh_install = ensure_storage_directories_exist()
    param_source = ctx.get_parameter_source("config_file")
    overwrite_profile = False
    if interactive:
        overwrite_profile = check_if_configs_exist(fresh_install)
    model_path, taxonomy_path, cfg = get_params(
        param_source,
        interactive,
        repository,
        min_taxonomy,
        model_path,
        taxonomy_path,
        config_file,
    )
    if overwrite_profile:
        click.echo(
            f"\nGenerating config file and profiles:\n    {DEFAULTS.CONFIG_FILE}\n    {DEFAULTS.TRAIN_PROFILE_DIR}\n"
        )
        recreate_system_profiles(overwrite=True)
    else:
        click.echo(f"\nGenerating config file:\n    {DEFAULTS.CONFIG_FILE}\n")
    if profile is not None:
        cfg = read_config(profile)
    elif interactive:
        new_cfg = walk_and_print_system_profiles()
        if new_cfg is not None:
            cfg = new_cfg
    # we should not override all paths with the serve model if special ENV vars exist
    if param_source != click.core.ParameterSource.ENVIRONMENT:
        cfg.chat.model = model_path
        cfg.serve.model_path = model_path
        cfg.evaluate.model = model_path
        cfg.generate.taxonomy_base = taxonomy_base
        cfg.generate.taxonomy_path = taxonomy_path
        cfg.evaluate.mt_bench_branch.taxonomy_path = taxonomy_path
    write_config(cfg)

    ready_text = "  You're ready to start using `ilab`. Enjoy!"
    separator = get_separator(ready_text)
    click.secho(
        f"\n{separator}\n    Initialization completed successfully!\n{ready_text}\n{separator}",
        fg="green",
    )


# walk_and_print_system_profiles prints interactive prompts asking the user to choose
# their hardware vendor and system profile
def walk_and_print_system_profiles() -> Config | None:
    """
    walk_and_print_system_profiles prints interactive prompts asking the user to choose
    their hardware vendor and system profile
    """
    cfg = None
    system_profile_files = []
    arch_family_processors: dict[str, list[list[str]]] = {}
    for dirpath, _dirnames, filenames in os.walk(DEFAULTS.SYSTEM_PROFILE_DIR):
        for filename in filenames:
            system_profile_files.append(os.path.join(dirpath, filename))
            arch_family_processor = os.path.relpath(
                os.path.join(dirpath, filename), DEFAULTS.SYSTEM_PROFILE_DIR
            ).split("/", 3)
            arch_family_processors.setdefault(arch_family_processor[0], []).append(
                arch_family_processor
            )

    click.echo(
        click.style(
            "Please choose a system profile.\n Profiles set hardware-specific defaults for all commands and sections of the configuration.",
            fg="green",
        )
    )
    # print info like APPLE, AMD, INTEL and have them select
    click.echo(
        click.style(
            "First, please select the hardware vendor your system falls into",
            bg="blue",
            fg="white",
        )
    )
    keys = list(arch_family_processors.keys())
    for idx, key in enumerate(keys, 1):
        print(f"[{idx}] {key.upper()}")

    system_profile_selection = click.prompt(
        "Enter the number of your choice",
        type=int,
        default=0,
    )

    # now print all choices in the selected hw vendor and have user choose
    if 1 <= system_profile_selection <= len(keys):
        key = keys[system_profile_selection - 1]
        click.echo(f"You selected: {key.upper()}")
        click.echo(
            click.style(
                "Next, please select the specific hardware configuration that most closely matches your system.",
                bg="blue",
                fg="white",
            )
        )
        click.echo("[0] No system profile")
        i = 1
        for arch_family_processor in arch_family_processors[key]:
            # the following logic is specifically for printing the name
            # we still want to preserve the arch_family_processor for when we open the file
            # if the last entry has an _, split that out. This follows the format like m2_max.yaml, we just want max.
            # Copy for preservation when working with printed names
            printed_arch_family_processor = copy(arch_family_processor)
            # Process the last element, extracting text after "_" if present, removing ".yaml"
            printed_arch_family_processor[-1] = re.sub(
                r".*_(\w+)\.yaml$|^(\w+)\.yaml$",
                lambda m: m.group(1) if m.group(1) else m.group(2),
                printed_arch_family_processor[-1],
            )
            # removes dupes in the case of `APPLE M2 M2` (the above logic is written to change things like M2_MAX.yaml into M2 MAX)
            printed_arch_family_processor = list(
                dict.fromkeys(printed_arch_family_processor)
            )

            # now echo it in all caps
            click.echo(
                f"[{i}] {' '.join(map(str, printed_arch_family_processor)).upper()}"
            )
            i += 1

        system_profile_selection = click.prompt(
            "Enter the number of your choice [hit enter for hardware defaults]",
            type=int,
            default=0,
        )

        # the file is SYSTEM_PROFILE_DIR/arch_family_procesors[key][selection-1]
        if 1 <= system_profile_selection <= len(system_profile_files):
            file = os.path.join(
                DEFAULTS.SYSTEM_PROFILE_DIR,
                "/".join(
                    map(str, arch_family_processors[key][system_profile_selection - 1])
                ),
            )
            click.echo(click.style(f"You selected: {file}", fg="green"))
            cfg = read_config(file)
        elif system_profile_selection == 0:
            click.echo(
                "No profile selected - any hardware acceleration for training must be configured manually."
            )
        else:
            click.secho(
                "Invalid selection. Please select a valid system profile option.",
                fg="red",
            )
            raise click.exceptions.Exit(1)
    return cfg


def check_if_configs_exist(fresh_install) -> bool:
    if exists(DEFAULTS.CONFIG_FILE):
        click.echo(f"Existing config file was found in:\n    {DEFAULTS.CONFIG_FILE}")
        overwrite = click.confirm("Do you still want to continue?")
        if not overwrite:
            raise click.exceptions.Exit(0)
    if exists(DEFAULTS.SYSTEM_PROFILE_DIR) and not fresh_install:
        return click.confirm(
            f"Existing system profiles were found in {DEFAULTS.SYSTEM_PROFILE_DIR}\nDo you want to restore these profiles to the default values?"
        )
    # default behavior should be do NOT overwrite files that could have just been created
    return False


def get_params_from_env(
    obj: typing.Optional[typing.Any],
) -> typing.Tuple[str, str, Config]:
    if obj is None or not hasattr(obj, "config"):
        raise ValueError("obj must not be None and must have a 'config' attribute")
    return (
        obj.config.generate.taxonomy_path,
        obj.config.generate.taxonomy_base,
        obj.config,
    )


def get_params(
    param_source: click.core.ParameterSource,
    interactive: bool,
    repository: str,
    min_taxonomy: bool,
    model_path: str,
    taxonomy_path: pathlib.Path,
    config: pathlib.Path | None,
) -> typing.Tuple[str, pathlib.Path, Config]:
    cfg = get_default_config()
    if config is not None:
        cfg = read_config(config)
    clone_taxonomy_repo = True
    if interactive:
        guide_text = "  This guide will help you to setup your environment"
        separator = get_separator(guide_text)
        click.echo(
            f"\n{separator}\n         Welcome to the InstructLab CLI\n{guide_text}\n{separator}\n"
        )
        click.echo(
            "Please provide the following values to initiate the "
            "environment [press Enter for defaults]:"
        )
        if param_source != click.core.ParameterSource.ENVIRONMENT:
            taxonomy_path = utils.expand_path(
                click.prompt("Path to taxonomy repo", default=taxonomy_path)
            )

    try:
        taxonomy_contents = listdir(taxonomy_path)
    except FileNotFoundError:
        taxonomy_contents = []
    if taxonomy_contents:
        clone_taxonomy_repo = False
    elif interactive and param_source != click.core.ParameterSource.ENVIRONMENT:
        click.echo(f"`{taxonomy_path}` seems to not exist or is empty.")
        clone_taxonomy_repo = click.confirm(
            f"Should I clone {repository} for you?",
            default=True,
        )

    # clone taxonomy repo if it needs to be cloned
    if clone_taxonomy_repo:
        click.echo(f"Cloning {repository}...")
        clone_depth = False if not min_taxonomy else 1
        try:
            Repo.clone_from(
                repository,
                taxonomy_path,
                branch="main",
                recurse_submodules=True,
                depth=clone_depth,
            )
        except GitError as exc:
            click.secho(f"Failed to clone taxonomy repo: {exc}", fg="red")
            click.secho(f"Please make sure to manually run `git clone {repository}`")
            raise click.exceptions.Exit(1)

    # check if models dir exists, and if so ask for which model to use
    models_dir = dirname(model_path)
    if (
        interactive
        and exists(models_dir)
        and param_source != click.core.ParameterSource.ENVIRONMENT
    ):
        model_path = utils.expand_path(
            click.prompt("Path to your model", default=model_path)
        )

    return model_path, taxonomy_path, cfg


def get_separator(text: str) -> str:
    text_length = len(text)
    try:
        terminal_width = shutil.get_terminal_size().columns
    except Exception:  # pylint: disable=broad-exception-caught
        # Exception can occur in non-interactive scenarios where no terminal is associated with the process
        terminal_width = text_length
    separator_length = min(text_length, terminal_width)
    return "-" * separator_length
