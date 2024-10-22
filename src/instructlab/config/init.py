# SPDX-License-Identifier: Apache-2.0

# Standard
from copy import copy
from math import floor
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
    _train,
    ensure_storage_directories_exist,
    get_default_config,
    get_profile_mappings,
    read_config,
    read_train_profile,
    recreate_system_profiles,
    write_config,
)
from instructlab.utils import convert_bytes_to_proper_mag


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
    "--train-profile",
    type=click.Path(),
    default=None,
    help="Overwrite the default training values in the generated config.yaml by passing in an existing training-specific yaml.",
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
    train_profile,
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
    if train_profile is not None:
        cfg.train = read_train_profile(train_profile)
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


def hw_auto_detect() -> tuple[str | None, int | None, _train, bool]:
    # Third Party
    import torch

    gpus = 0
    total_vram = 0
    train = _train()
    chosen_vram = None
    chosen_profile_name = None
    gpu_name = ""
    edited_cfg = False
    # try nvidia
    if torch.cuda.is_available() and torch.version.hip is None:
        click.echo("Detecting hardware...")
        gpus = torch.cuda.device_count()
        for i in range(gpus):
            properties = torch.cuda.get_device_properties(i)
            gpu_name = properties.name
            total_vram += properties.total_memory  # memory in B

    vram = int(floor(convert_bytes_to_proper_mag(total_vram)[0]))
    if vram == 0:
        return None, None, train, False
    matched, chosen_vram, chosen_profile_name, train = lookup_card(
        gpu_name=gpu_name, gpu_count=gpus, vram=vram
    )
    # if we used the vRAM to determine, we need to tell the user that
    if not matched:
        # edit nproc per node in the case that we didnt match an exact profile, so we use all GPUs on our system
        train.nproc_per_node = gpus
        edited_cfg = True
    return (chosen_profile_name, chosen_vram, train, edited_cfg)


# lookup_card checks if the user's GPU exists in our supported profile map. If not, we choose one based off of their vRAM
def lookup_card(
    gpu_name, gpu_count, vram
) -> tuple[bool, int | None, str | None, _train]:
    profiles = get_profile_mappings()
    card_entry = profiles.get(gpu_name)
    chosen_profile_name = None
    if card_entry is not None:
        chosen_profile_name = f"Nvidia {gpu_count}x {gpu_name}"
        # it is not guaranteed we match, even if we find an entry
        # what if the vram isn't found or what if the gpu count doesn't match
        matched, train = match_profile_based_on_gpu_count(
            card_entry=card_entry, gpu_count=gpu_count, vram=vram
        )
        if matched:
            return True, vram, chosen_profile_name, train
    vram, chosen_profile_name, train = match_profile_based_on_vram(vram=vram)
    # return False with the results of the vRAM matching to indicate we didn't find an exact match for this GPU configuration in our supported profiles
    return False, vram, chosen_profile_name, train


# match_profile_based_on_vram chooses a training profile based off the amount of vram in each profile
# this profile will be modified to match the number of GPUs on the user's system
def match_profile_based_on_vram(vram) -> tuple[int | None, str | None, _train]:
    chosen_profile_name = None
    chosen_vram = None
    train = _train()
    profiles = get_profile_mappings()
    for group_gpu_name, list_of_count_vram_and_config in profiles.items():
        for gpu_count_configs in list_of_count_vram_and_config:
            if (
                not isinstance(gpu_count_configs["gpu_count"], int)
                or not isinstance(gpu_count_configs["vram_and_config"], dict)
                or not isinstance(gpu_count_configs["vram_and_config"]["vram"], int)
                or not isinstance(
                    gpu_count_configs["vram_and_config"]["config"], _train
                )
            ):
                click.secho("Unable to retrieve train profiles", fg="red")
                raise click.exceptions.Exit(1)
            # we want to match based off if our vRAM is >= the Profile vRAM. However,
            # you do not want to overdo this and match a profile that has hundreds of GB less than your system
            # only match if the profile has 30GB less than our system.
            if (
                chosen_vram is None
                or chosen_vram > gpu_count_configs["vram_and_config"]["vram"]
            ) and (
                vram >= gpu_count_configs["vram_and_config"]["vram"]
                and vram - gpu_count_configs["vram_and_config"]["vram"] <= 30
            ):
                # set our return values
                gpus = gpu_count_configs["gpu_count"]
                chosen_profile_name = f"Nvidia {gpus}x {group_gpu_name}"
                chosen_vram = gpu_count_configs["vram_and_config"]["vram"]
                train = gpu_count_configs["vram_and_config"]["config"]
    return chosen_vram, chosen_profile_name, train


# match_profile_based_on_gpu_count looks for a training profile which directly matches the gpu name and count
# this training profile will be used as is
def match_profile_based_on_gpu_count(
    card_entry, gpu_count, vram
) -> tuple[bool, _train]:
    train = _train()
    matched = False
    for gpu_count_configs in card_entry:
        if not isinstance(gpu_count_configs["gpu_count"], int) or not isinstance(
            gpu_count_configs["vram_and_config"], dict
        ):
            click.secho("Unable to retrieve train profiles", fg="red")
            raise click.exceptions.Exit(1)
        # if GPU name and number matches, this is a direct match
        if (
            gpu_count == gpu_count_configs["gpu_count"]
            and vram == gpu_count_configs["vram_and_config"]["vram"]
        ):
            train = gpu_count_configs["vram_and_config"]["config"]
            matched = True
            break
    return matched, train


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
