# SPDX-License-Identifier: Apache-2.0

# Standard
from os import listdir
from os.path import dirname, exists, join
import pathlib
import typing

# Third Party
from git import GitError, Repo
import click

# First Party
from instructlab import clickext, utils
from instructlab.config.init import hw_auto_detect
from instructlab.configuration import (
    DEFAULTS,
    Config,
    ensure_storage_directories_exist,
    get_default_config,
    read_config,
    read_train_profile,
    recreate_train_profiles,
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
            f"Generating `{DEFAULTS.CONFIG_FILE}` and `{DEFAULTS.TRAIN_PROFILE_DIR}`..."
        )
        recreate_train_profiles(overwrite=True)
    else:
        click.echo(f"Generating `{DEFAULTS.CONFIG_FILE}`...")
    if train_profile is not None:
        cfg.train = read_train_profile(train_profile)
    elif interactive:
        (
            chosen_profile,
            chosen_vram,
            detected_train_profile,
            edited_cfg,
        ) = hw_auto_detect()
        yn = None
        if chosen_vram is not None and chosen_profile is not None:
            click.echo(
                f"We chose {chosen_profile} as your designated training profile. This is for systems with {chosen_vram} GB of vRAM."
            )
            if edited_cfg:
                click.echo(
                    "This profile is the best approximation for your system based off of the amount of vRAM. We modified it to match the number of GPUs you have."
                )
            yn = click.confirm("Is this profile correct?", default=True)
        if yn is None or not yn:
            train_profile_filenames = sorted(listdir(DEFAULTS.TRAIN_PROFILE_DIR))
            # generate human-friendly text out of train profile filenames
            profile_options = []
            for filename in train_profile_filenames:
                # one gpu type
                split_filename = filename.split("_")
                if len(split_filename) == 2:
                    gputype1 = split_filename[0]
                    num_gpu = split_filename[1].split(".")[0]
                    profile_options.append(f"Nvidia {gputype1} {num_gpu} ({filename})")
                # two gpu types
                elif len(split_filename) == 3:
                    gputype1 = split_filename[0]
                    gputype2 = split_filename[1]
                    num_gpu = split_filename[2].split(".")[0]
                    profile_options.append(
                        f"Nvidia {gputype1}/{gputype2} {num_gpu} ({filename})"
                    )
            click.echo(
                "Please choose a train profile to use.\nTrain profiles assist with the complexity of configuring InstructLab training for specific GPU hardware.\nYou can still take advantage of hardware acceleration for training even if your hardware is not listed."
            )
            click.echo("[0] No profile (CPU, Apple Metal, AMD ROCm)")
            for i, value in enumerate(profile_options):
                click.echo(f"[{i+1}] {value}")
            train_profile_selection = click.prompt(
                "Enter the number of your choice [hit enter for hardware defaults]",
                type=int,
                default=0,
            )
            if 1 <= train_profile_selection <= len(profile_options):
                click.echo(
                    f"You selected: {profile_options[train_profile_selection - 1]}"
                )
                cfg.train = read_train_profile(
                    join(
                        DEFAULTS.TRAIN_PROFILE_DIR,
                        train_profile_filenames[train_profile_selection - 1],
                    )
                )
            elif train_profile_selection == 0:
                click.echo(
                    "No profile selected - any hardware acceleration for training must be configured manually."
                )
            else:
                click.secho(
                    "Invalid selection. Please select a valid train profile option.",
                    fg="red",
                )
                raise click.exceptions.Exit(1)
        else:
            cfg.train = detected_train_profile
    # we should not override all paths with the serve model if special ENV vars exist
    if param_source != click.core.ParameterSource.ENVIRONMENT:
        cfg.chat.model = model_path
        cfg.serve.model_path = model_path
        cfg.evaluate.model = model_path
        cfg.generate.taxonomy_base = taxonomy_base
        cfg.generate.taxonomy_path = taxonomy_path
        cfg.evaluate.mt_bench_branch.taxonomy_path = taxonomy_path
    write_config(cfg)

    click.secho(
        "Initialization completed successfully, you're ready to start using `ilab`. Enjoy!",
        fg="green",
    )


def check_if_configs_exist(fresh_install) -> bool:
    if exists(DEFAULTS.CONFIG_FILE):
        overwrite = click.confirm(
            f"Found {DEFAULTS.CONFIG_FILE}, do you still want to continue?"
        )
        if not overwrite:
            raise click.exceptions.Exit(0)
    if exists(DEFAULTS.TRAIN_PROFILE_DIR) and not fresh_install:
        return click.confirm(
            f"Existing training profiles were found in {DEFAULTS.TRAIN_PROFILE_DIR}\nDo you want to restore these profiles to the default values?"
        )
    # default behavior should be do NOT overwrite files that could have just been created
    return False


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
        click.echo(
            "Welcome to InstructLab CLI. This guide will help you to setup your environment."
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
        clone_taxonomy_repo = click.confirm(
            f"`{taxonomy_path}` seems to not exist or is empty. Should I clone {repository} for you?",
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
