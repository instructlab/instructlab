# SPDX-License-Identifier: Apache-2.0

# Standard
from os import listdir
from os.path import dirname, exists
import logging
import pathlib
import typing

# Third Party
from git import GitError, Repo
import click

# First Party
from instructlab import clickext, utils
from instructlab.config.init import initialize_config
from instructlab.configuration import (
    DEFAULTS,
    Config,
    configs_exist,
    ensure_storage_directories_exist,
    get_default_config,
    profiles_exist,
    read_config,
    write_config,
)
from instructlab.defaults import DEFAULT_INDENT

logger = logging.getLogger(__name__)


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
    param_source = ctx.get_parameter_source("config_file")
    fresh_install = ensure_storage_directories_exist()
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
    # auto-detect and write config
    new_cfg, arch_family_processors, is_default_config = initialize_config(
        cfg=cfg,
        profile=profile,
        overwrite_profile=overwrite_profile,
        fresh_install=fresh_install,
        write_to_disk=False,
    )
    if is_default_config and interactive:
        # if auto detection didn't work, just continue prompting as usual!
        key = prompt_user_to_choose_vendors(arch_family_processors)

        if key is not None:
            for i, arch_family_processor in enumerate(
                arch_family_processors[key], start=1
            ):
                click.echo(
                    f"[{i}] {' '.join(map(str, arch_family_processor[0])).upper()}"
                )
            # pass the specific manufacturer specific list of tuples containing config names
            new_cfg = prompt_user_to_choose_profile(arch_family_processors[key])
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
    write_config(cfg=cfg)
    # this means auto-detect failed so we didn't print anything
    if is_default_config:
        utils.print_init_success()


# prompt_user_to_choose_vendors asks the user which hardware vendor best matches their system
def prompt_user_to_choose_vendors(
    arch_family_processors: dict[str, list[tuple[list[str], str]]],
) -> str | None:
    """
    prompt_user_to_choose_vendors asks the user which hardware vendor best matches their system
    """
    key = None
    click.echo(
        click.style(
            "Please choose a system profile.\nProfiles set hardware-specific defaults for all commands and sections of the configuration.",
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
    click.echo("[0] NO SYSTEM PROFILE")
    keys = list(arch_family_processors.keys())
    for idx, k in enumerate(keys, 1):
        click.echo(f"[{idx}] {k.upper()}")

    system_profile_selection = click.prompt(
        "Enter the number of your choice",
        type=click.IntRange(0, len(keys)),
        default=0,
    )
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
        click.echo("[0] NO SYSTEM PROFILE")
    elif system_profile_selection == 0:
        click.echo(
            "No profile selected - ilab will use generic code defaults - these may not be optimized for your system."
        )
    return key


# prompt_user_to_choose_profile asks the user to choose which specific profile for the hardware vendor best matches their system
def prompt_user_to_choose_profile(arch_family_processors) -> Config | None:
    """
    prompt_user_to_choose_profile asks the user to choose which specific profile for the hardware vendor best matches their system
    """
    cfg = None
    system_profile_selection = click.prompt(
        "Enter the number of your choice [hit enter for hardware defaults]",
        type=click.IntRange(0, len(arch_family_processors)),
        default=0,
    )
    # the file is SYSTEM_PROFILE_DIR/arch_family_processors[key][selection-1]
    if 1 <= system_profile_selection <= len(arch_family_processors):
        file = arch_family_processors[system_profile_selection - 1][1]
        click.secho(f"You selected: {file}", fg="green")
        cfg = read_config(file)
    elif system_profile_selection == 0:
        click.echo(
            "No profile selected - ilab will use generic code defaults - these may not be optimized for your system."
        )
    return cfg


def check_if_configs_exist(fresh_install) -> bool:
    if configs_exist():
        click.echo(
            f"Existing config file was found in:\n{DEFAULT_INDENT}{DEFAULTS.CONFIG_FILE}"
        )
        overwrite = click.confirm("Do you still want to continue?")
        if not overwrite:
            raise click.exceptions.Exit(0)
    if profiles_exist(fresh_install=fresh_install):
        return click.confirm(
            f"\nExisting system profiles were found in:\n{DEFAULT_INDENT}{DEFAULTS.SYSTEM_PROFILE_DIR}\nDo you want to restore these profiles to the default values?"
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
        separator = utils.get_separator(guide_text)
        click.echo(
            f"\n{separator}\n         Welcome to the InstructLab CLI\n{guide_text}\n{separator}\n"
        )
        click.echo(
            "Please provide the following values to initiate the "
            "environment [press 'Enter' for default options when prompted]"
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
