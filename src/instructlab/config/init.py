# SPDX-License-Identifier: Apache-2.0

# Standard
from math import floor
from os import listdir
from os.path import dirname, exists
import logging
import os
import pathlib
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
from instructlab.defaults import DEFAULT_INDENT
from instructlab.utils import convert_bytes_to_proper_mag

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
            f"\nGenerating config file and profiles:\n{DEFAULT_INDENT}{DEFAULTS.CONFIG_FILE}\n{DEFAULT_INDENT}{DEFAULTS.SYSTEM_PROFILE_DIR}\n"
        )
        recreate_system_profiles(overwrite=True)
    else:
        click.echo(
            f"\nGenerating config file:\n{DEFAULT_INDENT}{DEFAULTS.CONFIG_FILE}\n"
        )
    if profile is not None:
        cfg = read_config(profile)
    elif interactive:
        new_cfg = hw_auto_detect()
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
        f"\n{separator}\n{DEFAULT_INDENT}Initialization completed successfully!\n{ready_text}\n{separator}",
        fg="green",
    )


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


def walk_and_print_system_profiles(
    vram: int, gpus: int, chip_name: str
) -> Config | None:
    """
    walk_and_print_system_profiles prints interactive prompts asking the user to choose
    their hardware vendor and system profile. If a profile is auto detected to match the
    current hardware, that is returned before prompting to user to manually choose.
    This returns a Config object either auto-detected for the user or manually selected by them.
    """
    cfg = None
    arch_family_processors: dict[str, list[tuple[list[str], str]]] = {}
    for dirpath, _dirnames, filenames in os.walk(DEFAULTS.SYSTEM_PROFILE_DIR):
        for filename in filenames:
            # keep track of the file to open
            system_profile_file = os.path.join(dirpath, filename)
            sys_profile = read_config(config_file=system_profile_file)
            full_chip_name = []
            # see if our system matches the attributes of this metadata file. If so, this is our config
            # the easiest way to do this is to take the combo of the manufacturer, family, gpu count, and SKU info to see if it matches.
            if sys_profile.metadata.cpu_info is not None:
                # keep track of name in case we need to print menu
                full_chip_name = sys_profile.metadata.cpu_info.lower().split(" ")
                # we are on a cpu system, that makes this easier (no SKU info)
                if chip_name == sys_profile.metadata.cpu_info.lower():
                    cfg = sys_profile
                    break
            elif (
                sys_profile.metadata.gpu_manufacturer is not None
                and sys_profile.metadata.gpu_family is not None
            ):
                # do not include SKU in the name we store
                full_chip_name = [
                    sys_profile.metadata.gpu_manufacturer.lower(),
                    sys_profile.metadata.gpu_family.lower(),
                    f"x{sys_profile.metadata.gpu_count}",
                ]
                # need to include the case of no SKU
                all_gpu_skus = [""] + (
                    sys_profile.metadata.gpu_sku
                    if sys_profile.metadata.gpu_sku is not None
                    else []
                )
                for sku_to_match in all_gpu_skus:
                    assert isinstance(sys_profile.metadata.gpu_manufacturer, str)
                    assert isinstance(sys_profile.metadata.gpu_family, str)
                    # the .strip() is necessary to match the empty SKU or else we get a trailing space
                    str_chip_name = (
                        " ".join(
                            [
                                sys_profile.metadata.gpu_manufacturer,
                                sys_profile.metadata.gpu_family,
                                sku_to_match,
                            ]
                        )
                        .strip()
                        .lower()
                    )
                    if (
                        str_chip_name == chip_name
                        and gpus == sys_profile.metadata.gpu_count
                    ):
                        # this is our config
                        cfg = sys_profile
                        break
            else:
                logger.warning(f"No CPU or GPU info in profile {system_profile_file}")
                continue
            if cfg is not None:
                break
            arch_family_processors.setdefault(full_chip_name[0], []).append(
                (full_chip_name, system_profile_file)
            )
        if cfg is not None:
            break
    if cfg is not None:
        if "a100" in chip_name and vram == 320 and gpus == 8:
            cfg.train.max_batch_len = 10000
        chosen_profile = click.style(
            " ".join(full_chip_name).upper(),
            bg="blue",
            fg="white",
        )
        click.secho(
            f"We have detected the {chosen_profile} profile as an exact match for your system."
        )
        return cfg

    # if auto detection didn't work, just continue prompting as usual!
    key = prompt_user_to_choose_vendors(arch_family_processors)

    if key is not None:
        for i, arch_family_processor in enumerate(arch_family_processors[key], start=1):
            click.echo(f"[{i}] {' '.join(map(str, arch_family_processor[0])).upper()}")
        # pass the specific manufacturer specific list of tuples containing config names
        cfg = prompt_user_to_choose_profile(arch_family_processors[key])
    return cfg


def is_hpu_available() -> bool:
    """
    is_hpu_available checks if torch is built with HPU support
    if torch has the hpu attribute or we can successfully create a torch.device for HPU, return true.
    else, return false.
    """
    # Third Party
    import torch

    # Check for HPU availability without errors
    try:
        hpu_available = hasattr(torch, "hpu") and torch.device("hpu") is not None
        return hpu_available
    # pylint: disable=broad-exception-caught
    except Exception:
        return False


def get_gpu_or_cpu() -> tuple[str, int, int]:
    """
    get_gpu_or_cpu figures out what kind of hardware the user has and returns the name in the form of 'nvidia l4 x4' as well as if this is a CPU and if the user is on Linux
    """
    # Third Party
    import torch

    gpus = 0
    total_vram = 0
    chip_name = ""
    no_rocm_or_hpu = torch.version.hip is None and not is_hpu_available()
    # try nvidia
    if torch.cuda.is_available() and no_rocm_or_hpu:
        click.echo("Detecting hardware...")
        gpus = torch.cuda.device_count()
        for i in range(gpus):
            chip_name, vram = utils.get_cuda_device_properties(i)
            # if the SKU is something like A100-SXM4-40GB, split it for easier reading later
            if "-" in chip_name:
                chip_split = chip_name.split("-")
                chip_name = " ".join(chip_split)
            total_vram += vram

    vram = int(floor(convert_bytes_to_proper_mag(total_vram)[0]))
    # only do this on a CUDA system ROCm and HPU can have strange results when getting CPU info.
    if vram == 0 and no_rocm_or_hpu:
        # if no vRAM, try to see if we are on a CPU
        chip_name = get_chip_name()
        # ok, now we have a chip name. this means we can walk the supported profile names and see if they match
    return (
        chip_name,
        vram,
        gpus,
    )


# hw_auto_detect looks at a user's GPUs or CPU configuration and chooses the system profile which matches your system
def hw_auto_detect() -> Config | None:
    """
    hw_auto_detect looks at a user's GPUs or CPU configuration and chooses the system profile which matches your system
    """

    (
        chip_name,
        vram,
        gpus,
    ) = get_gpu_or_cpu()

    return walk_and_print_system_profiles(vram, gpus, chip_name)


def check_if_configs_exist(fresh_install) -> bool:
    if exists(DEFAULTS.CONFIG_FILE):
        click.echo(
            f"Existing config file was found in:\n{DEFAULT_INDENT}{DEFAULTS.CONFIG_FILE}"
        )
        overwrite = click.confirm("Do you still want to continue?")
        if not overwrite:
            raise click.exceptions.Exit(0)
    if exists(DEFAULTS.SYSTEM_PROFILE_DIR) and not fresh_install:
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
        separator = get_separator(guide_text)
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


def get_separator(text: str) -> str:
    text_length = len(text)
    try:
        terminal_width = shutil.get_terminal_size().columns
    except Exception:  # pylint: disable=broad-exception-caught
        # Exception can occur in non-interactive scenarios where no terminal is associated with the process
        terminal_width = text_length
    separator_length = min(text_length, terminal_width)
    return "-" * separator_length


def get_chip_name() -> str:
    """
    get_chip_name returns the name of the processor on the system (Linux or Mac for now).
    """
    # Standard
    import platform
    import subprocess

    system = platform.system()

    if system == "Darwin":  # macOS
        try:
            # macOS: Use system_profiler for detailed hardware info
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                check=True,
            )
            for line in result.stdout.splitlines():
                if "Chip" in line or "Processor Name" in line:
                    return line.split(":")[-1].strip().lower()
        except subprocess.CalledProcessError:
            return "Unsupported"

    elif system == "Linux":  # Linux
        try:
            # Linux: Read /proc/cpuinfo for CPU model information
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        cpu_name = line.split(":")[-1].strip().lower().split(" ")[0]
                        cpu_name = f"{cpu_name} cpu"
                        if "intel" in cpu_name:
                            # intel returns intel(r)
                            cpu_name = "intel cpu"
        except FileNotFoundError:
            return "Unsupported"

    logger.warning(
        "ilab is only officially supported on Linux and MacOS with M-Series Chips"
    )
    return "Unsupported"
