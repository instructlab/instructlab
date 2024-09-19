# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
import click
import nvidia_smi
import pynvml

# First Party
from instructlab import clickext, utils
from instructlab.configuration import (
    Config,
    _train,
    render_configs_and_profiles,
    write_config,
)
from instructlab.utils import convert_bytes_to_proper_mag

logger = logging.getLogger(__name__)

PROFILES = [
    "Single Consumer GPU",
    "Single Server GPU",
    "Multi Consumer GPU",
    "Multi Server GPU",
    "MacOS",
]


@click.command(name="set")
@clickext.display_params
@click.pass_context
def set(ctx):
    gpus = 0
    if utils.are_nvidia_gpus_available():
        pynvml.nvmlInit()
        gpus = nvidia_smi.nvmlDeviceGetCount()
    elif not utils.is_macos_with_m_chip():
        gpus = int(click.prompt("How many Dedicated GPUs do you have?"))
    # prorile_mappings is the following format:
    # [(MAX_VRAM_FOR_THIS_GROUP, CFG_FOR_THIS_GROUP, [{"TRAIN_PROFILE_NAME": (VRAM_MAX_FOR_PROFILE, TRAIN_PROFILE)}....])]
    # this means we have two sets of vRAM requirements, a top level one to tell us, when looping, if we just skip over this set all together
    # and an internal vRAM amount PER train profile, so if we are picking the best train profile we choose the one which matches our VRAM Requirements the most.
    profile_mappings = render_configs_and_profiles(gpus)
    click.echo("Please choose a system profile to use")
    click.echo("[0] Choose for me (Hardware defaults)")
    for i, value in enumerate(PROFILES):
        click.echo(f"[{i+1}] {value}")
    profile_selection = click.prompt(
        "Enter the number of your choice [hit enter for the hardware defaults profile]",
        type=int,
        default=0,
    )
    # if they want us to choose for them:
    # read vRAM, and map it to the profile which best matches.
    train = None
    cfg = None
    if profile_selection == 0:
        train, cfg = choose_configuration_for_hardware(
            gpus=gpus, profile_mappings=profile_mappings
        )
    else:
        train, cfg = prompt_for_configuration(
            profile_mappings=profile_mappings, profile_selection=profile_selection
        )
    if train is not None and cfg is not None:
        cfg.train = train
        ctx.obj.config = cfg
        write_config(cfg)
    else:
        logger.warn(
            "You tried to initialize a profile with a CPU only machine, please use the defaults provided by `ilab config init`"
        )


# choose_configuration_for_hardware takes the amnt of GPUs on the system, and the profile mappings of vram -> train profile + cfg and returns the ones that
# match the vRAM requirements.
def choose_configuration_for_hardware(
    gpus: int, profile_mappings
) -> tuple[_train, Config]:
    total_vram = 0
    for gpu in range(gpus):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu)
        gpu_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        total_vram += int(gpu_info.total)
    total_vram = convert_bytes_to_proper_mag(total_vram)[0]
    # total vram can be used to pick a profile, the profiles should be mapped least to greatest in terms of available vram.
    train = None
    cfg = None
    if total_vram > 0:
        # this is a map of the maximum amount of vram for the config before moving onto the next one
        for config_and_profile in profile_mappings:
            # if the systems vram is < total vram for any of these profiles, then our profile is in here.
            if total_vram < config_and_profile[0]:
                for profile_names_and_values in config_and_profile[2]:
                    if train is not None:
                        break
                    for vram_and_profile in list(profile_names_and_values.values()):
                        if total_vram < vram_and_profile[0]:
                            train = vram_and_profile[1]
                            break
                cfg = config_and_profile[1]
    return train, cfg


def prompt_for_configuration(
    profile_mappings: list[tuple[int, Config, list[dict[str, tuple[int, _train]]]]],
    profile_selection: int,
) -> tuple[_train, Config]:
    cfg = profile_mappings[profile_selection - 1][1]
    train = None
    names_and_profiles_list = profile_mappings[profile_selection - 1][2]
    if len(names_and_profiles_list) > 1:
        click.echo(
            "Please choose the GPU Training profile that best matches your system"
        )
        for ind, name_and_profile in enumerate(names_and_profiles_list):
            for name in name_and_profile:
                click.echo(f"[{ind}] {name}")
        train_profile_selection = click.prompt(
            "Enter the number of your choice [hit enter for the hardware defaults train profile]",
            type=int,
            default=0,
        )
        train = list(names_and_profiles_list[train_profile_selection].items())[0][1][
            1
        ]  # [1]
    else:
        train = list(names_and_profiles_list[0].items())[0][1][1]
    return train, cfg
