# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import logging
import os
import time
import nvidia_smi

# Third Party
import click

# First Party
from instructlab import utils
from instructlab import clickext
from instructlab.configuration import render_configs_and_profiles, DEFAULTS, Config, _train, write_config
from instructlab.model.backends.backends import is_model_gguf, is_model_safetensors
from instructlab.utils import convert_bytes_to_proper_mag, print_table

logger = logging.getLogger(__name__)

PROFILES = ["Single Consumer GPU", "Multi Consumer GPU", "Single Server GPU", "Multi Server GPU", "MacOS"]

@click.command(name="set")
@clickext.display_params
@click.pass_context
def set(ctx):
    gpus = 0
    if utils.are_nvidia_gpus_available():
        gpus = nvidia_smi.nvmlDeviceGetCount()
    elif not utils.is_macos_with_m_chip():
        gpus = int(click.prompt("How many Dedicated GPUs do you have?"))
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
    if profile_selection == 0:
        total_vram = 0    
        for gpu in range(gpus):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu)
            gpu_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            total_vram += int(gpu_info.total)
    else:
        cfg = profile_mappings[profile_selection-1][0]
        names_and_profiles_list = profile_mappings[profile_selection-1][1]
        if len(names_and_profiles_list) > 1:
            click.echo("Please choose the GPU Training profile that best matches your system")
            for ind, name_and_profile in enumerate(names_and_profiles_list):
                for name, profile in name_and_profile.items():
                    click.echo(f"[{ind}] {name}")
            train_profile_selection = click.prompt(
                "Enter the number of your choice [hit enter for the hardware defaults train profile]",
                type=int,
                default=0,
            )
            train = (list(names_and_profiles_list[train_profile_selection].items())[0][1]) #[1]
        else:
            train = (list(names_and_profiles_list[0].items())[0][1])
        cfg.train = train
        ctx.obj.config = cfg
        write_config(cfg)
        
           
