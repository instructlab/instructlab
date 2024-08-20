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
from instructlab.configuration import render_configs_and_profiles, DEFAULTS
from instructlab.model.backends.backends import is_model_gguf, is_model_safetensors
from instructlab.utils import convert_bytes_to_proper_mag, print_table

logger = logging.getLogger(__name__)


@click.command(name="set")
@clickext.display_params
@click.command()
@click.argument(
    "profile",
    nargs=-1,
   #default="default",
    #type=click.Choice(["default", PROFILE_MAPPINGS.keys()]),
)
def set(profile: str):
    if profile == "":
        #cfg = get_profile(profile)
        print()
    else:
        gpus = 0
        if utils.are_nvidia_gpus_available():
            gpus = nvidia_smi.nvmlDeviceGetCount()
        elif not utils.is_macos_with_m_chip():
            gpus = int(click.prompt("How many Dedicated GPUs do you have?"))
        profile_mappings = render_configs_and_profiles(gpus)    
        vals = list(profile_mappings.values())
        click.echo("Please choose a system profile to use")
        click.echo("[0] Choose for me (Hardware defaults)")
        for i, value in enumerate(vals):
            entry = dict(value)
            click.echo(f"[{i+1}] {list(entry.keys())[0]}")
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
        
           
