# SPDX-License-Identifier: Apache-2.0

# Standard
from math import floor
import logging

# First Party
from instructlab.configuration import _train, get_profile_mappings
from instructlab.utils import convert_bytes_to_proper_mag

logger = logging.getLogger(__name__)


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
        logger.info("Detecting Hardware...")
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
                raise ValueError("Unable to retrieve train profiles")

            if (
                chosen_vram is None
                or chosen_vram > gpu_count_configs["vram_and_config"]["vram"]
            ) and vram < gpu_count_configs["vram_and_config"]["vram"]:
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
            raise ValueError("Unable to retrieve train profiles")
        # if GPU name and number matches, this is a direct match
        if (
            gpu_count == gpu_count_configs["gpu_count"]
            and vram == gpu_count_configs["vram_and_config"]["vram"]
        ):
            train = gpu_count_configs["vram_and_config"]["config"]
            matched = True
            break
    return matched, train
