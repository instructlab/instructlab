# Standard
from math import floor
import logging
import os

# First Party
from instructlab import utils
from instructlab.configuration import (
    DEFAULTS,
    Config,
    configs_exist,
    ensure_storage_directories_exist,
    get_default_config,
    profiles_exist,
    read_config,
    recreate_system_profiles,
    write_config,
)
from instructlab.defaults import DEFAULT_INDENT
from instructlab.utils import convert_bytes_to_proper_mag

logger = logging.getLogger(__name__)


def initialize_config(
    cfg: Config | None = None,
    profile: os.PathLike | None = None,
    overwrite_profile: bool = False,
    fresh_install: bool = False,
    write_to_disk: bool = True,
) -> tuple[Config | None, dict[str, list[tuple[list[str], str]]], bool]:
    """
    initialize_config is a standalone way to get an auto-detected InstructLab configuration file
    This will check your hardware and automatically select a profile for you. If none match, the default
    configuration will be returned.

    This function optionally takes some arguments if being called by the CLI. Leaving all of these blank will assume you want a fresh config.


    The return values are a Config object, a list of the parsed profiles on disk, and a bool indicating if we selected a profile.
    Most of these return values are used in the CLI.

    Args:
      cfg: Config | None
      profile: Path | None
      overwrite_profile: bool
      fresh_install: bool
      write_to_disk: bool

    Returns:
       cfg: Config
       arch_family_processors: dict[str, list[tuple[list[str], str]]]
       is_default_config: bool

    """
    cfg = get_default_config() if cfg is None else cfg
    is_default_config = True
    arch_family_processors: dict[str, list[tuple[list[str], str]]] = {}
    fresh_install = fresh_install or ensure_storage_directories_exist()
    overwrite_profiles = (
        overwrite_profile or configs_exist() or profiles_exist(fresh_install)
    )
    if overwrite_profiles:
        logger.info(
            f"\nGenerating config file and profiles:\n{DEFAULT_INDENT}{DEFAULTS.CONFIG_FILE}\n{DEFAULT_INDENT}{DEFAULTS.SYSTEM_PROFILE_DIR}\n"
        )
        recreate_system_profiles(overwrite=True)
    else:
        print(f"\nGenerating config file:\n{DEFAULT_INDENT}{DEFAULTS.CONFIG_FILE}\n")
    if profile is not None:
        cfg = read_config(profile)
        is_default_config = False
    else:
        new_cfg, arch_family_processors = hw_auto_detect()
        if new_cfg is not None:
            is_default_config = False
            cfg = new_cfg
    if write_to_disk:
        write_config(cfg)

    if not is_default_config:
        utils.print_init_success()
    return cfg, arch_family_processors, is_default_config


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


# hw_auto_detect looks at a user's GPUs or CPU configuration and chooses the system profile which matches your system
def hw_auto_detect() -> tuple[Config | None, dict[str, list[tuple[list[str], str]]]]:
    """
    hw_auto_detect looks at a user's GPUs or CPU configuration and chooses the system profile which matches your system
    """

    (
        chip_name,
        vram,
        gpus,
    ) = get_gpu_or_cpu()

    return walk_and_choose_system_profile(vram, gpus, chip_name)


def walk_and_choose_system_profile(
    vram: int, gpus: int, chip_name: str
) -> tuple[Config | None, dict[str, list[tuple[list[str], str]]]]:
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
        processor = " ".join(full_chip_name).upper()
        print(
            f"We have detected the {processor} profile as an exact match for your system."
        )

    return cfg, arch_family_processors


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
        logger.info("Detecting hardware...")
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
