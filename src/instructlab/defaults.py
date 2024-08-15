# SPDX-License-Identifier: Apache-2.0
# Standard
from os import path

# Third Party
from xdg_base_dirs import xdg_cache_home, xdg_config_home, xdg_data_home
import httpx

ILAB_PACKAGE_NAME = "instructlab"
CONFIG_FILENAME = "config.yaml"
CONFIG_VERSION = "1.0.0"

# Model families understood by ilab
MODEL_FAMILIES = {"merlinite", "mixtral"}

# Map model names to their family
MODEL_FAMILY_MAPPINGS = {
    "granite": "merlinite",
}

# Default log format
LOG_FORMAT = "%(levelname)s %(asctime)s %(name)s:%(lineno)d: %(message)s"


class STORAGE_DIR_NAMES:
    ILAB = "instructlab"
    DATASETS = "datasets"
    CHECKPOINTS = "checkpoints"
    OCI = "oci"
    MODELS = "models"
    TAXONOMY = "taxonomy"
    INTERNAL = (
        "internal"  # for storing all ilab-internal files the user doesn't need to see
    )
    CHATLOGS = "chatlogs"
    PHASED = "phased"


class _InstructlabDefaults:
    """
    Class that defines the default paths used by ilab.
    We define them this way so that they can be lazy-loaded and overridden by tests.
    This way, they are defined when they are read instead of when the module is imported.
    """

    # define static defaults up here
    API_KEY = "no_api_key"

    # ILAB_GLOBAL_CONFIG is the environment variable that can be used to override the default config
    # file. When set, the CLI will use the file specified in the environment variable as a sample to
    # generate the default config file.
    ILAB_GLOBAL_CONFIG = "ILAB_GLOBAL_CONFIG"
    ILAB_TRAIN_PROFILE_DIR = "ILAB_TRAIN_PROFILE_DIR"

    # TODO: Consolidate --model and --model-path into one --model-path flag since we always need a path now
    MODEL_NAME_OLD = "merlinite-7b-lab-Q4_K_M"
    MERLINITE_GGUF_REPO = "instructlab/merlinite-7b-lab-GGUF"
    GGUF_MODEL_NAME = "merlinite-7b-lab-Q4_K_M.gguf"
    MODEL_REPO = "instructlab/granite-7b-lab"
    JUDGE_MODEL_MT = "prometheus-eval/prometheus-8x7b-v2.0"
    TAXONOMY_REPO = "https://github.com/instructlab/taxonomy.git"
    TAXONOMY_BASE = "origin/main"
    MAX_CONTEXT_SIZE = 4096
    # TODO: these constants should be removed, they should not leak out
    NUM_CPUS = 10
    CHUNK_WORD_COUNT = 1000
    CONNECTION_TIMEOUT = httpx.Timeout(timeout=30.0)
    # use spawn start method, fork is not thread-safe
    MULTIPROCESSING_START_METHOD = "spawn"
    SDG_PIPELINE = "simple"
    SDG_SCALE_FACTOR = 30

    # When otherwise unknown, ilab uses this as the default family
    MODEL_FAMILY = "merlinite"
    ADDITIONAL_ARGS_DEFAULTS = {
        "learning_rate": 2e-5,
        "warmup_steps": 25,
        "random_seed": 42,
        "node_rank": 0,
        "nnodes": 1,
        "rdzv_id": 123,
        "rdzv_endpoint": "127.0.0.1:12222",
        "deepspeed_cpu_offload_optimizer_ratio": 1,
        "deepspeed_cpu_offload_optimizer_pin_memory": False,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }

    def __init__(self):
        self._reset()

    def _reset(self):
        """
        Utility function which is mostly used for testing purposes to clear the cache.
        Otherwise, all tests will used cached temporary directories and cause errors.
        """
        self._cache_home = path.join(xdg_cache_home(), ILAB_PACKAGE_NAME)
        self._config_dir = path.join(xdg_config_home(), ILAB_PACKAGE_NAME)
        self._data_dir = path.join(xdg_data_home(), ILAB_PACKAGE_NAME)

    @property
    def CHECKPOINTS_DIR(self) -> str:
        return path.join(self._data_dir, STORAGE_DIR_NAMES.CHECKPOINTS)

    @property
    def OCI_DIR(self) -> str:
        return path.join(self._cache_home, STORAGE_DIR_NAMES.OCI)

    @property
    def DATASETS_DIR(self) -> str:
        return path.join(self._data_dir, STORAGE_DIR_NAMES.DATASETS)

    @property
    def CONFIG_FILE(self) -> str:
        return path.join(self._config_dir, CONFIG_FILENAME)

    @property
    def MODELS_DIR(self) -> str:
        return path.join(self._cache_home, STORAGE_DIR_NAMES.MODELS)

    @property
    def DEFAULT_MODEL(self) -> str:
        return path.join(self.MODELS_DIR, self.GGUF_MODEL_NAME)

    @property
    def DEFAULT_JUDGE_MODEL(self) -> str:
        return path.join(self.MODELS_DIR, self.JUDGE_MODEL_MT)

    @property
    def TAXONOMY_DIR(self) -> str:
        return path.join(self._data_dir, STORAGE_DIR_NAMES.TAXONOMY)

    @property
    def CHATLOGS_DIR(self) -> str:
        return path.join(self._data_dir, STORAGE_DIR_NAMES.CHATLOGS)

    @property
    def PHASED_DIR(self) -> str:
        return path.join(self._data_dir, STORAGE_DIR_NAMES.PHASED)

    @property
    def INTERNAL_DIR(self) -> str:
        """
        This directory is used for storing all misc. files that the user doesn't need to see.

        For example, during training we output an intermediate dataset with the tokenized
        instructions and the generated responses.
        Usually this gets outputted into /dev/shm, however this may not be an option on every system,
        so we would store it in here as a fall-back.
        """
        return path.join(self._data_dir, STORAGE_DIR_NAMES.INTERNAL)

    @property
    def PROMPT_FILE(self) -> str:
        return path.join(self.INTERNAL_DIR, "prompt.txt")

    @property
    def SEED_FILE(self) -> str:
        return path.join(self.INTERNAL_DIR, "seed_tasks.json")

    @property
    def EVAL_DATA_DIR(self) -> str:
        return path.join(self.INTERNAL_DIR, "eval_data")

    @property
    def TRAIN_CONFIG_DIR(self) -> str:
        return path.join(self.INTERNAL_DIR, "train_configuration")

    @property
    def TRAIN_PROFILE_DIR(self) -> str:
        return path.join(self.TRAIN_CONFIG_DIR, "profiles")

    @property
    def TRAIN_ADDITIONAL_OPTIONS_DIR(self) -> str:
        return path.join(self.TRAIN_CONFIG_DIR, "additional")

    @property
    def TRAIN_ADDITIONAL_OPTIONS_FILE(self) -> str:
        return path.join(self.TRAIN_ADDITIONAL_OPTIONS_DIR, "additional_args.yaml")

    @property
    def TRAIN_DEFAULT_PROFILE(self) -> str:
        return path.join(self.TRAIN_PROFILE_DIR, "default.yaml")

    @property
    def TRAIN_A100_H100_X4_PROFILE(self) -> str:
        return path.join(self.TRAIN_PROFILE_DIR, "A100_H100_x4.yaml")

    @property
    def TRAIN_A100_H100_X8_PROFILE(self) -> str:
        return path.join(self.TRAIN_PROFILE_DIR, "A100_H100_x8.yaml")

    @property
    def TRAIN_A100_H100_X2_PROFILE(self) -> str:
        return path.join(self.TRAIN_PROFILE_DIR, "A100_H100_x2.yaml")

    @property
    def TRAIN_L40_X8_PROFILE(self) -> str:
        return path.join(self.TRAIN_PROFILE_DIR, "L40_x8.yaml")

    @property
    def TRAIN_L40_X4_PROFILE(self) -> str:
        return path.join(self.TRAIN_PROFILE_DIR, "L40_x4.yaml")

    @property
    def TRAIN_L4_X8_PROFILE(self) -> str:
        return path.join(self.TRAIN_PROFILE_DIR, "L4_x8.yaml")


DEFAULTS = _InstructlabDefaults()
