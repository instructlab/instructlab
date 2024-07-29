# SPDX-License-Identifier: Apache-2.0

# Standard
from os import path
from re import match
from typing import Optional
import os
import sys
import typing

# Third Party
# pylint: disable=ungrouped-imports
from instructlab.training import (
    DeepSpeedOptions,
    LoraOptions,
    TorchrunArgs,
    TrainingArgs,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)
import click
import httpx
import platformdirs
import yaml

# Local
from . import log

ILAB_PACKAGE_NAME = "instructlab"
CONFIG_FILENAME = "config.yaml"
CONFIG_VERSION = "1.0.0"


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
    SDG_SCALE_FACTOR = 30

    # When otherwise unknown, ilab uses this as the default family
    MODEL_FAMILY = "merlinite"

    def __init__(self):
        self._reset()

    def _reset(self):
        """
        Utility function which is mostly used for testing purposes to clear the cache.
        Otherwise, all tests will used cached temporary directories and cause errors.
        """
        pd = platformdirs.PlatformDirs(appname=ILAB_PACKAGE_NAME)
        self._cache_home = pd.user_cache_dir
        self._config_dir = pd.user_config_dir
        self._data_dir = pd.user_data_dir

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
    def TAXONOMY_DIR(self) -> str:
        return path.join(self._data_dir, STORAGE_DIR_NAMES.TAXONOMY)

    @property
    def CHATLOGS_DIR(self) -> str:
        return path.join(self._data_dir, STORAGE_DIR_NAMES.CHATLOGS)

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


DEFAULTS = _InstructlabDefaults()


# Model families understood by ilab
MODEL_FAMILIES = {"merlinite", "mixtral"}

# Map model names to their family
MODEL_FAMILY_MAPPINGS = {
    "granite": "merlinite",
}


class ConfigException(Exception):
    """An exception that a configuration file has an error."""


class _general(BaseModel):
    """Class describing various top-level configuration options for all commands."""

    # model configuration
    model_config = ConfigDict(extra="ignore")

    # additional fields with defaults
    log_level: StrictStr = "INFO"
    debug_level: int = 0

    @field_validator("log_level")
    def validate_log_level(cls, v):
        # TODO: remove 'valid_levels' once we switch to support Python 3.11+ and call
        # "logging.getLevelNamesMapping()" instead
        valid_levels = [
            "DEBUG",
            "INFO",
            "WARNING",
            "WARN",
            "FATAL",
            "CRITICAL",
            "ERROR",
            "NOTSET",
        ]
        if v.upper() not in valid_levels:
            raise ValueError(
                f"'{v}' is not a valid log level name. valid levels: {valid_levels}"
            )
        return v.upper()

    @model_validator(mode="after")
    def after_debug_level(self):
        # set debug level when log level is DEBUG
        if self.log_level == "DEBUG" and self.debug_level == 0:
            self.debug_level = 1
        return self


class _chat(BaseModel):
    """Class describing configuration of the chat sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore")

    # required fields
    model: str

    # additional fields with defaults
    vi_mode: bool = False
    visible_overflow: bool = True
    context: str = "default"
    session: typing.Optional[str] = None
    logs_dir: str = Field(
        default_factory=lambda: DEFAULTS.CHATLOGS_DIR
    )  # use a lambda to avoid caching
    greedy_mode: bool = False
    max_tokens: typing.Optional[int] = None


class _serve_vllm(BaseModel):
    """Class describing configuration of vllm serving backend."""

    llm_family: str = ""
    # arguments to pass into vllm process
    vllm_args: list[str] | None = None


class _serve_llama_cpp(BaseModel):
    """Class describing configuration of llama-cpp serving backend."""

    gpu_layers: int = -1
    max_ctx_size: PositiveInt = 4096
    llm_family: str = ""


class _serve(BaseModel):
    """Class describing configuration of the serve sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    # vllm configuration
    vllm: _serve_vllm = _serve_vllm(
        llm_family="",
        vllm_args=[],
    )

    # llama-cpp configuration
    llama_cpp: _serve_llama_cpp = _serve_llama_cpp(
        gpu_layers=-1,
        max_ctx_size=4096,
        llm_family="",
    )

    # required fields
    model_path: StrictStr = Field(default_factory=lambda: DEFAULTS.DEFAULT_MODEL)
    # additional fields with defaults
    host_port: StrictStr = "127.0.0.1:8000"
    chat_template: Optional[str] = None
    backend: Optional[str] = (
        None  # we don't set a default value here since it's auto-detected
    )

    def api_base(self):
        """Returns server API URL, based on the configured host and port"""
        return get_api_base(self.host_port)


class _generate(BaseModel):
    """Class describing configuration of the generate sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore")

    # required fields
    model: StrictStr
    taxonomy_path: StrictStr
    taxonomy_base: StrictStr

    # additional fields with defaults
    teacher: _serve = Field(default_factory=_serve)
    num_cpus: PositiveInt = DEFAULTS.NUM_CPUS
    chunk_word_count: PositiveInt = DEFAULTS.CHUNK_WORD_COUNT
    # DEPRECATED: see sdg_scale_factor instead
    # Left in place so that we can still detect and give a warning if its
    # specified in an old configuration file.
    num_instructions: Optional[int] = Field(
        default=-1, deprecated="see 'sdg_scale_factor' instead", exclude=True
    )
    sdg_scale_factor: Optional[PositiveInt] = DEFAULTS.SDG_SCALE_FACTOR
    output_dir: StrictStr = Field(default_factory=lambda: DEFAULTS.DATASETS_DIR)
    prompt_file: StrictStr = Field(default_factory=lambda: DEFAULTS.PROMPT_FILE)
    seed_file: StrictStr = Field(default_factory=lambda: DEFAULTS.SEED_FILE)


class _mmlu(BaseModel):
    few_shots: int
    batch_size: str


class _mtbench(BaseModel):
    judge_model: str
    output_dir: str
    max_workers: int


class _mtbenchbranch(BaseModel):
    taxonomy_path: str


class _mmlubranch(BaseModel):
    tasks_dir: str


class _evaluate(BaseModel):
    # model configuration
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    model: Optional[str] = None
    base_model: str
    branch: Optional[str] = None
    base_branch: Optional[str] = None
    gpus: Optional[int] = 1
    mmlu: _mmlu
    mmlu_branch: _mmlubranch
    mt_bench: _mtbench
    mt_bench_branch: _mtbenchbranch


class _train(BaseModel):
    # model configuration
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    model_path: str

    data_path: str
    ckpt_output_dir: str
    data_output_dir: str

    max_seq_len: int
    max_batch_len: int
    num_epochs: int
    effective_batch_size: int
    save_samples: int

    deepspeed_cpu_offload_optimizer: bool

    lora_rank: int
    lora_quantize_dtype: str | None

    is_padding_free: bool

    nproc_per_node: int

    additional_args: dict[str, typing.Any]


class Config(BaseModel):
    """Configuration for the InstructLab CLI."""

    # required fields
    chat: _chat
    generate: _generate
    serve: _serve
    version: str

    # additional fields with defaults
    general: _general = _general()

    # train configuration
    train: Optional[_train] = None

    # evaluate configuration
    evaluate: _evaluate

    # model configuration
    model_config = ConfigDict(extra="ignore")


def get_default_config() -> Config:
    """Generates default configuration for CLI"""
    return Config(
        version=CONFIG_VERSION,
        chat=_chat(
            model=DEFAULTS.DEFAULT_MODEL,
        ),
        generate=_generate(
            model=DEFAULTS.DEFAULT_MODEL,
            taxonomy_path=DEFAULTS.TAXONOMY_DIR,
            taxonomy_base=DEFAULTS.TAXONOMY_BASE,
        ),
        serve=_serve(
            model_path=DEFAULTS.DEFAULT_MODEL,
            llama_cpp=_serve_llama_cpp(
                gpu_layers=-1,
                max_ctx_size=4096,
                llm_family="",
            ),
            vllm=_serve_vllm(
                llm_family="",
                vllm_args=[],
            ),
        ),
        train=_train(
            model_path=DEFAULTS.MODEL_REPO,
            data_path=DEFAULTS.DATASETS_DIR,
            ckpt_output_dir=DEFAULTS.CHECKPOINTS_DIR,
            data_output_dir=DEFAULTS.INTERNAL_DIR,
            max_seq_len=4096,
            max_batch_len=10000,
            num_epochs=10,
            effective_batch_size=3840,
            save_samples=250000,
            lora_quantize_dtype="nf4",
            lora_rank=4,
            nproc_per_node=1,
            deepspeed_cpu_offload_optimizer=False,
            additional_args={},
            is_padding_free=False,
        ),
        evaluate=_evaluate(
            base_model=DEFAULTS.MODEL_REPO,
            mt_bench=_mtbench(
                judge_model=DEFAULTS.JUDGE_MODEL_MT,
                output_dir=DEFAULTS.EVAL_DATA_DIR,
                max_workers=16,
            ),
            mmlu=_mmlu(
                few_shots=2,
                batch_size="auto",
            ),
            mt_bench_branch=_mtbenchbranch(taxonomy_path=DEFAULTS.TAXONOMY_DIR),
            mmlu_branch=_mmlubranch(tasks_dir=DEFAULTS.DATASETS_DIR),
        ),
    )


def read_train_profile(train_file) -> _train:
    try:
        with open(train_file, "r", encoding="utf-8") as yamlfile:
            content = yaml.safe_load(yamlfile)
            return _train(**content)
    except ValidationError as exc:
        msg = f"{exc.error_count()} errors in {train_file}:\n"
        for err in exc.errors():
            msg += (
                "- "
                + err.get("type", "")
                + " "
                + "->".join(err.get("loc", ""))  # type: ignore
                + ": "
                + err.get("msg", "").lower()
                + "\n"
            )
        raise ConfigException(msg) from exc


def read_config(
    config_file: str | os.PathLike[str] | None = None,
) -> Config:
    """Reads configuration from disk."""
    config_file = DEFAULTS.CONFIG_FILE if config_file is None else config_file
    try:
        with open(config_file, "r", encoding="utf-8") as yamlfile:
            content = yaml.safe_load(yamlfile)
            return Config(**content)
    except ValidationError as exc:
        msg = f"{exc.error_count()} errors in {config_file}:\n"
        for err in exc.errors():
            msg += (
                "- "
                + err.get("type", "")
                + " "
                + "->".join(err.get("loc", ""))  # type: ignore
                + ": "
                + err.get("msg", "").lower()
                + "\n"
            )
        raise ConfigException(msg) from exc


def get_dict(cfg: Config) -> dict[str, typing.Any]:
    """Returns configuration as a dictionary"""
    return cfg.model_dump()


def write_config(cfg: Config, config_file: typing.Optional[str] = None) -> None:
    """Writes configuration to a disk"""
    config_file = DEFAULTS.CONFIG_FILE if config_file is None else config_file
    with open(config_file, "w", encoding="utf-8") as yamlfile:
        d = cfg.model_dump_json()
        loaded = yaml.load(d, Loader=yaml.SafeLoader)
        yaml.dump(loaded, stream=yamlfile)


def get_api_base(host_port: str) -> str:
    """Returns server API URL, based on the provided host_port"""
    return f"http://{host_port}/v1"


def get_model_family(forced, model_path):
    forced = MODEL_FAMILY_MAPPINGS.get(forced, forced)
    if forced and forced.lower() not in MODEL_FAMILIES:
        raise ConfigException(f"Unknown model family: {forced}")

    # Try to guess the model family based on the model's filename
    guess = match(r"^\w*", path.basename(model_path)).group(0).lower()
    guess = MODEL_FAMILY_MAPPINGS.get(guess, guess)

    return guess if guess in MODEL_FAMILIES else DEFAULTS.MODEL_FAMILY


def ensure_storage_directories_exist():
    """
    Ensures that the default directories used by ilab exist.
    """
    dirs_to_make = [
        DEFAULTS._cache_home,
        DEFAULTS._config_dir,
        DEFAULTS._data_dir,
        DEFAULTS.CHATLOGS_DIR,
        DEFAULTS.CHECKPOINTS_DIR,
        DEFAULTS.OCI_DIR,
        DEFAULTS.DATASETS_DIR,
        DEFAULTS.EVAL_DATA_DIR,
        DEFAULTS.INTERNAL_DIR,
        DEFAULTS.MODELS_DIR,
        DEFAULTS.TAXONOMY_DIR,
        DEFAULTS.TRAIN_CONFIG_DIR,
        DEFAULTS.TRAIN_PROFILE_DIR,
        DEFAULTS.TRAIN_ADDITIONAL_OPTIONS_DIR,
    ]

    for dirpath in dirs_to_make:
        os.makedirs(dirpath, exist_ok=True)

    additional_args_and_defaults = {
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

    # create exper_args file for users to see/edit
    with open(DEFAULTS.TRAIN_ADDITIONAL_OPTIONS_FILE, "w", encoding="utf-8") as outfile:
        yaml.dump(additional_args_and_defaults, outfile)


class Lab:
    """Lab object holds high-level information about ilab CLI"""

    def __init__(
        self,
        config_obj: Config,
        config_file: str | os.PathLike[str] | None,
        error_msg: str | None,
    ) -> None:
        self.config = config_obj
        self.config_file = config_file
        self.error_msg = error_msg

    def ensure_config(self, ctx: click.Context) -> None:
        """Ensure that a config was loaded

        The init() function does not have access to the name of 2nd level
        subcommands. It only sees "config" for the nested subcommand
        `ilab config init`. First level subcommand functions call this
        method when they need a config for one of their subcommands.
        """
        if self.error_msg is not None:
            ctx.fail(self.error_msg)


def init(
    ctx: click.Context, config_file: str | os.PathLike[str], debug_level: int = 0
) -> None:
    config_obj: Config
    error_msg: str | None = None
    ensure_storage_directories_exist()
    if config_file == "DEFAULT":
        config_obj = get_default_config()
    elif os.path.isfile(config_file):
        try:
            config_obj = read_config(config_file)
        except ConfigException as e:
            # delayed, so ilab config init can override a broken config.
            config_obj = get_default_config()
            error_msg = str(e)
    else:
        config_obj = get_default_config()
        error_msg = (
            f"`{config_file}` does not exist or is not a readable file.\n"
            "Please run `ilab config init` or point to a valid configuration "
            "file using `--config=<path>`."
        )

    # special case: --help should always work
    if config_obj is None and "--help" in sys.argv[1:]:
        config_obj = get_default_config()
        error_msg = None

    ctx.obj = Lab(config_obj, config_file, error_msg)
    if config_obj is not None:
        ctx.default_map = get_dict(config_obj)

        # --verbose option overrides config file
        if debug_level > 0:
            config_obj.general.log_level = "DEBUG"
            config_obj.general.debug_level = debug_level

        # setup logging
        log.configure_logging(
            log_level=config_obj.general.log_level.upper(),
            debug_level=config_obj.general.debug_level,
        )
        # subtly get the additional args per cfg section
        # if any are missing, add in sane defaults
        train_additional = ctx.default_map["train"]["additional_args"]
        ctx.default_map["train"]["additional_args"] = finish_additional_train_args(
            train_additional
        )
    else:
        ctx.default_map = None


def map_train_to_library(params):
    # first do a lazy unwrap into the respective options
    train_args = TrainingArgs(**params)
    torch_args = TorchrunArgs(**params)

    ds_args = DeepSpeedOptions(
        cpu_offload_optimizer=params["deepspeed_cpu_offload_optimizer"],
        cpu_offload_optimizer_ratio=params["deepspeed_cpu_offload_optimizer_ratio"],
        cpu_offload_optimizer_pin_memory=params[
            "deepspeed_cpu_offload_optimizer_pin_memory"
        ],
    )

    lora_args = LoraOptions(
        rank=params["lora_rank"],
        alpha=params["lora_alpha"],
        dropout=params["lora_dropout"],
        target_modules=params["lora_target_modules"],
        quantize_data_type=params["lora_quantize_dtype"],
    )

    train_args.deepspeed_options = ds_args
    train_args.lora = lora_args

    return train_args, torch_args


def finish_additional_train_args(current_additional):
    with open(
        DEFAULTS.TRAIN_ADDITIONAL_OPTIONS_FILE, "r", encoding="utf-8"
    ) as yamlfile:
        content = yaml.safe_load(yamlfile)
    for key, val in content.items():
        if key not in current_additional:
            current_additional[key] = val

    return current_additional
