# SPDX-License-Identifier: Apache-2.0

# Standard
from os import path
from re import match
from typing import Any, Optional, Union
import os
import sys
import textwrap
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
from pydantic_core import PydanticUndefined
from ruamel.yaml import YAML, CommentedMap
from typing_extensions import deprecated as Deprecated
import click
import httpx
import platformdirs

# Local
from . import log

ILAB_PACKAGE_NAME = "instructlab"
CONFIG_FILENAME = "config.yaml"
CONFIG_VERSION = "1.0.0"

# Initialize ruamel.yaml
yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)


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
    log_level: StrictStr = Field(default="INFO", description="Log level for logging.")
    debug_level: int = Field(default=0, description="Debug level for logging.")

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
    model: str = Field(
        default_factory=lambda: DEFAULTS.DEFAULT_MODEL,
        description="Model to be used for chatting with.",
    )
    # additional fields with defaults
    vi_mode: bool = Field(default=False, description="Enable vim keybindings for chat.")
    visible_overflow: bool = Field(
        default=True,
        description="Renders vertical overflow if enabled, displays elipses otherwise.",
    )
    context: str = Field(
        default="default",
        description="Predefined setting or environment that influences the behavior and responses of the chat assistant. Each context is associated with a specific prompt that guides the assistant on how to respond to user inputs.",
    )
    session: typing.Optional[str] = Field(
        default=None, description="Filepath of a dialog session file."
    )
    logs_dir: str = Field(
        default_factory=lambda: DEFAULTS.CHATLOGS_DIR,
        description="Directory where chat logs are stored.",
    )  # use a lambda to avoid caching
    greedy_mode: bool = Field(
        default=False,
        description="Sets temperature to 0 if enabled, leading to more deterministic responses.",
    )
    max_tokens: typing.Optional[int] = Field(
        default=None,
        description="The maximum number of tokens that can be generated in the chat completion. Be aware that larger values use more memory.",
    )

    @model_validator(mode="before")
    def fill_defaults(
        cls, values: typing.Dict[str, typing.Any]
    ) -> typing.Dict[str, typing.Any]:
        defaults = {"model": DEFAULTS.DEFAULT_MODEL}
        return finish_cfg_section(defaults, values)


class _serve_vllm(BaseModel):
    """Class describing configuration of vllm serving backend."""

    llm_family: str = Field(
        default="",  # TODO: convert to None and use a pattern to validate
        description="Large Language Model Family",
        examples=["merlinite", "granite"],
    )
    max_startup_attempts: int | None = Field(
        default=120,
        description="Maximum number of attempts to start the vLLM server.",
    )
    gpus: Optional[int] = Field(default=None, description="Number of GPUs to use.")
    # arguments to pass into vllm process
    vllm_args: list[str] | None = Field(
        default_factory=list,
        description="vLLM specific arguments. All settings can be passed as a list of strings, see: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html",
        examples=[
            ["--dtype", "auto"],
            ["--lora-alpha", "32"],
        ],
    )


class _serve_llama_cpp(BaseModel):
    """Class describing configuration of llama-cpp serving backend."""

    gpu_layers: int = Field(
        default=-1,
        description="Number of model layers to offload to GPU. -1 means all layers.",
    )
    max_ctx_size: PositiveInt = Field(
        default=4096,
        description="Maximum number of tokens that can be processed by the model.",
    )
    llm_family: str = Field(
        default="",  # TODO: convert to None and use a pattern to validate
        description="Large Language Model Family",
        examples=["merlinite", "granite"],
    )


class _serve(BaseModel):
    """Class describing configuration of the serve sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore", protected_namespaces=())
    # vllm configuration
    vllm: _serve_vllm = Field(
        default_factory=lambda: _serve_vllm,
        description="vLLM serving settings.",
    )
    # llama-cpp configuration
    llama_cpp: _serve_llama_cpp = Field(
        default_factory=lambda: _serve_llama_cpp,
        description="llama-cpp serving settings.",
    )
    model_path: StrictStr = Field(
        default_factory=lambda: DEFAULTS.DEFAULT_MODEL,
        description="Directory where model to be served is stored.",
    )
    # additional fields with defaults
    host_port: StrictStr = Field(
        default="127.0.0.1:8000", description="Host and port to serve on."
    )
    chat_template: Optional[str] = Field(
        default=None,
        description="Chat template to supply to the model. Possible values: 'auto'(default), 'tokenizer', a path to a jinja2 file.",
        examples=[
            "auto",
            "tokenizer",
            "A filesystem path expressing the location of a custom template",
        ],
    )
    # we don't set a default value here since it's auto-detected
    backend: Optional[str] = Field(
        default=None,
        description="Serving backend to use to host the model.",
        examples=["vllm", "llama-cpp"],
        pattern="vllm|llama-cpp",
    )

    def api_base(self):
        """Returns server API URL, based on the configured host and port"""
        return get_api_base(self.host_port)

    @model_validator(mode="before")
    def fill_defaults(
        cls, values: typing.Dict[str, typing.Any]
    ) -> typing.Dict[str, typing.Any]:
        defaults = {
            "model_path": DEFAULTS.DEFAULT_MODEL,
            "llama_cpp": {"gpu_layers": -1, "max_ctx_size": 4096, "llm_family": ""},
            "vllm": {"llm_family": "", "vllm_args": [], "max_startup_attempts": 120},
        }
        return finish_cfg_section(defaults, values)


class _generate(BaseModel):
    """Class describing configuration of the generate sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore")
    pipeline: Optional[str] = Field(
        default=DEFAULTS.SDG_PIPELINE,
        description="Data generation pipeline to use. Available: 'simple', 'full', or a valid path to a directory of pipeline workflow YAML files. Note that 'full' requires a larger teacher model, Mixtral-8x7b.",
    )
    model: StrictStr = Field(
        default_factory=lambda: DEFAULTS.DEFAULT_MODEL,
        description="Teacher model that will be used to synthetically generate training data.",
    )
    taxonomy_path: StrictStr = Field(
        default_factory=lambda: DEFAULTS.TAXONOMY_DIR,
        description="Directory where taxonomy is stored and accessed from.",
    )
    taxonomy_base: StrictStr = Field(
        default=DEFAULTS.TAXONOMY_BASE,
        description="Branch of taxonomy used to calculate diff against.",
    )
    # additional fields with defaults
    teacher: _serve = Field(default_factory=_serve, description="Teacher configuration")
    num_cpus: PositiveInt = Field(
        default=DEFAULTS.NUM_CPUS,
        description="Number of CPU cores to use for generation.",
    )
    chunk_word_count: PositiveInt = Field(
        default=DEFAULTS.CHUNK_WORD_COUNT,
        description="Maximum number of words per chunk.",
    )
    # DEPRECATED: see sdg_scale_factor instead
    # Left in place so that we can still detect and give a warning if its
    # specified in an old configuration file.
    num_instructions: Optional[int] = Field(
        default=-1,
        description="Number of instructions to use",
        deprecated="see 'sdg_scale_factor' instead",
        exclude=True,
    )
    sdg_scale_factor: Optional[PositiveInt] = Field(
        default=DEFAULTS.SDG_SCALE_FACTOR,
        description="The total number of instructions to be generated.",
    )
    output_dir: StrictStr = Field(
        default_factory=lambda: DEFAULTS.DATASETS_DIR,
        description="Directory where generated datasets are stored.",
    )
    prompt_file: StrictStr = Field(
        default_factory=lambda: DEFAULTS.PROMPT_FILE,
        description="Path to prompt file to be used for generation.",
    )
    # TODO: remove this? It's not used anywhere, was removed by 19b9f4794f79ef81578c00c901bac3ee9db8c046
    seed_file: StrictStr = Field(
        description="Path to seed file to be used for generation.",
        default_factory=lambda: DEFAULTS.SEED_FILE,
        deprecated=True,
    )

    @model_validator(mode="before")
    def fill_defaults(
        cls, values: typing.Dict[str, typing.Any]
    ) -> typing.Dict[str, typing.Any]:
        defaults = {
            "model": DEFAULTS.DEFAULT_MODEL,
            "taxonomy_path": DEFAULTS.TAXONOMY_DIR,
            "taxonomy_base": DEFAULTS.TAXONOMY_BASE,
        }

        return finish_cfg_section(defaults, values)


class _mmlu(BaseModel):
    few_shots: int = Field(
        default=5,
        description="Number of question-answer pairs provided in the context preceding the question used for evaluation.",
    )
    batch_size: str | int = Field(
        default="auto",
        description="Batch size for evaluation. Valid values are a positive integer or 'auto' to select the largest batch size that will fit in memory.",
    )

    @model_validator(mode="before")
    def fill_defaults(
        cls, values: typing.Dict[str, typing.Any]
    ) -> typing.Dict[str, typing.Any]:
        defaults = {"few_shots": 5, "batch_size": "auto"}
        return finish_cfg_section(defaults, values)


class _mtbench(BaseModel):
    judge_model: str = Field(
        default=DEFAULTS.JUDGE_MODEL_MT,
        description="Directory where model to be used as judge is stored.",
    )
    output_dir: str = Field(
        default_factory=lambda: DEFAULTS.EVAL_DATA_DIR,
        description="Directory where evaluation results are stored.",
    )
    max_workers: int = Field(
        default=16,
        description="Number of workers to use for evaluation.",
    )

    @model_validator(mode="before")
    def fill_defaults(
        cls, values: typing.Dict[str, typing.Any]
    ) -> typing.Dict[str, typing.Any]:
        defaults = {
            "judge_model": DEFAULTS.JUDGE_MODEL_MT,
            "output_dir": DEFAULTS.EVAL_DATA_DIR,
            "max_workers": 16,
        }
        return finish_cfg_section(defaults, values)


class _mtbenchbranch(BaseModel):
    taxonomy_path: str = Field(
        default_factory=lambda: DEFAULTS.TAXONOMY_DIR,
        description="Path to where base taxonomy is stored.",
    )

    @model_validator(mode="before")
    def fill_defaults(
        cls, values: typing.Dict[str, typing.Any]
    ) -> typing.Dict[str, typing.Any]:
        defaults = {"taxonomy_path": DEFAULTS.TAXONOMY_DIR}

        return finish_cfg_section(defaults, values)


class _mmlubranch(BaseModel):
    tasks_dir: str = Field(
        default_factory=lambda: DEFAULTS.DATASETS_DIR,
        description="Directory where custom MMLU tasks are stored.",
    )

    @model_validator(mode="before")
    def fill_defaults(
        cls, values: typing.Dict[str, typing.Any]
    ) -> typing.Dict[str, typing.Any]:
        defaults = {"tasks_dir": DEFAULTS.DATASETS_DIR}
        return finish_cfg_section(defaults, values)


class _evaluate(BaseModel):
    # model configuration
    model_config = ConfigDict(extra="ignore", protected_namespaces=())
    model: Optional[str] = Field(
        default=None,
        description="Model to be used for evaluation.",
    )
    base_model: str = Field(
        default=DEFAULTS.MODEL_REPO,
        description="Directory where model to be used for evaluation is stored.",
    )
    branch: Optional[str] = Field(
        default=None,
        description="Taxonomy branch containing custom skills/knowledge that should be used for evaluation runs.",
    )
    base_branch: Optional[str] = Field(default=None, description="Base taxonomy branch")
    gpus: Optional[int] = Field(
        default=None, description="Number of GPUs to use for running evaluation."
    )
    mmlu: _mmlu = Field(
        default_factory=lambda: _mmlu,
        description="MMLU benchmarking settings",
    )
    mmlu_branch: _mmlubranch = Field(
        default_factory=lambda: DEFAULTS.DATASETS_DIR,
        description="Settings to run MMLU against a branch of taxonomy containing custom skills/knowledge used for training.",
    )
    mt_bench: _mtbench = Field(
        default_factory=lambda: _mtbench,
        description="Multi-turn benchmarking settings for skills.",
    )
    mt_bench_branch: _mtbenchbranch = Field(
        default_factory=lambda: DEFAULTS.TAXONOMY_DIR,
        description="Settings to run MT-Bench against a branch of taxonomy containing custom skills/knowledge used for training",
    )

    @model_validator(mode="before")
    def fill_defaults(
        cls, values: typing.Dict[str, typing.Any]
    ) -> typing.Dict[str, typing.Any]:
        defaults = {
            "base_model": DEFAULTS.MODEL_REPO,
            "mt_bench": {
                "judge_model": DEFAULTS.JUDGE_MODEL_MT,
                "output_dir": DEFAULTS.EVAL_DATA_DIR,
                "max_workers": 16,
            },
            "mmlu": {"few_shots": 5, "batch_size": "auto"},
            "mt_bench_branch": {"taxonomy_path": DEFAULTS.TAXONOMY_DIR},
            "mmlu_branch": {"tasks_dir": DEFAULTS.DATASETS_DIR},
        }
        return finish_cfg_section(defaults, values)


class _train(BaseModel):
    # model configuration
    model_config = ConfigDict(extra="ignore", protected_namespaces=())
    model_path: str = Field(
        default=DEFAULTS.MODEL_REPO,
        description="Directory where the model to be trained is stored.",
    )
    data_path: str = Field(
        default_factory=lambda: DEFAULTS.DATASETS_DIR,
        description="For the training library (primary training method), this specifies the path to the dataset file. For legacy training (MacOS/Linux), this specifies the path to the directory.",
    )
    ckpt_output_dir: str = Field(
        default_factory=lambda: DEFAULTS.CHECKPOINTS_DIR,
        description="Directory where periodic training checkpoints are stored.",
    )
    data_output_dir: str = Field(
        default_factory=lambda: DEFAULTS.INTERNAL_DIR,
        description="Directory where the processed training data is stored (post filtering/tokenization/masking).",
    )
    max_seq_len: int = Field(
        default=4096,
        description="Maximum sequence length to be included in the training set. Samples exceeding this length will be dropped.",
    )
    max_batch_len: int = Field(
        default=10000,
        description="Maximum tokens per gpu for each batch that will be handled in a single step. If running into out-of-memory errors, this value can be lowered but not below the `max_seq_len`.",
    )
    num_epochs: int = Field(
        default=10, description="Number of epochs to run training for."
    )
    effective_batch_size: int = Field(
        default=3840,
        description="The number of samples in a batch that the model should see before its parameters are updated.",
    )
    save_samples: int = Field(
        default=250000,
        description="Number of samples the model should see before saving a checkpoint.",
    )
    checkpoint_at_epoch: bool = Field(
        default=True, description="Save a checkpoint at the end of each epoch."
    )
    deepspeed_cpu_offload_optimizer: bool = Field(
        default=False, description="Allow CPU offload for deepspeed optimizer."
    )
    lora_rank: int | None = Field(
        default=None,
        description="Rank of low rank matrices to be used during training.",
    )
    lora_quantize_dtype: str | None = Field(
        default=None,
        description="The data type for quantization in LoRA training. Valid options are 'None' and 'nf4'.",
        examples=["nf4"],
    )
    is_padding_free: bool = Field(
        default=False,
        description="Boolean to indicate if the model being trained is a padding-free transformer model such as Granite.",
    )
    nproc_per_node: int = Field(
        default=1,
        description="Number of GPUs to use for training. This value is not supported in legacy training or MacOS.",
    )
    additional_args: dict[str, typing.Any] = Field(
        default_factory=dict,
        description="Additional arguments to pass to the training script. These arguments are passed as key-value pairs to the training script.",
    )
    # additional training configuration for
    # lab-multiphase training.
    # TODO: could move into its own object.
    # Not strictly necessary for a correct training object.
    phased_phase1_num_epochs: int | None = Field(
        default=None,
        gt=0,
        description="Number of epochs to run training for during phase1.",
    )
    # anything greater than 0 enables samples_per_save for the phase.
    phased_phase1_samples_per_save: int = Field(
        default=0,
        description="Number of samples the model should see before saving a checkpoint during phase1. Disabled when set to 0.",
        ge=0,
    )
    phased_phase1_effective_batch_size: int | None = Field(
        default=128,
        description="Phased phase1 effective batch size.",
    )
    phased_phase2_num_epochs: int | None = Field(
        default=None,
        gt=0,
        description="Number of epochs to run training for during phase2.",
    )
    # phased_phase2_samples_per_save is disabled when the value is 0.
    # anything greater than 0 enables samples_per_save for the phase.
    phased_phase2_samples_per_save: int = Field(
        default=0,
        ge=0,
        description="Number of samples the model should see before saving a checkpoint during phase2. Disabled when set to 0.",
    )
    phased_phase2_effective_batch_size: int | None = Field(
        default=3840, description="Phased phase2 effective batch size."
    )
    phased_mt_bench_judge: str | None = Field(
        default=None, description="Judge model path for phased MT-Bench evaluation."
    )

    @model_validator(mode="before")
    def fill_defaults(
        cls, values: typing.Dict[str, typing.Any]
    ) -> typing.Dict[str, typing.Any]:
        defaults = {
            "model_path": DEFAULTS.MODEL_REPO,
            "data_path": DEFAULTS.DATASETS_DIR,
            "ckpt_output_dir": DEFAULTS.CHECKPOINTS_DIR,
            "data_output_dir": DEFAULTS.INTERNAL_DIR,
            "max_seq_len": 4096,
            "max_batch_len": 10000,
            "num_epochs": 10,
            "effective_batch_size": 3840,
            "save_samples": 250000,
            "lora_quantize_dtype": None,
            "lora_rank": None,
            "nproc_per_node": 1,
            "deepspeed_cpu_offload_optimizer": False,
            "additional_args": {},
            "is_padding_free": False,
        }

        return finish_cfg_section(defaults, values)


class Config(BaseModel):
    """Configuration for the InstructLab CLI."""

    chat: _chat = Field(
        default_factory=_chat, description="Chat configuration section."
    )
    generate: _generate = Field(
        default_factory=_generate, description="Generate configuration section."
    )
    serve: _serve = Field(
        default_factory=_serve, description="Serve configuration section."
    )
    # additional fields with defaults
    general: _general = Field(
        default_factory=_general, description="General configuration section."
    )
    # train configuration
    train: _train = Field(
        default_factory=_train, description="Train configuration section."
    )
    # evaluate configuration
    evaluate: _evaluate = Field(
        default_factory=_evaluate, description="Evaluate configuration section."
    )
    # model configuration
    model_config = ConfigDict(extra="ignore")
    version: str = Field(
        default=CONFIG_VERSION,
        description="Configuration file structure version.",
        frozen=True,  # don't allow this to be changed anywhere in the code
    )

    @model_validator(mode="before")
    def fill_defaults(
        cls, values: typing.Dict[str, typing.Any]
    ) -> typing.Dict[str, typing.Any]:
        defaults = {
            "chat": {"model": DEFAULTS.DEFAULT_MODEL},
            "serve": {
                "model_path": DEFAULTS.DEFAULT_MODEL,
                "llama_cpp": {"gpu_layers": -1, "max_ctx_size": 4096, "llm_family": ""},
                "vllm": {"llm_family": "", "vllm_args": []},
            },
            "generate": {
                "model": DEFAULTS.DEFAULT_MODEL,
                "taxonomy_path": DEFAULTS.TAXONOMY_DIR,
                "taxonomy_base": DEFAULTS.TAXONOMY_BASE,
            },
            "evaluate": {
                "model": DEFAULTS.DEFAULT_MODEL,
                "taxonomy_path": DEFAULTS.TAXONOMY_DIR,
                "taxonomy_base": DEFAULTS.TAXONOMY_BASE,
            },
            "train": {
                "model_path": DEFAULTS.MODEL_REPO,
                "data_path": DEFAULTS.DATASETS_DIR,
                "ckpt_output_dir": DEFAULTS.CHECKPOINTS_DIR,
                "data_output_dir": DEFAULTS.INTERNAL_DIR,
                "max_seq_len": 4096,
                "max_batch_len": 10000,
                "num_epochs": 10,
                "effective_batch_size": 3840,
                "save_samples": 250000,
                "lora_quantize_dtype": None,
                "lora_rank": 4,
                "nproc_per_node": 1,
                "deepspeed_cpu_offload_optimizer": False,
                "additional_args": {},
                "is_padding_free": False,
            },
            "version": CONFIG_VERSION,
        }

        for key, val in defaults.items():
            if values.get(key) is None and val is not None:
                values[key] = val

        return values


def finish_cfg_section(
    defaults: dict, values: typing.Dict[str, typing.Any]
) -> typing.Dict[str, typing.Any]:
    for key, val in defaults.items():
        if values.get(key) is None and val is not None:
            values[key] = val

    return values


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
                max_startup_attempts=120,
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
            phased_phase1_num_epochs=10,
            phased_phase1_effective_batch_size=128,
            phased_phase2_num_epochs=10,
            phased_phase2_effective_batch_size=3840,
            phased_mt_bench_judge=DEFAULTS.DEFAULT_JUDGE_MODEL,
        ),
        evaluate=_evaluate(
            base_model=DEFAULTS.MODEL_REPO,
            mt_bench=_mtbench(
                judge_model=DEFAULTS.DEFAULT_JUDGE_MODEL,
                output_dir=DEFAULTS.EVAL_DATA_DIR,
                max_workers=16,
            ),
            mmlu=_mmlu(
                few_shots=5,
                batch_size="auto",
            ),
            mt_bench_branch=_mtbenchbranch(taxonomy_path=DEFAULTS.TAXONOMY_DIR),
            mmlu_branch=_mmlubranch(tasks_dir=DEFAULTS.DATASETS_DIR),
        ),
    )


EIGHT_GPU_TRAIN_AH = _train(
    additional_args={
        "warmup_steps": 25,
        "learning_rate": 2e-5,
        "lora_dropout": 0.1,
         "lora_alpha": 32,
    },
    ckpt_output_dir=os.path.join(DEFAULTS._data_dir, "checkpoints"),
    data_output_dir=os.path.join(DEFAULTS._data_dir, "internal"),
    data_path=os.path.join(DEFAULTS._data_dir, "datasets"),
    nproc_per_node=8,
    effective_batch_size=128,
    lora_quantize_dtype=None,
    lora_rank=0,
    max_batch_len=60000,
    max_seq_len=4096,
    save_samples=1000,
    deepspeed_cpu_offload_optimizer=False,
    is_padding_free=True,
    num_epochs=8,
    model_path=os.path.join(
        DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"
    ),
)

FOUR_GPU_TRAIN_AH = _train(
    additional_args={
        "warmup_steps": 25,
        "learning_rate": 2e-5,
        "lora_dropout": 0.1,
        "lora_alpha": 32,
    },
    ckpt_output_dir=os.path.join(DEFAULTS._data_dir, "checkpoints"),
    data_output_dir=os.path.join(DEFAULTS._data_dir, "internal"),
    data_path=os.path.join(DEFAULTS._data_dir, "datasets"),
    nproc_per_node=4,
    effective_batch_size=128,
    lora_quantize_dtype=None,
    lora_rank=0,
    max_batch_len=54000,
    max_seq_len=4096,
    save_samples=1000,
    deepspeed_cpu_offload_optimizer=False,
    is_padding_free=True,
    num_epochs=8,
    model_path=os.path.join(
        DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"
    ),
)

TWO_GPU_TRAIN_AH = _train(
    additional_args={
        "warmup_steps": 25,
        "learning_rate": 2e-5,
        "lora_dropout": 0.1,
        "lora_alpha": 32,
        "deepspeed_cpu_offload_optimizer_ratio": 1,
        "deepspeed_cpu_offload_optimizer_pin_memory": True,
    },
    ckpt_output_dir=os.path.join(DEFAULTS._data_dir, "checkpoints"),
    data_output_dir=os.path.join(DEFAULTS._data_dir, "internal"),
    data_path=os.path.join(DEFAULTS._data_dir, "datasets"),
    nproc_per_node=2,
    effective_batch_size=128,
    lora_quantize_dtype=None,
    lora_rank=0,
    max_batch_len=60000,
    max_seq_len=4096,
    save_samples=1000,
    deepspeed_cpu_offload_optimizer=True,
    is_padding_free=True,
    num_epochs=8,
    model_path=os.path.join(
        DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"
    ),
)

EIGHT_L_FORTY_GPU = _train(
    additional_args={
        "warmup_steps": 25,
        "learning_rate": 2e-5,
        "lora_dropout": 0.1,
        "lora_alpha": 32,
    },
    ckpt_output_dir=os.path.join(DEFAULTS._data_dir, "checkpoints"),
    data_output_dir=os.path.join(DEFAULTS._data_dir, "internal"),
    data_path=os.path.join(DEFAULTS._data_dir, "datasets"),
    nproc_per_node=8,
    effective_batch_size=128,
    lora_quantize_dtype=None,
    lora_rank=0,
    max_batch_len=30000,
    max_seq_len=4096,
    save_samples=1000,
    deepspeed_cpu_offload_optimizer=False,
    is_padding_free=True,
    num_epochs=8,
    model_path=os.path.join(
        DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"
    ),
)

FOUR_L_FORTY_GPU = _train(
    additional_args={
        "warmup_steps": 25,
        "learning_rate": 2e-5,
        "lora_dropout": 0.1,
        "lora_alpha": 32,
        "deepspeed_cpu_offload_optimizer_ratio": 1,
        "deepspeed_cpu_offload_optimizer_pin_memory": True,
    },
    ckpt_output_dir=os.path.join(DEFAULTS._data_dir, "checkpoints"),
    data_output_dir=os.path.join(DEFAULTS._data_dir, "internal"),
    data_path=os.path.join(DEFAULTS._data_dir, "datasets"),
    nproc_per_node=4,
    effective_batch_size=128,
    lora_quantize_dtype=None,
    lora_rank=0,
    max_batch_len=30000,
    max_seq_len=4096,
    save_samples=1000,
    num_epochs=8,
    deepspeed_cpu_offload_optimizer=True,
    is_padding_free=True,
    model_path=os.path.join(
        DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"
    ),
)

EIGHT_L_FOUR_GPU = _train(
    additional_args={
        "warmup_steps": 25,
        "learning_rate": 2e-5,
        "lora_dropout": 0.1,
        "lora_alpha": 32,
        "deepspeed_cpu_offload_optimizer_ratio": 1,
        "deepspeed_cpu_offload_optimizer_pin_memory": True,
    },
    ckpt_output_dir=os.path.join(DEFAULTS._data_dir, "checkpoints"),
    data_output_dir=os.path.join(DEFAULTS._data_dir, "internal"),
    data_path=os.path.join(DEFAULTS._data_dir, "datasets"),
    nproc_per_node=8,
    effective_batch_size=128,
    lora_quantize_dtype=None,
    lora_rank=0,
    max_batch_len=30000,
    max_seq_len=4096,
    save_samples=1000,
    num_epochs=8,
    deepspeed_cpu_offload_optimizer=True,
    is_padding_free=True,
    model_path=os.path.join(
        DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"
    ),
)

SINGLE_SERVER_GPU_TRAIN = _train(
    additional_args={
        "warmup_steps": 25,
        "learning_rate": 2e-5,
        "lora_dropout": 0.1,
        "lora_alpha": 32,
        "deepspeed_cpu_offload_optimizer_ratio": 1,
        "deepspeed_cpu_offload_optimizer_pin_memory": True,
    },
    ckpt_output_dir=os.path.join(DEFAULTS._data_dir, "checkpoints"),
    data_output_dir=os.path.join(DEFAULTS._data_dir, "internal"),
    data_path=os.path.join(DEFAULTS._data_dir, "datasets"),
    nproc_per_node=1,
    effective_batch_size=96,
    lora_quantize_dtype='nf4',
    lora_rank=2,
    max_batch_len=60000,
    max_seq_len=4096,
    save_samples=1000,
    deepspeed_cpu_offload_optimizer=True,
    is_padding_free=True,
    num_epochs=8,
    model_path=os.path.join(
        DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"
    ),
)

MACOS_TRAIN = _train(
    additional_args={
        "warmup_steps": 25,
        "learning_rate": 2e-5,
        "lora_dropout": 0.1,
        "lora_alpha": 32,
        "deepspeed_cpu_offload_optimizer_ratio": 1,
        "deepspeed_cpu_offload_optimizer_pin_memory": True,
    },
    ckpt_output_dir=os.path.join(DEFAULTS._data_dir, "checkpoints"),
    data_output_dir=os.path.join(DEFAULTS._data_dir, "internal"),
    data_path=os.path.join(DEFAULTS._data_dir, "datasets"),
    nproc_per_node=1,
    effective_batch_size=96,
    lora_quantize_dtype='nf4',
    lora_rank=2,
    max_batch_len=60000,
    max_seq_len=4096,
    save_samples=1000,
    deepspeed_cpu_offload_optimizer=True,
    is_padding_free=True,
    num_epochs=5,
    model_path=os.path.join(
        DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"
    ),
)

SINGLE_CONSUMER_GPU_TRAIN = _train(
    additional_args={
        "warmup_steps": 25,
        "learning_rate": 2e-5,
        "lora_dropout": 0.1,
        "lora_alpha": 32,
        "deepspeed_cpu_offload_optimizer_ratio": 1,
        "deepspeed_cpu_offload_optimizer_pin_memory": True,
    },
    ckpt_output_dir=os.path.join(DEFAULTS._data_dir, "checkpoints"),
    data_output_dir=os.path.join(DEFAULTS._data_dir, "internal"),
    data_path=os.path.join(DEFAULTS._data_dir, "datasets"),
    nproc_per_node=1,
    effective_batch_size=96,
    lora_quantize_dtype='nf4',
    lora_rank=2,
    max_batch_len=30000,
    max_seq_len=4096,
    save_samples=1000,
    deepspeed_cpu_offload_optimizer=True,
    is_padding_free=True,
    num_epochs=5,
    model_path=os.path.join(
        DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"
    ),
)


MULTI_CONSUMER_GPU_TRAIN = _train(
    additional_args={
        "warmup_steps": 25,
        "learning_rate": 2e-5,
        "lora_dropout": 0.1,
        "lora_alpha": 32,
        "deepspeed_cpu_offload_optimizer_ratio": 1,
        "deepspeed_cpu_offload_optimizer_pin_memory": True,
    },
    ckpt_output_dir=os.path.join(DEFAULTS._data_dir, "checkpoints"),
    data_output_dir=os.path.join(DEFAULTS._data_dir, "internal"),
    data_path=os.path.join(DEFAULTS._data_dir, "datasets"),
    nproc_per_node=2,
    effective_batch_size=96,
    lora_quantize_dtype='nf4',
    lora_rank=2,
    max_batch_len=30000,
    max_seq_len=4096,
    save_samples=1000,
    deepspeed_cpu_offload_optimizer=True,
    is_padding_free=True,
    num_epochs=5,
    model_path=os.path.join(
        DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"
    ),
)

def read_train_profile(train_file) -> _train:
    try:
        with open(train_file, "r", encoding="utf-8") as yamlfile:
            content = yaml.load(yamlfile)
            _expand_paths(content)
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
            content = yaml.load(yamlfile)
            _expand_paths(content)
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


def _expand_paths(content: dict | list):
    if isinstance(content, dict):
        for key, value in content.items():
            expanded_value = _expand_value(value)
            if expanded_value is not None:
                content[key] = expanded_value
    elif isinstance(content, list):
        for i, value in enumerate(content):
            expanded_value = _expand_value(value)
            if expanded_value is not None:
                content[i] = expanded_value


def _expand_value(value):
    if isinstance(value, str):
        expanded_value = os.path.expanduser(value)
        return os.path.expandvars(expanded_value)
    if isinstance(value, (dict, list)):
        _expand_paths(value)
    return None


def get_dict(cfg: Config) -> dict[str, typing.Any]:
    """Returns configuration as a dictionary"""
    return cfg.model_dump()


def write_config(cfg: Config, config_file: typing.Optional[str] = None) -> None:
    """Writes configuration to a disk"""
    config_file = DEFAULTS.CONFIG_FILE if config_file is None else config_file
    write_config_to_yaml(cfg, config_file)


def config_to_commented_map(
    cfg: Union[Config, BaseModel], indent: int = 0
) -> CommentedMap:
    """
    Convert a Pydantic model to a CommentedMap with comments derived from field descriptions.

    This function iterates through the fields of our Config model, converting each field to a
    CommentedMap entry. If a field is itself a model, the function handles it recursively.
    Comments are added to each field based on its description and default value.

    Args:
        cfg (Union[Config, BaseModel]): The Pydantic model instance to convert.
        indent (int, optional): The indentation level for nested fields. Defaults to 0.

    Returns:
        CommentedMap: A CommentedMap representation of the Config model with comments.
    """
    cm = CommentedMap()

    sorted_fields = sorted(cfg.model_fields.items())

    # Loop through the fields of the model
    for field_name, field in sorted_fields:
        value = getattr(cfg, field_name)
        description = field.description
        default_value = field.default
        deprecated = field.deprecated
        examples = field.examples
        default_factory = field.default_factory

        # Recursively handle nested models
        if isinstance(value, BaseModel):
            # If the value is a BaseModel but has Field attributes honor them
            set_comment(
                cm, field_name, description, default_value, deprecated, examples, indent
            )

            # Now recursively handle the nested model
            nested_cm = config_to_commented_map(value, indent + 2)
            cm[field_name] = nested_cm
        else:
            # If the default value comes from a factory, evaluate it and use the result as the default value
            if default_value is PydanticUndefined:
                if default_factory is not None and callable(default_factory):
                    default_value = default_factory()
            set_comment(
                cm, field_name, description, default_value, deprecated, examples, indent
            )
            cm[field_name] = value

    return cm


def set_comment(
    cm: CommentedMap,
    field_name: str,
    description: str | None,
    default_value: Any,
    deprecated: Deprecated | str | bool | None,
    examples: list[Any] | None,
    indent: int,
):
    """
    Set a comment for a field in a CommentedMap.

    This function adds a comment to a field in a CommentedMap based on the field's description
    and default value. The comment is added before the field key.

    Example of a YAML field with a comment:

    ```
    # This is the description of the field. It can be a longer text that wraps to
    #   the next line if needed.
    # Default: some_value
    # Deprecated: reason for deprecation
    # Examples:
    # - An example
    # - Another example
    field_name: value
    ```
    Args:
        cm (CommentedMap): The CommentedMap to modify.
        field_name (str): The name of the field to comment.
        description (str): The description of the field.
        default_value (any): The default value of the field.
        deprecated (bool): Whether the field is deprecated.
        indent (int): The indentation level for the comment.
    """
    comment_parts = []

    if description:
        comment_parts.append(
            # Initialize a TextWrapper.We use break_long_words=False to prevent breaking long words.
            # It ensures that words are kept intact, which is usually preferable in most text
            # wrapping scenarios. This is especially important for things like URLs, long variable
            # names, or other strings that shouldn't be split.
            textwrap.fill(description, width=80, break_long_words=False)
        )

    if default_value is not PydanticUndefined:
        if default_value == "":
            comment_parts.append("Default: ''")
        else:
            comment_parts.append(f"Default: {default_value}")

    if deprecated:
        if isinstance(deprecated, str):
            comment_parts.append(f"Deprecated: {deprecated}")
        else:
            comment_parts.append("Deprecated")

    # Join all parts with line breaks
    full_comment = "\n".join(comment_parts)

    # Add examples if present
    if examples:
        full_comment += "\nExamples:"
        for example in examples:
            full_comment += f"\n  - {example}"

    cm.yaml_set_comment_before_after_key(
        field_name,
        before=full_comment,
        indent=indent,
    )


def write_config_to_yaml(cfg: Config, file_path: str):
    """
    Write a Pydantic model to a YAML file with comments derived from field descriptions.

    Args:
        cfg (Config): The Pydantic model to write to YAML.
        file_path (str): The path to the YAML file to write.
    """
    commented_map = config_to_commented_map(cfg)
    with open(file_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(commented_map, yaml_file)


def get_api_base(host_port: str) -> str:
    """Returns server API URL, based on the provided host_port"""
    return f"http://{host_port}/v1"


def get_model_family(family, model_path):
    family = MODEL_FAMILY_MAPPINGS.get(family, family)
    if family:
        if family.lower() not in MODEL_FAMILIES:
            raise ConfigException(f"Unknown model family: {family}")

        return family.lower()

    # If family is not set try to guess the model family based on the model's filename
    guess = match(r"^\w*", path.basename(model_path)).group(0).lower()
    guess = MODEL_FAMILY_MAPPINGS.get(guess, guess)

    return guess if guess in MODEL_FAMILIES else DEFAULTS.MODEL_FAMILY


def ensure_storage_directories_exist() -> bool:
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
        DEFAULTS.PHASED_DIR,
    ]

    for dirpath in dirs_to_make:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

    fresh_install = recreate_train_profiles()

    # create expert_args file for users to see/edit
    if not os.path.isfile(DEFAULTS.TRAIN_ADDITIONAL_OPTIONS_FILE):
        with open(
            DEFAULTS.TRAIN_ADDITIONAL_OPTIONS_FILE, "w", encoding="utf-8"
        ) as outfile:
            yaml.dump(DEFAULTS.ADDITIONAL_ARGS_DEFAULTS, outfile)

    return fresh_install


# recreate_train_profiles creates all train profiles in the proper directory and takes an argument, overwrite, which will write to the files even if they already exist
def recreate_train_profiles(overwrite: bool = False) -> bool:
    TRAIN_DIR_EXPECTED_FILES = {
        "A100_H100_x8.yaml",
        "A100_H100_x4.yaml",
        "A100_H100_x2.yaml",
        "L40_x8.yaml",
        "L40_x4.yaml",
        "L4_x8.yaml",
    }

    fresh_install = False
    profile_dir = os.environ.get(DEFAULTS.ILAB_TRAIN_PROFILE_DIR)
    if profile_dir != "" and profile_dir is not None:
        # the train dir exists, read it
        # expect only the supported cfgs.
        for file in TRAIN_DIR_EXPECTED_FILES:
            new_file = os.path.join(DEFAULTS.TRAIN_PROFILE_DIR, file)
            tmpl_file = os.path.join(profile_dir, file)
            train_cfg = read_train_profile(tmpl_file)
            if not os.path.isfile(new_file):
                # If any of the train profiles are missing, treat this as a new system. We do not want to
                # prompt the user TWICE if they want to overwrite the train profiles.
                fresh_install = True
                with open(new_file, "w", encoding="utf-8") as outfile:
                    d = train_cfg.model_dump_json()
                    loaded = yaml.load(d)
                    yaml.dump(loaded, outfile)
    else:
        to_write = {
            DEFAULTS.TRAIN_A100_H100_X4_PROFILE: FOUR_GPU_TRAIN_AH,
            DEFAULTS.TRAIN_A100_H100_X8_PROFILE: EIGHT_GPU_TRAIN_AH,
            DEFAULTS.TRAIN_A100_H100_X2_PROFILE: TWO_GPU_TRAIN_AH,
            DEFAULTS.TRAIN_L40_X8_PROFILE: EIGHT_L_FORTY_GPU,
            DEFAULTS.TRAIN_L40_X4_PROFILE: FOUR_L_FORTY_GPU,
            DEFAULTS.TRAIN_L4_X8_PROFILE: EIGHT_L_FOUR_GPU,
        }

        for file, train_cfg in to_write.items():
            if not os.path.isfile(file) or overwrite:
                # If any of the train profiles are missing, treat this as a new system. We do not want to
                # prompt the user TWICE if they want to overwrite the train profiles.
                fresh_install = True
                with open(file, "w", encoding="utf-8") as outfile:
                    d = train_cfg.model_dump_json()
                    loaded = yaml.load(d)
                    yaml.dump(loaded, outfile)
    return fresh_install


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




def render_configs_and_profiles(gpus: int) ->  list[tuple[Config, list[dict[str, _train]]]]:
    single_gpu_confg = Config(
        generate=_generate(
            sdg_scale_factor=10,
            pipeline="full",
            teacher=_serve(
                model_path="~/.cache/instructlab/models/mistralai/Mistral-7B-Instruct-v0.2",
                vllm=_serve_vllm(
                    gpus=1,
                    llm_family='mixtral'       
                )
            )
        )
    )
    multi_gpu_config = Config(
        generate=_generate(
            sdg_scale_factor=10,
            pipeline="full",
            teacher=_serve(
                model_path="~/.cache/instructlab/models/mistralai/Mistral-7B-Instruct-v0.2",
                vllm=_serve_vllm(
                    gpus=gpus,
                    llm_family='mixtral'       
                )
            )
        )
    )
    single_server_gpu_config = Config(
            serve=_serve(
                vllm=_serve_vllm(
                    gpus=1,
                ),
                model_path="~/.cache/instructlab/models/instructlab/granite-7b-lab",
            ),
            generate=_generate(
            sdg_scale_factor=10,
            pipeline="full",
            teacher=_serve(
                model_path="~/.cache/instructlab/models/mistralai/Mistral-7B-Instruct-v0.2",
                vllm=_serve_vllm(
                    gpus=2,
                    llm_family='mixtral'       
                )
            )
        )
    )
    multi_server_gpu_config = Config(
        serve=_serve(
            vllm=_serve_vllm(
                gpus=gpus,
            ),
            model_path="~/.cache/instructlab/models/instructlab/granite-7b-lab",
        ),
        generate=_generate(
            teacher=_serve(
            pipeline="full",
                model_path="~/.cache/instructlab/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
                vllm=_serve_vllm(
                    gpus=2,
                    llm_family='mixtral'       
                )
            )
        )
    )
    macos_config = Config(
        serve=_serve(
            llama_cpp=_serve_llama_cpp(
           gpu_layers=15 
        ),
            model_path="~/.cache/instructlab/models/granite-7b-lab-Q4_K_M.gguf",
        ),
        generate=_generate(
            sdg_scale_factor=10,
            pipeline="full",
            teacher=_serve(
                model_path="~/.cache/instructlab/models/granite-7b-lab-Q4_K_M.gguf",
            )
        )
    )


    return [
        (single_gpu_confg, [{"NVIDIA RTX 3070/3080/4070/4080": SINGLE_CONSUMER_GPU_TRAIN}]),
        (multi_gpu_config, [{"NVIDIA RTX 3070/3080/4070/4080": MULTI_CONSUMER_GPU_TRAIN}]),
        (single_server_gpu_config, [{"NVIDIA A100/H100/L40/L4": SINGLE_SERVER_GPU_TRAIN}]),
        (multi_server_gpu_config, [{"NVIDIA 2x A100/H100": TWO_GPU_TRAIN_AH}, {"NVIDIA 4x A100/H100": FOUR_GPU_TRAIN_AH}, {"NVIDIA 8x A100 or H100": EIGHT_GPU_TRAIN_AH}, {"NVIDIA 4x L40": FOUR_L_FORTY_GPU}, {"NVIDIA 8x L40": EIGHT_L_FORTY_GPU}, {"NVIDIA 8x L4": EIGHT_L_FOUR_GPU}]),
        (macos_config, [MACOS_TRAIN]),
    ]


def init(
    ctx: click.Context, config_file: str | os.PathLike[str], debug_level: int = 0
) -> None:
    config_obj: Config
    error_msg: str | None = None
    if config_file == "DEFAULT":
        # if user passed --config DEFAULT we should ensure all proper dirs are created
        ensure_storage_directories_exist()
        config_obj = get_default_config()
    elif os.path.isfile(config_file):
        try:
            # only create dirs if the user passed --config <some_file_other_than_default>
            if config_file != DEFAULTS.CONFIG_FILE:
                ensure_storage_directories_exist()
            config_obj = read_config(config_file)
        except ConfigException as e:
            # delayed, so ilab config init can override a broken config.
            config_obj = get_default_config()
            error_msg = str(e)
    else:
        # if the user is running a cmd without --config then we should NOT create the persistent storage dirs for them, they must run `ilab config init`
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


def map_train_to_library(ctx, params):
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

    lora_args = LoraOptions(rank=0)
    lora_enabled = False
    lora_options = False
    if params["lora_rank"]:
        lora_enabled = True
        lora_args.rank = params["lora_rank"]
    if params["lora_alpha"] is not None:
        lora_options = True
        lora_args.alpha = params["lora_alpha"]
    if params["lora_dropout"] is not None:
        lora_options = True
        lora_args.dropout = params["lora_dropout"]
    if params["lora_target_modules"] is not None:
        lora_options = True
        lora_args.target_modules = params["lora_target_modules"]
    if params["lora_quantize_dtype"] is not None:
        lora_options = True
        lora_args.quantize_data_type = params["lora_quantize_dtype"]
    if lora_enabled and params["is_padding_free"]:
        ctx.fail(
            "Cannot combine LoRA with a padding free model. Set lora_rank to 0 or disable is_padding_free"
        )
    if lora_options and not lora_enabled:
        click.secho("LoRA is disabled (rank=0), ignoring all additional LoRA args")

    train_args.deepspeed_options = ds_args
    train_args.lora = lora_args

    return train_args, torch_args


def finish_additional_train_args(current_additional):
    additional_args_and_defaults = {}
    try:
        with open(
            DEFAULTS.TRAIN_ADDITIONAL_OPTIONS_FILE, "r", encoding="utf-8"
        ) as yamlfile:
            additional_args_and_defaults = yaml.load(yamlfile)
    except FileNotFoundError:
        additional_args_and_defaults = DEFAULTS.ADDITIONAL_ARGS_DEFAULTS
        # user has not run `ilab config init`, yet. This should only happen once
    _expand_paths(additional_args_and_defaults)
    for key, val in additional_args_and_defaults.items():
        if key not in current_additional:
            current_additional[key] = val

    return current_additional


def storage_dirs_exist() -> bool:
    dirs_to_check = [
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
        DEFAULTS.PHASED_DIR,
    ]
    return all(os.path.exists(dirpath) for dirpath in dirs_to_check)
