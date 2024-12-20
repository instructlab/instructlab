# SPDX-License-Identifier: Apache-2.0

# Standard
from os import path
from re import match
from typing import Any, Optional, Union
import enum
import logging
import os
import pathlib
import sys
import textwrap
import typing

# Third Party
# pylint: disable=ungrouped-imports
from instructlab.training import (
    DeepSpeedOptions,
    DistributedBackend,
    FSDPOptions,
    LoraOptions,
    TorchrunArgs,
    TrainingArgs,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticUndefined
from ruamel.yaml import YAML, CommentedMap
from typing_extensions import deprecated as Deprecated
import click

# First Party
from instructlab.utils import get_model_arch, use_legacy_pretraining_format

# Local
from . import log
from .defaults import (
    CONFIG_VERSION,
    DEFAULTS,
    LOG_FORMAT,
    MODEL_FAMILIES,
    MODEL_FAMILY_MAPPINGS,
)

# Initialize ruamel.yaml
yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)

# Enable fast-download using external dependency "hf_transfer"
# Used by the "ilab model download" command
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


logger = logging.getLogger(__name__)


class ConfigException(Exception):
    """An exception that a configuration file has an error."""


class _general(BaseModel):
    """Class describing various top-level configuration options for all commands."""

    # model configuration
    model_config = ConfigDict(extra="ignore")

    # additional fields with defaults
    log_level: StrictStr = Field(default="INFO", description="Log level for logging.")
    debug_level: int = Field(default=0, description="Debug level for logging.")
    log_format: StrictStr = Field(
        default=LOG_FORMAT,
        description="Log format. https://docs.python.org/3/library/logging.html#logrecord-attributes",
        validate_default=True,
    )
    use_legacy_tmpl: bool = Field(
        default=False,
        description="Use legacy IBM Granite chat template (default uses 3.0 Instruct template)",
    )

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

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, log_format):
        try:
            logging.PercentStyle(log_format).validate()
            return log_format
        except ValueError as e:
            raise ValueError(
                f"\nFailed to configure log format: {e}\n"
                "Have you specified a valid log format?\n"
                "Consider reading: https://docs.python.org/3/library/logging.html#logrecord-attributes"
            ) from e

    @model_validator(mode="after")
    def after_debug_level(self):
        # set debug level when log level is DEBUG
        if self.log_level == "DEBUG" and self.debug_level == 0:
            self.debug_level = 1
        return self


class _chat(BaseModel):
    """Class describing configuration of the 'chat' sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore")
    model: str = Field(
        default_factory=lambda: DEFAULTS.DEFAULT_CHAT_MODEL,
        description="Model to be used for chatting with.",
    )
    # additional fields with defaults
    vi_mode: bool = Field(default=False, description="Enable vim keybindings for chat.")
    visible_overflow: bool = Field(
        default=True,
        description="Renders vertical overflow if enabled, displays ellipses otherwise.",
    )
    context: str = Field(
        default="default",
        description="Predefined setting or environment that influences the behavior and responses of the chat assistant. Each context is associated with a specific prompt that guides the assistant on how to respond to user inputs. Available contexts: default, cli_helper.",
    )
    session: typing.Optional[str] = Field(
        default=None, description="Filepath of a dialog session file."
    )
    # use a lambda to avoid caching
    logs_dir: str = Field(
        default_factory=lambda: DEFAULTS.CHATLOGS_DIR,
        description="Directory where chat logs are stored.",
    )
    max_tokens: typing.Optional[int] = Field(
        default=None,
        description="The maximum number of tokens that can be generated in the chat completion. Be aware that larger values use more memory.",
    )
    temperature: float = Field(
        default=1.0,
        description="Controls the randomness of the model's responses. Lower values make the output more deterministic, while higher values produce more random results.",
    )


class _serve_vllm(BaseModel):
    """Class describing configuration of vLLM serving backend."""

    llm_family: str = Field(
        default="",  # TODO: convert to None and use a pattern to validate
        description="Large Language Model Family",
        examples=["granite", "mixtral"],
    )
    max_startup_attempts: int | None = Field(
        default=120,
        description="Maximum number of attempts to start the vLLM server.",
    )
    gpus: Optional[int] = Field(default=None, description="Number of GPUs to use.")
    # arguments to pass into vLLM process
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
        default=DEFAULTS.MAX_CONTEXT_SIZE,
        description="Maximum number of tokens that can be processed by the model.",
    )
    llm_family: str = Field(
        default="",  # TODO: convert to None and use a pattern to validate
        description="Large Language Model Family",
        examples=["granite", "mixtral"],
    )


class _serve_server(BaseModel):
    """Class describing configuration of server serving backend."""

    host: StrictStr = Field(default="127.0.0.1", description="Host to serve on.")
    port: StrictInt = Field(default=8000, description="Port to serve on.")
    current_max_ctx_size: PositiveInt = Field(
        default=DEFAULTS.MAX_CONTEXT_SIZE,
        description="Maximum number of tokens that can be processed by the currently served model.",
    )


class _serve(BaseModel):
    """Class describing configuration of the 'serve' sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore", protected_namespaces=())
    # vLLM configuration
    vllm: _serve_vllm = Field(
        default_factory=_serve_vllm,
        description="vLLM serving settings.",
    )
    # llama-cpp configuration
    llama_cpp: _serve_llama_cpp = Field(
        default_factory=_serve_llama_cpp,
        description="llama-cpp serving settings.",
    )
    model_path: StrictStr = Field(
        default_factory=lambda: DEFAULTS.DEFAULT_CHAT_MODEL,
        description="Directory where model to be served is stored.",
    )
    # additional fields with defaults
    server: _serve_server = Field(
        default=_serve_server(),
        description="Server configuration including host and port.",
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
        return get_api_base(self.server.host, self.server.port)


class _generate(BaseModel):
    """Class describing configuration of the 'generate' sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore")
    pipeline: Optional[str] = Field(
        default=DEFAULTS.SDG_PIPELINE,
        description="Data generation pipeline to use. Available: 'simple', 'full', or a valid path to a directory of pipeline workflow YAML files. Note that 'full' requires a larger teacher model, Mixtral-8x7b.",
    )
    max_num_tokens: Optional[int] = Field(
        default=DEFAULTS.SDG_MAX_NUM_TOKENS,
        description="The maximum amount of tokens for the model to generate during knowledge generation. A lower number yields less data but a faster SDG run. It is reccomended to use this on consumer hardware",
    )
    model: StrictStr = Field(
        default_factory=lambda: DEFAULTS.DEFAULT_TEACHER_MODEL,
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
    teacher: _serve = Field(
        default_factory=lambda: _serve(model_path=DEFAULTS.DEFAULT_TEACHER_MODEL),
        description="Teacher configuration",
    )
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
    # TODO: remove this? It's not used anywhere, was removed by 19b9f4794f79ef81578c00c901bac3ee9db8c046
    # related issue: https://github.com/instructlab/instructlab/issues/2261
    seed_file: StrictStr = Field(
        description="Path to seed file to be used for generation.",
        default_factory=lambda: DEFAULTS.SEED_FILE,
        deprecated=True,
    )


class _mmlu(BaseModel):
    """Class describing configuration of MMLU evaluation benchmark."""

    few_shots: int = Field(
        default=5,
        description="Number of question-answer pairs provided in the context preceding the question used for evaluation.",
    )
    batch_size: str | int = Field(
        default="auto",
        description="Batch size for evaluation. Valid values are a positive integer or 'auto' to select the largest batch size that will fit in memory.",
    )


class _mtbench(BaseModel):
    """Class describing configuration of MTBench evaluation benchmark."""

    judge_model: str = Field(
        default_factory=lambda: DEFAULTS.JUDGE_MODEL_MT,
        description="Judge model for mt_bench and mt_bench_branch.",
    )
    output_dir: str = Field(
        default_factory=lambda: DEFAULTS.EVAL_DATA_DIR,
        description="Directory where evaluation results are stored.",
    )
    max_workers: str | int = Field(
        default="auto",
        description="Number of workers to use for evaluation with mt_bench or mt_bench_branch. Must be a positive integer or 'auto'.",
    )


class _mtbenchbranch(BaseModel):
    """Class describing configuration of MTBenchBranch evaluation benchmark."""

    taxonomy_path: str = Field(
        default_factory=lambda: DEFAULTS.TAXONOMY_DIR,
        description="Path to where base taxonomy is stored.",
    )


class _mmlubranch(BaseModel):
    """Class describing configuration of MMLUBranch evaluation benchmark."""

    tasks_dir: str = Field(
        default_factory=lambda: DEFAULTS.DATASETS_DIR,
        description="Directory where custom MMLU tasks are stored.",
    )


class _evaluate(BaseModel):
    """Class describing configuration of the 'evaluate' sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore", protected_namespaces=())
    model: Optional[str] = Field(
        default=None,
        description="Model to be evaluated",
    )
    base_model: str = Field(
        default=DEFAULTS.MODEL_REPO,
        description="Base model to compare with 'model' for mt_bench_branch and mmlu_branch.",
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
        default_factory=_mmlu,
        description="MMLU benchmarking settings",
    )
    mmlu_branch: _mmlubranch = Field(
        default_factory=_mmlubranch,
        description="Settings to run MMLU against a branch of taxonomy containing custom skills/knowledge used for training.",
    )
    mt_bench: _mtbench = Field(
        default_factory=_mtbench,
        description="Multi-turn benchmarking settings for skills.",
    )
    mt_bench_branch: _mtbenchbranch = Field(
        default_factory=_mtbenchbranch,
        description="Settings to run MT-Bench against a branch of taxonomy containing custom skills/knowledge used for training",
    )


class _train(BaseModel):
    """Class describing configuration of the 'train' sub-command."""

    # model configuration
    model_config = ConfigDict(
        extra="ignore",
        protected_namespaces=(),
        use_enum_values=True,  # populate models with the value property of enums, rather than the raw enum.
    )
    pipeline: str = Field(
        default="full",
        description="Training pipeline to use. Simple is for systems with limited resources, full is for more capable consumer systems (64 GB of RAM), and accelerated is for systems with a dedicated GPU.",
        examples=["simple", "full", "accelerated"],
        pattern="simple|full|accelerated",
    )
    model_path: str = Field(
        default=DEFAULTS.MODEL_REPO,
        description="Directory where the model to be trained is stored.",
    )
    device: str = Field(
        default="cpu",
        description="PyTorch device to use. Use 'cpu' for 'simple' and 'full' training on Linux. Use 'mps' for 'full' training on MacOS Metal Performance Shader. Use 'cuda' for Nvidia CUDA / AMD ROCm GPUs. Use 'hpu' for Intel Gaudi GPUs.",
        examples=["cpu", "mps", "cuda", "hpu"],
        pattern="cpu|mps|cuda|hpu",
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
        default=5000,
        description="Maximum tokens per gpu for each batch that will be handled in a single step. If running into out-of-memory errors, this value can be lowered but not below the `max_seq_len`.",
    )
    num_epochs: int = Field(
        default=10, description="Number of epochs to run training for."
    )
    effective_batch_size: int = Field(
        default=64,
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
    fsdp_cpu_offload_optimizer: bool = Field(
        default=False, description="Allow CPU offload for FSDP optimizer."
    )
    distributed_backend: DistributedBackend = Field(
        default=DistributedBackend.FSDP,
        description="Pick a distributed training backend framework for GPU accelerated full fine-tuning.",
        validate_default=True,  # ensures that the 'use_enum_values' flag takes effect on the default value
    )
    lora_rank: int | None = Field(
        default=0,
        description="Rank of low rank matrices to be used during training.",
    )
    lora_quantize_dtype: str | None = Field(
        default="nf4",
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
    disable_flash_attn: Optional[bool] = Field(
        default=False,
        description="Whether or not we should disable the use of flash-attention during training. This is useful when using older GPUs.",
    )
    additional_args: dict[str, typing.Any] = Field(
        default_factory=dict,
        description="Additional arguments to pass to the training script. These arguments are passed as key-value pairs to the training script.",
    )
    # additional training configuration for
    # lab-multiphase and skills-only training as applicable.
    # TODO: could move into its own object.
    # Not strictly necessary for a correct training object.
    phased_phase1_num_epochs: int | None = Field(
        default=7,
        gt=0,
        description="Number of epochs to run training for during phase1 (experimentally optimal number is 7).",
    )
    # anything greater than 0 enables samples_per_save for the phase.
    phased_phase1_samples_per_save: int = Field(
        default=0,
        ge=0,
        description="Number of samples the model should see before saving a checkpoint during phase1. Disabled when set to 0.",
    )
    phased_phase1_learning_rate: float = Field(
        default=2e-5,
        ge=0,
        description="Learning rate for phase1 knowledge training.",
    )
    phased_phase1_effective_batch_size: int | None = Field(
        default=128,
        description="Phased phase1 effective batch size.",
    )
    phased_phase2_num_epochs: int | None = Field(
        default=10,
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
    phased_phase2_learning_rate: float = Field(
        default=6e-6,
        ge=0,
        description="Learning rate for phase2 skills training.",
    )
    phased_phase2_effective_batch_size: int | None = Field(
        default=3840, description="Phased phase2 effective batch size."
    )
    phased_mt_bench_judge: str | None = Field(
        default_factory=lambda: DEFAULTS.DEFAULT_JUDGE_MODEL,
        description="Judge model path for phased MT-Bench evaluation.",
    )
    phased_base_dir: str | None = Field(
        default_factory=lambda: DEFAULTS.PHASED_DIR,
        description="Base directory for organization of end-to-end intermediate outputs.",
    )
    training_journal: str | None = Field(
        default=None,
        description="Optional path to a yaml file that tracks the progress of multiphase training.",
    )


class _metadata(BaseModel):
    # model configuration
    model_config = ConfigDict(extra="ignore")
    cpu_info: str | None = Field(
        default=None,
        description="Manufacturer, Family, and SKU of the system CPU, ex: Apple M3 Max",
    )
    gpu_manufacturer: str | None = Field(
        default=None, description="Manufacturer of the system GPU, ex: Nvidia"
    )
    gpu_family: str | None = Field(
        default=None, description="Family of the system GPU, ex: H100"
    )
    gpu_count: int | None = Field(
        default=None, description="Amount of GPUs on the system, ex: 8"
    )
    gpu_sku: list[str] | None = Field(
        default=None,
        description="Specific SKU related information about the given GPU, ex: PCIe, NVL",
    )


class Config(BaseModel):
    """Configuration for the InstructLab CLI.
    Config options are defined by the respective subclasses and are loaded into a single 'Config' object here
    Instantation of this object should be done via 'get_default_config()'
    Note that values here can be overriden by a users 'config.yaml' or command line overrides in some cases
    """

    # chat configuration
    chat: _chat = Field(
        default_factory=_chat, description="Chat configuration section."
    )
    # generate configuration
    generate: _generate = Field(
        default_factory=_generate, description="Generate configuration section."
    )
    # serve configuration (includes both llama-cpp and vLLM configuration)
    serve: _serve = Field(
        default_factory=_serve, description="Serve configuration section."
    )
    # train configuration
    train: _train = Field(
        default_factory=_train, description="Train configuration section."
    )
    # evaluate configuration
    evaluate: _evaluate = Field(
        default_factory=_evaluate, description="Evaluate configuration section."
    )
    # additional fields with defaults
    general: _general = Field(
        default_factory=_general, description="General configuration section."
    )
    # model configuration
    model_config = ConfigDict(extra="ignore")
    # config file versioning
    version: str = Field(
        default=CONFIG_VERSION,
        description="Configuration file structure version.",
        frozen=True,  # don't allow this to be changed anywhere in the code
    )
    # metadata about the configuration, specifically GPU information
    metadata: _metadata = Field(
        default_factory=_metadata,
        description="Metadata pertaining to the specifics of the system which the Configuration is meant to be applied to.",
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
    model_path=os.path.join(DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"),
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
    model_path=os.path.join(DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"),
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
    model_path=os.path.join(DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"),
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
    model_path=os.path.join(DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"),
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
    model_path=os.path.join(DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"),
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
    model_path=os.path.join(DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"),
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
    lora_quantize_dtype="nf4",
    lora_rank=2,
    max_batch_len=60000,
    max_seq_len=4096,
    save_samples=1000,
    deepspeed_cpu_offload_optimizer=True,
    is_padding_free=True,
    num_epochs=8,
    model_path=os.path.join(DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"),
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
    lora_quantize_dtype="nf4",
    lora_rank=2,
    max_batch_len=30000,
    max_seq_len=4096,
    save_samples=1000,
    deepspeed_cpu_offload_optimizer=True,
    is_padding_free=True,
    num_epochs=5,
    model_path=os.path.join(DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"),
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
    lora_quantize_dtype="nf4",
    lora_rank=2,
    max_batch_len=30000,
    max_seq_len=4096,
    save_samples=1000,
    deepspeed_cpu_offload_optimizer=True,
    is_padding_free=True,
    num_epochs=5,
    model_path=os.path.join(DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"),
)


SINGLE_L4 = _train(
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
    lora_quantize_dtype="nf4",
    lora_rank=2,
    max_batch_len=30000,
    max_seq_len=4096,
    save_samples=1000,
    deepspeed_cpu_offload_optimizer=True,
    is_padding_free=True,
    num_epochs=5,
    model_path=os.path.join(DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"),
)

SINGLE_L40 = _train(
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
    lora_quantize_dtype="nf4",
    lora_rank=2,
    max_batch_len=30000,
    max_seq_len=4096,
    save_samples=1000,
    deepspeed_cpu_offload_optimizer=True,
    is_padding_free=True,
    num_epochs=5,
    model_path=os.path.join(DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"),
)

SINGLE_A100_H100 = _train(
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
    effective_batch_size=128,
    lora_quantize_dtype=None,
    lora_rank=0,
    max_batch_len=60000,
    max_seq_len=4096,
    save_samples=1000,
    deepspeed_cpu_offload_optimizer=True,
    is_padding_free=True,
    num_epochs=8,
    model_path=os.path.join(DEFAULTS._cache_home, "models/instructlab/granite-7b-lab"),
)


TRAIN_DIR_EXPECTED_FILES = {
    "A100_H100_x8.yaml",
    "A100_H100_x4.yaml",
    "A100_H100_x2.yaml",
    "L40_x8.yaml",
    "L40_x4.yaml",
    "L4_x8.yaml",
}


def get_default_config() -> Config:
    """Generates default configuration for CLI"""
    return Config()


def read_config(
    config_file: str | os.PathLike[str] | None = None,
) -> Config:
    """Reads configuration from disk."""
    config_file = DEFAULTS.CONFIG_FILE if config_file is None else config_file
    try:
        with open(config_file, "r", encoding="utf-8") as yamlfile:
            content = yaml.load(yamlfile)
            if not isinstance(content, dict):
                raise ConfigException(
                    f"Expected a dictionary but got {type(content).__name__} in file {config_file}."
                )
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
    except TypeError as exc:
        raise ConfigException(
            f"Failed to load config from {config_file}: {exc}"
        ) from exc


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
            # When the default value's type is an Enum or Enum value, convert it to a string
            elif isinstance(default_value, enum.Enum):
                default_value = default_value.value
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
            # Initialize a TextWrapper. We use break_long_words=False to prevent breaking long words.
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


def get_api_base(host: str, port: int) -> str:
    """Returns server API URL, based on the provided host and port"""
    return f"http://{host}:{port}/v1"


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
        DEFAULTS.TRAIN_ADDITIONAL_OPTIONS_DIR,
        DEFAULTS.PHASED_DIR,
        DEFAULTS.SYSTEM_PROFILE_DIR,
    ]

    for dirpath in dirs_to_make:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

    fresh_install = recreate_system_profiles()

    # open file in read mode to load current contents

    train_options_dict: dict[str, Any] = {}
    try:
        with open(
            DEFAULTS.TRAIN_ADDITIONAL_OPTIONS_FILE, "r", encoding="utf-8"
        ) as infile:
            train_options_dict = yaml.load(infile) or {}
    except FileNotFoundError:
        logger.debug("Additional Arguments internal file not found, creating now")

    # check if there are new keys in defaults
    if len(DEFAULTS.ADDITIONAL_ARGS_DEFAULTS.keys()) > len(train_options_dict.keys()):
        # open file in write mode and update contents if there are new keys
        with open(
            DEFAULTS.TRAIN_ADDITIONAL_OPTIONS_FILE, "w", encoding="utf-8"
        ) as outfile:
            yaml.dump(DEFAULTS.ADDITIONAL_ARGS_DEFAULTS, outfile)

    return fresh_install


# read_and_create_system_profiles walks the given dir, reads the templated files and writes them into
# the DEFAULTS.SYSTEM_PROFILE_DIR preserving their arch, processor name, and yaml file name
# pylint: disable=broad-exception-caught
def read_and_create_system_profiles(profiles_dir: str, overwrite: bool) -> bool:
    """
    read_and_create_system_profiles walks the given dir, reads the templated files and writes them into
    the DEFAULTS.SYSTEM_PROFILE_DIR preserving their arch, processor name, and yaml file name
    """
    # if we are overwriting, remove the existing dir. We should do this because
    # the directory being provided, if using $ILAB_SYSTEM_PROFILE_DIR, could have a different structure.
    # if the directory structure differs, some leftover profiles could be left behind for selection
    if overwrite:
        # Standard
        import shutil

        for item in os.listdir(DEFAULTS.SYSTEM_PROFILE_DIR):
            item_path = os.path.join(DEFAULTS.SYSTEM_PROFILE_DIR, item)
            # Check if the item is a directory
            if os.path.isdir(item_path):
                # Remove the directory and its contents
                shutil.rmtree(item_path)
    fresh_install = False
    for dirpath, _dirnames, filenames in os.walk(profiles_dir):
        for filename in filenames:
            # open file on disk
            file_path = os.path.join(dirpath, filename)
            try:
                # read system profile from disk
                with open(file_path, "r", encoding="utf-8") as file:
                    content = yaml.load(file)
                # ensure that the profile is a valid cfg before
                try:
                    _ = Config(**content)
                except Exception as exc:
                    raise ValueError(
                        f"System Profile: {file_path} is not a valid config {exc}"
                    ) from exc
                # get just the arch/name/profile_name
                relative_path = os.path.relpath(file_path, profiles_dir)
                # new file is the DEFAULTS.SYSTEM_PROFILE_DIR/arch/name/profile_name.yaml
                new_file = os.path.join(DEFAULTS.SYSTEM_PROFILE_DIR, relative_path)
                fresh_install = not os.path.exists(new_file)
                directory = os.path.dirname(new_file)

                # Create directories if they don't exist
                if directory:
                    os.makedirs(directory, exist_ok=True)
                if not os.path.isfile(new_file) or overwrite:
                    with open(new_file, "w", encoding="utf-8") as outfile:
                        yaml.dump(content, outfile)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    return fresh_install


# recreate_system_profiles writes all profile directories found in src/instructlab to disk
# the location is ~/.local/share/instructlab/internal/system_profiles
def recreate_system_profiles(overwrite: bool = False) -> bool:
    """
    recreate_system_profiles writes all profile directories found in src/instructlab to disk
    the location is ~/.local/share/instructlab/internal/system_profiles
    """
    profile_dir = os.environ.get(DEFAULTS.ILAB_SYSTEM_PROFILE_DIR)
    if profile_dir != "" and profile_dir is not None:
        return read_and_create_system_profiles(profile_dir, overwrite)
    # else, we aren't reading from disk, so we need to find where we are
    # Get the directory where we are
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Compute the path to the 'profiles' directory relative to the current file
    profile_dir = os.path.join(current_dir, "profiles")

    return read_and_create_system_profiles(profile_dir, overwrite)


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
            fmt=config_obj.general.log_format,
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

    fsdp_args = FSDPOptions(
        cpu_offload_params=params["fsdp_cpu_offload_optimizer"],
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
    train_args.fsdp_options = fsdp_args
    train_args.distributed_backend = DistributedBackend(params["distributed_backend"])
    train_args.lora = lora_args
    if params["pipeline"] == "full":
        train_args.disable_flash_attn = True

    student_model_arch = get_model_arch(pathlib.Path(params["model_path"]))
    if ctx.obj.config.general.use_legacy_tmpl:
        train_args.use_legacy_tmpl = True
    else:
        train_args.use_legacy_tmpl = use_legacy_pretraining_format(
            params["model_path"],
            student_model_arch,
        )
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
        DEFAULTS.TRAIN_ADDITIONAL_OPTIONS_DIR,
        DEFAULTS.PHASED_DIR,
        DEFAULTS.SYSTEM_PROFILE_DIR,
    ]
    return all(os.path.exists(dirpath) for dirpath in dirs_to_check)
