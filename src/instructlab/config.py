# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from os import path
from re import match
from typing import Optional

# Third Party
from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveInt,
    StrictStr,
    ValidationError,
    field_validator,
)
import httpx
import yaml
from ilab_train.config import TorchrunTrainArgs, FullTrainArgs

DEFAULT_API_KEY = "no_api_key"
DEFAULT_CONFIG = "config.yaml"
DEFAULT_MODEL = "merlinite-7b-lab-Q4_K_M"
DEFAULT_MODEL_PATH = f"models/{DEFAULT_MODEL}.gguf"
DEFAULT_TAXONOMY_REPO = "https://github.com/instructlab/taxonomy.git"
DEFAULT_TAXONOMY_PATH = "taxonomy"
DEFAULT_TAXONOMY_BASE = "origin/main"
DEFAULT_YAML_RULES = "yaml_rules.yaml"
MAX_CONTEXT_SIZE = 4096
# TODO: these constants should be removed, they should not leak out
DEFAULT_NUM_CPUS = 10
DEFAULT_CHUNK_WORD_COUNT = 1000
DEFAULT_NUM_INSTRUCTIONS = 100
DEFAULT_PROMPT_FILE = "prompt.txt"
DEFAULT_GENERATED_FILES_OUTPUT_DIR = "generated"
DEFAULT_CONNECTION_TIMEOUT = httpx.Timeout(timeout=30.0)
# use spawn start method, fork is not thread-safe
DEFAULT_MULTIPROCESSING_START_METHOD = "spawn"


class ConfigException(Exception):
    """An exception that a configuration file has an error."""


class _general(BaseModel):
    """Class describing various top-level configuration options for all commands."""

    # model configuration
    model_config = ConfigDict(extra="ignore")

    # additional fields with defaults
    log_level: StrictStr = "INFO"

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


class _chat(BaseModel):
    """Class describing configuration of the chat sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore")

    # required fields
    model: StrictStr

    # additional fields with defaults
    vi_mode: bool = False
    visible_overflow: bool = True
    context: str = "default"
    session: Optional[str] = None
    logs_dir: str = "data/chatlogs"
    greedy_mode: bool = False
    max_tokens: Optional[int] = None


class _generate(BaseModel):
    """Class describing configuration of the generate sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore")

    # required fields
    model: StrictStr
    taxonomy_path: StrictStr
    taxonomy_base: StrictStr

    # additional fields with defaults
    num_cpus: PositiveInt = DEFAULT_NUM_CPUS
    chunk_word_count: PositiveInt = DEFAULT_CHUNK_WORD_COUNT
    num_instructions: PositiveInt = DEFAULT_NUM_INSTRUCTIONS
    output_dir: StrictStr = DEFAULT_GENERATED_FILES_OUTPUT_DIR
    prompt_file: StrictStr = DEFAULT_PROMPT_FILE
    seed_file: StrictStr = "seed_tasks.json"


class _serve(BaseModel):
    """Class describing configuration of the serve sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    # required fields
    model_path: StrictStr

    # additional fields with defaults
    host_port: StrictStr = "127.0.0.1:8000"
    gpu_layers: int = -1
    max_ctx_size: PositiveInt = 4096

    def api_base(self):
        """Returns server API URL, based on the configured host and port"""
        return get_api_base(self.host_port)


class Config(BaseModel):
    """Configuration for the InstructLab CLI."""

    # required fields
    chat: _chat
    generate: _generate
    serve: _serve

    # additional fields with defaults
    general: _general = _general()

    # model configuration
    model_config = ConfigDict(extra="ignore")

@dataclass
class TrainingConfig:
    """
    Contains configuration options that are populated from the commandline
    and passed to the training code.

    Right now this is just a 1:1 map between the training args and the CLI args,
    but we can and should change this in the future to be more human.

    XXX(Aldo): This should use OmegaConf instead of the dataclass because we get lot of this stuff for free
    """
    nproc_per_node: int
    nnodes: int
    node_rank: int
    rdzv_id: int
    rdzv_endpoint: str
    data_path: str
    input_dir: str
    model_name_or_path: str
    output_dir: str
    num_epochs: int
    effective_batch_size: int
    learning_rate: float
    num_warmup_steps: int
    save_samples: int
    log_level: str
    seed: int
    mock_data: bool
    mock_len: int
    is_granite: bool
    max_batch_len: int
    # I don't believe this is actually used anywhere anymore,
    # but we should still keep it to avoid changing too much at once
    samples_per_gpu: int

    def __str__(self):
        """
        Allows pretty-printing this config
        """
        return yaml.dump(vars(self), sort_keys=False)

    def get_torchrun_config(self) -> TorchrunTrainArgs:
        """
        XXX(osilkin): This is a temporary solution to pass all the training args
                        In the future the configs will differ so there'll be more 
                        transformation happening here
        """
        return TorchrunTrainArgs(
            nnodes=self.nnodes,
            node_rank=self.node_rank,
            nproc_per_node=self.nproc_per_node,
            rdzv_endpoint=self.rdzv_endpoint,
            rdzv_id=self.rdzv_id,
        )

    def get_full_train_args(self) -> FullTrainArgs:
        """
        XXX(osilkin): This is a temporary solution to pass all the training args
                        In the future the configs will differ so there'll be more 
                        transformation happening here
        """
        return FullTrainArgs(
            data_path=self.data_path,
            effective_batch_size=self.effective_batch_size,
            input_dir=self.input_dir,
            learning_rate=self.learning_rate,
            is_granite=self.is_granite,
            max_batch_len=self.max_batch_len,
            log_level=self.log_level,
            mock_data=self.mock_data,
            mock_len=self.mock_len,
            model_name_or_path=self.model_name_or_path,
            num_epochs=self.num_epochs,
            num_warmup_steps=self.num_warmup_steps,
            output_dir=self.output_dir,
            samples_per_gpu=self.samples_per_gpu,
            save_samples=self.save_samples,
            seed=self.seed,
        )

def read_training_config(fp: str) -> TrainingConfig:
    """
    Returns a training config by reading from the given path.
    Throws a FileNotFoundError if the file doesn't exist.
    """
    if not os.path.exists(fp):
        raise FileNotFoundError(f'config file \'{fp}\' does not exist')
    
    with open(fp, 'r', encoding='utf-8') as infile:
        config_dict = yaml.safe_load(infile)
    
    config = TrainingConfig(**config_dict)
    return config

def get_default_config():
    """Generates default configuration for CLI"""
    return Config(
        chat=_chat(model=DEFAULT_MODEL),
        generate=_generate(
            model=DEFAULT_MODEL,
            taxonomy_path=DEFAULT_TAXONOMY_PATH,
            taxonomy_base=DEFAULT_TAXONOMY_BASE,
        ),
        serve=_serve(model_path=DEFAULT_MODEL_PATH),
    )


def read_config(config_file=DEFAULT_CONFIG):
    """Reads configuration from disk."""
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
                + "->".join(err.get("loc", ""))
                + ": "
                + err.get("msg", "").lower()
                + "\n"
            )
        raise ConfigException(msg) from exc


def get_dict(cfg):
    """Returns configuration as a dictionary"""
    return cfg.model_dump()


def write_config(cfg, config_file=DEFAULT_CONFIG):
    """Writes configuration to a disk"""
    with open(config_file, "w", encoding="utf-8") as yamlfile:
        yaml.safe_dump(get_dict(cfg), stream=yamlfile)


def get_api_base(host_port):
    """Returns server API URL, based on the provided host_port"""
    return f"http://{host_port}/v1"


def get_model_family(forced, model_path):
    return forced if forced else match(r"^\w*", path.basename(model_path)).group(0)
