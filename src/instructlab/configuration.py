# SPDX-License-Identifier: Apache-2.0

# Standard
from os import path
from re import match
from typing import Optional
import os
import sys

# Third Party
from instructlab.training import TorchrunArgs, TrainingArgs
from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveInt,
    StrictStr,
    ValidationError,
    field_validator,
)
import click
import httpx
import yaml

# Local
from . import log

DEFAULT_API_KEY = "no_api_key"
DEFAULT_CONFIG = "config.yaml"
# TODO: Consolidate --model and --model-path into one --model-path flag since we always need a path now
DEFAULT_MODEL_OLD = "merlinite-7b-lab-Q4_K_M"
DEFAULT_MODEL = "models/merlinite-7b-lab-Q4_K_M.gguf"
DEFAULT_MODEL_PATH = "models/merlinite-7b-lab-Q4_K_M.gguf"
DEFAULT_MODEL_REPO = "instructlab/granite-7b-lab"
DEFAULT_TAXONOMY_REPO = "https://github.com/instructlab/taxonomy.git"
DEFAULT_TAXONOMY_PATH = "taxonomy"
DEFAULT_TAXONOMY_BASE = "origin/main"
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

# When otherwise unknown, ilab uses this as the default family
DEFAULT_MODEL_FAMILY = "merlinite"

# Model families understood by ilab
MODEL_FAMILIES = set(("merlinite", "mixtral"))

# Map model names to their family
MODEL_FAMILY_MAPPINGS = {
    "granite": "merlinite",
}

DEFAULT_CKPT_DIR = "checkpoints"
DEFAULT_OUT_DIR = "train-output"


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


class _train(BaseModel):
    train_args: TrainingArgs
    torch_args: TorchrunArgs


class Config(BaseModel):
    """Configuration for the InstructLab CLI."""

    # required fields
    chat: _chat
    generate: _generate
    serve: _serve

    # additional fields with defaults
    general: _general = _general()

    # train configuration
    train: Optional[_train] = None

    # model configuration
    model_config = ConfigDict(extra="ignore")


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
        train=_train(
            train_args=TrainingArgs(
                model_path=DEFAULT_MODEL_REPO,
                data_path=DEFAULT_GENERATED_FILES_OUTPUT_DIR,
                ckpt_output_dir=DEFAULT_CKPT_DIR,
                data_output_dir=DEFAULT_OUT_DIR,
                max_seq_len=4096,
                max_batch_len=10000,
                num_epochs=10,
                effective_batch_size=3840,
                save_samples=25000,
                learning_rate=2e-6,
                warmup_steps=800,
                is_padding_free=False,
                random_seed=42,
            ),
            torch_args=TorchrunArgs(
                node_rank=0,
                nnodes=1,
                nproc_per_node=1,
                rdzv_id=123,
                rdzv_endpoint="127.0.0.1:12222",
            ),
        ),
    )


def read_train_profile(train_file):
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
                + "->".join(err.get("loc", ""))
                + ": "
                + err.get("msg", "").lower()
                + "\n"
            )
        raise ConfigException(msg) from exc


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
    forced = MODEL_FAMILY_MAPPINGS.get(forced, forced)
    if forced and forced.lower() not in MODEL_FAMILIES:
        raise ConfigException("Unknown model family: %s" % forced)

    # Try to guess the model family based on the model's filename
    guess = match(r"^\w*", path.basename(model_path)).group(0).lower()
    guess = MODEL_FAMILY_MAPPINGS.get(guess, guess)

    return guess if guess in MODEL_FAMILIES else DEFAULT_MODEL_FAMILY


class Lab:
    """Lab object holds high-level information about ilab CLI"""

    def __init__(self, config_obj: Config):
        self.config = config_obj


def init(ctx, config_file):
    if (
        ctx.invoked_subcommand not in {"config", "init", "sysinfo"}
        and "--help" not in sys.argv[1:]
    ):
        if config_file == "DEFAULT":
            config_obj = get_default_config()
        elif not os.path.isfile(config_file):
            config_obj = None
            ctx.fail(
                f"`{config_file}` does not exists, please run `ilab init` "
                "or point to a valid configuration file using `--config=<path>`."
            )
        else:
            try:
                config_obj = read_config(config_file)
            except ConfigException as ex:
                raise click.ClickException(str(ex))
        # setup logging
        log.configure_logging(log_level=config_obj.general.log_level.upper())
        ctx.obj = Lab(config_obj)
        # default_map holds a dictionary with default values for each command parameters
        training_dict = get_dict(ctx.obj.config)
        # since torch and train args are sep, they need to be combined into a single `train` entity for the default map
        # this is because the flags for `ilab model train` will only be populated if the default map has a single `train` entry, not two.
        training_dict["train"] = (
            training_dict["train"]["train_args"] | training_dict["train"]["torch_args"]
        )
        ctx.default_map = training_dict
