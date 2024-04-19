# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Optional

# Third Party
from pydantic import BaseModel, ConfigDict, PositiveInt, StrictStr, ValidationError
import httpx
import yaml

DEFAULT_API_KEY = "no_api_key"
DEFAULT_CONFIG = "config.yaml"
DEFAULT_MODEL = "merlinite-7b-lab-Q4_K_M"
DEFAULT_MODEL_PATH = f"models/{DEFAULT_MODEL}.gguf"
DEFAULT_TAXONOMY_REPO = "git@github.com:instructlab/taxonomy.git"
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


class ConfigException(Exception):
    """An exception that a configuration file has an error."""


class _general(BaseModel):
    """Class describing various top-level configuration options for all commands."""

    # model configuration
    model_config = ConfigDict(extra="ignore")

    # optional fields
    log_level: Optional[StrictStr] = "INFO"


class _chat(BaseModel):
    """Class describing configuration of the chat sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore")

    # required fields
    model: StrictStr

    # optional fields
    vi_mode: Optional[bool] = False
    visible_overflow: Optional[bool] = True
    context: Optional[str] = "default"
    session: Optional[str] = None
    logs_dir: Optional[str] = "data/chatlogs"
    greedy_mode: Optional[bool] = False


class _generate(BaseModel):
    """Class describing configuration of the generate sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore")

    # required fields
    model: StrictStr
    taxonomy_path: StrictStr
    taxonomy_base: StrictStr

    # optional fields
    num_cpus: Optional[PositiveInt] = DEFAULT_NUM_CPUS
    chunk_word_count: Optional[PositiveInt] = DEFAULT_CHUNK_WORD_COUNT
    num_instructions: Optional[PositiveInt] = DEFAULT_NUM_INSTRUCTIONS
    output_dir: Optional[StrictStr] = DEFAULT_GENERATED_FILES_OUTPUT_DIR
    prompt_file: Optional[StrictStr] = DEFAULT_PROMPT_FILE
    seed_file: Optional[StrictStr] = "seed_tasks.json"


class _serve(BaseModel):
    """Class describing configuration of the serve sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    # required fields
    model_path: StrictStr

    # optional fields
    host_port: Optional[StrictStr] = "127.0.0.1:8000"
    gpu_layers: Optional[int] = -1
    max_ctx_size: Optional[PositiveInt] = 4096

    def api_base(self):
        """Returns server API URL, based on the configured host and port"""
        return get_api_base(self.host_port)


class Config(BaseModel):
    """Configuration for the InstructLab CLI."""

    # required fields
    chat: _chat
    generate: _generate
    serve: _serve

    # optional fields
    general: Optional[_general] = _general()

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
