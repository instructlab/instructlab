# Standard
from dataclasses import asdict, dataclass
import os
import textwrap

# Third Party
import yaml

DEFAULT_CONFIG = "config.yaml"
DEFAULT_MODEL_PATH = "models/ggml-malachite-7b-0226-Q4_K_M.gguf"
DEFAULT_MODEL = "ggml-malachite-7b-0226-Q4_K_M"
DEFAULT_TAXONOMY_REPO = "git@github.com:open-labrador/taxonomy.git"
DEFAULT_TAXONOMY_PATH = "taxonomy"
DEFAULT_PROMPT_FILE = "prompt.txt"
DEFAULT_SEED_FILE = "seed_tasks.json"


class ConfigException(Exception):
    """An exception that a configuration file doesn't exists or it doesn't contain valid YAML."""

    def __init__(self, filename):
        super().__init__(
            f"Configuration file {filename} does not exist or contains invalid YAML."
        )


@dataclass
class _general:
    log_level: str


@dataclass
class _chat:
    model: str
    context: str
    session: str


@dataclass
class _generate:
    model: str
    num_cpus: int
    num_instructions: int
    taxonomy_path: str
    prompt_file: str
    seed_file: str


@dataclass
class _list:
    taxonomy_path: str


@dataclass
class _serve:
    model_path: str
    gpu_layers: int


@dataclass
class Config:
    general: _general
    chat: _chat
    generate: _generate
    list: _list
    serve: _serve

    def __post_init__(self):
        if not isinstance(self.general, dict):
            return

        # pylint: disable=not-a-mapping
        self.general = _general(**self.general)
        self.chat = _chat(**self.chat)
        self.generate = _generate(**self.generate)
        self.list = _list(**self.list)
        self.serve = _serve(**self.serve)


def read_config(config_file=DEFAULT_CONFIG):
    """Reads configuration from disk"""
    try:
        with open(config_file, "r", encoding="utf-8") as yamlfile:
            cfg = yaml.safe_load(yamlfile)
            return Config(**cfg)
    except Exception as exc:
        raise ConfigException(config_file) from exc


def get_dict(cfg):
    """Returns configuration as a dictionary"""
    return asdict(cfg)


def write_config(cfg, config_file=DEFAULT_CONFIG):
    """Writes configuration to a disk"""
    with open(config_file, "w", encoding="utf-8") as yamlfile:
        yaml.safe_dump(get_dict(cfg), stream=yamlfile)


def get_default_config():
    """Generates default configuration for CLI"""
    general = _general(log_level="INFO")
    chat = _chat(model=DEFAULT_MODEL, context="default", session=None)
    generate = _generate(
        model=DEFAULT_MODEL,
        num_cpus=10,
        num_instructions=100,
        taxonomy_path=DEFAULT_TAXONOMY_PATH,
        prompt_file=DEFAULT_PROMPT_FILE,
        seed_file=DEFAULT_SEED_FILE,
    )
    # pylint: disable=redefined-builtin
    list = _list(taxonomy_path=DEFAULT_TAXONOMY_PATH)
    serve = _serve(model_path=DEFAULT_MODEL_PATH, gpu_layers=-1)
    return Config(general=general, chat=chat, generate=generate, list=list, serve=serve)


def create_config_file():
    chat_config_toml_txt = textwrap.dedent(
        """
    api_base = "http://localhost:8000/v1"
    api_key = "no_api_key"
    model = "malachite-7b"
    vi_mode = false
    visible_overflow = true

    [contexts]
    default = "You are Labrador, an AI language model developed by IBM DMF (Data Model Factory) Alignment Team. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
    cli_helper = "You are an expert for command line interface and know all common commands. Answer the command to execute as it without any explanation."
    dictionary = "You are a professional English-Chinese translator. Translate the input to the other language by providing its part of speech (POS) followed by up-to 5 common but distinct translations in this format: `[{POS}] {translation 1}; {translation 2}; ...`. Do not provide nonexistent results."
    """
    )
    chat_config_file_name = "chat-cli.toml"
    if not os.path.isfile(chat_config_file_name):
        if os.path.dirname(chat_config_file_name) != "":
            os.makedirs(os.path.dirname(chat_config_file_name), exist_ok=True)
        with open(chat_config_file_name, "w", encoding="utf-8") as model_file:
            model_file.write(chat_config_toml_txt)
