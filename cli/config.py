# Standard
from dataclasses import asdict, dataclass

# Third Party
import yaml

DEFAULT_CONFIG = "config.yaml"
DEFAULT_CHAT_LOGS = "data/chatlogs"
DEFAULT_MODEL = "merlinite-7b-Q4_K_M"
DEFAULT_MODEL_PATH = f"models/{DEFAULT_MODEL}.gguf"
DEFAULT_HOST_PORT = "localhost:8000"
DEFAULT_API_KEY = "no_api_key"
DEFAULT_VI_MODE = False
DEFAULT_VISIBLE_OVERFLOW = True
DEFAULT_TAXONOMY_REPO = "git@github.com:instruct-lab/taxonomy.git"
DEFAULT_TAXONOMY_PATH = "taxonomy"
DEFAULT_TAXONOMY_BRANCH = "main"
DEFAULT_PROMPT_FILE = "prompt.txt"
DEFAULT_SEED_FILE = "seed_tasks.json"
DEFAULT_GENERATED_FILES_OUTPUT_DIR = "generated"


class ConfigException(Exception):
    """An exception that a configuration file doesn't exists or it doesn't contain valid YAML."""

    def __init__(self, filename):
        super().__init__(
            f"Configuration file {filename} does not exist or contains invalid YAML."
        )


@dataclass
class _general:
    log_level: str


# pylint: disable=too-many-instance-attributes
@dataclass
class _chat:
    model: str
    vi_mode: bool
    visible_overflow: bool
    context: str
    session: str
    logs_dir: str


@dataclass
class _generate:
    model: str
    num_cpus: int
    num_instructions: int
    taxonomy_path: str
    output_dir: str
    prompt_file: str
    seed_file: str


@dataclass
class _serve:
    host_port: str
    model_path: str
    gpu_layers: int

    def api_base(self):
        """Returns server API URL, based on the configured host and port"""
        return get_api_base(self.host_port)


@dataclass
class Config:
    general: _general
    chat: _chat
    generate: _generate
    serve: _serve

    def __post_init__(self):
        if not isinstance(self.general, dict):
            return

        # pylint: disable=not-a-mapping
        self.general = _general(**self.general)
        self.chat = _chat(**self.chat)
        self.generate = _generate(**self.generate)
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
    chat = _chat(
        model=DEFAULT_MODEL,
        vi_mode=DEFAULT_VI_MODE,
        visible_overflow=DEFAULT_VISIBLE_OVERFLOW,
        context="default",
        session=None,
        logs_dir=DEFAULT_CHAT_LOGS,
    )
    generate = _generate(
        model=DEFAULT_MODEL,
        num_cpus=10,
        num_instructions=100,
        taxonomy_path=DEFAULT_TAXONOMY_PATH,
        output_dir=DEFAULT_GENERATED_FILES_OUTPUT_DIR,
        prompt_file=DEFAULT_PROMPT_FILE,
        seed_file=DEFAULT_SEED_FILE,
    )
    # pylint: disable=redefined-builtin
    serve = _serve(
        host_port=DEFAULT_HOST_PORT,
        model_path=DEFAULT_MODEL_PATH,
        gpu_layers=-1,
    )
    return Config(general=general, chat=chat, generate=generate, serve=serve)


def get_api_base(host_port):
    """Returns server API URL, based on the provided host_port"""
    return f"http://{host_port}/v1"
