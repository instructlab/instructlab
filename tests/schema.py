# Standard
import enum

# Third Party
from pydantic import BaseModel


class LogLevel(str, enum.Enum):
    info = "INFO"
    debug = "DEBUG"
    warning = "WARNING"
    error = "ERROR"
    critical = "CRITICAL"


class Chat(BaseModel):
    api_base: str
    api_key: str
    context: str
    session: None
    model: str
    vi_mode: bool
    visible_overflow: bool


class General(BaseModel):
    log_level: LogLevel


class Generate(BaseModel):
    model: str
    num_cpus: int
    num_instructions: int
    prompt_file: str
    seed_file: str
    taxonomy_path: str


class Serve(BaseModel):
    gpu_layers: int
    model_path: str


class Config(BaseModel):
    """Root of Pydantic schema"""

    chat: Chat
    general: General
    generate: Generate
    serve: Serve
