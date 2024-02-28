# Copyright The Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from typing import Any, Optional
import logging
import os
import textwrap

# Third Party
import yaml

_DEF_CFG_FILE = "./cli/config/config.yml"
_DEF_MODEL_NAME = "ggml-malachite-7b-Q4_K_M"
_DEF_MODEL_PATH = "./models/ggml-malachite-7b-Q4_K_M.gguf"
_DEF_TAXONOMY_PATH = "./taxonomy"
_DEF_LOG_LEVEL = "info"
_DEF_NUM_GPUS = 10
_DEF_NUM_GPU_LAYERS = -1
_DEF_SESSION = ""
_DEF_CONTEXT = ""
_DEF_NUM_INSTRS = 100
_DEF_SEED_TASK_PATH = "./cli/generator/seed_tasks.jsonl"
_DEF_PROMPT_FILE_PATH = "./cli/generator/prompt.txt"
_CHAT_CATEGORY = "chat"
_GENERATE_CATEGORY = "generate"
_LIST_CATEGORY = "list"
_LOG_CATEGORY = "log"
_SERVE_CATEGORY = "serve"
_CTX_SETTING = "context"
_MODEL_SETTING = "model"
_MODEL_PATH_SETTING = "model_path"
_SESSION_SETTING = "session"
_PATH_TO_TAX_SETTING = "path_to_taxonomy"
_NUM_CPUS_SETTING = "num_cpus"
_LEVEL_SETTING = "level"
_NUM_GPUS_LAY_SETTING = "num_gpu_layers"
_NUM_INSTS_SETTING = "num_instructions_to_generate"
_SEED_TASK_PATH_SETTING = "seed_tasks_path"
_PROMPT_FILE_PATH = "prompt_file_path"


class Config(object):
    """Configuration for CLI"""

    def __init__(self, config_yml_path: Optional[str] = None):
        if config_yml_path:
            cfg_file = config_yml_path
        else:
            cfg_file = _DEF_CFG_FILE

        if not os.path.exists(cfg_file):
            raise ValueError("Config file doesn't exist: ", cfg_file)

        with open(cfg_file, "r") as yamlfile:
            self.cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

    def _get_setting(self, category: str, setting: str, default_value: Any) -> Any:
        if not self.cfg.get(category):
            value = default_value
        else:
            value = self.cfg.get(category).get(setting, default_value)
        return value

    def get_chat_context(self) -> str:
        return self._get_setting(_CHAT_CATEGORY, _CTX_SETTING, _DEF_CONTEXT)

    def get_chat_model(self) -> str:
        return self._get_setting(_CHAT_CATEGORY, _MODEL_SETTING, _DEF_MODEL_NAME)

    def get_chat_session(self) -> str:
        return self._get_setting(_CHAT_CATEGORY, _SESSION_SETTING, _DEF_SESSION)

    def get_generate_model(self) -> str:
        return self._get_setting(_GENERATE_CATEGORY, _MODEL_SETTING, _DEF_MODEL_NAME)

    def get_generate_num_cpus(self) -> int:
        return self._get_setting(_GENERATE_CATEGORY, _NUM_CPUS_SETTING, _DEF_NUM_GPUS)

    def get_generate_taxonomy(self) -> str:
        return self._get_setting(
            _GENERATE_CATEGORY, _PATH_TO_TAX_SETTING, _DEF_TAXONOMY_PATH
        )

    def get_generate_num_instructions(self) -> int:
        return self._get_setting(
            _GENERATE_CATEGORY, _NUM_INSTS_SETTING, _DEF_NUM_INSTRS
        )

    def get_generate_seed_task_path(self) -> str:
        return self._get_setting(
            _GENERATE_CATEGORY, _SEED_TASK_PATH_SETTING, _DEF_SEED_TASK_PATH
        )

    def get_generate_prompt_file_path(self) -> str:
        return self._get_setting(
            _GENERATE_CATEGORY, _PROMPT_FILE_PATH, _DEF_PROMPT_FILE_PATH
        )

    def get_list_taxonomy(self) -> str:
        return self._get_setting(
            _LIST_CATEGORY, _PATH_TO_TAX_SETTING, _DEF_TAXONOMY_PATH
        )

    def get_log_level(self) -> int:
        level = self._get_setting(_LOG_CATEGORY, _LEVEL_SETTING, _DEF_LOG_LEVEL)

        if level == "info":
            return logging.INFO
        elif level == "debug":
            return logging.DEBUG
        elif level == "warn":
            return logging.WARNING
        elif level == "error":
            return logging.ERROR
        elif level == "fatal":
            return logging.FATAL
        elif level == "critical":
            return logging.CRITICAL
        else:
            raise ValueError("Config: Unknown logging level")

    def get_serve_model_path(self) -> str:
        return self._get_setting(_SERVE_CATEGORY, _MODEL_PATH_SETTING, _DEF_MODEL_PATH)

    def get_serve_n_gpu_layers(self) -> int:
        return self._get_setting(
            _SERVE_CATEGORY, _NUM_GPUS_LAY_SETTING, _DEF_NUM_GPU_LAYERS
        )


def create_config_file(config_file_name="./config.yml"):
    # pylint: disable=line-too-long
    """
    Create default config file.
    TODO: Remove this function after config class is updated.

    Parameters:
    - config_path (str): Path to create the default config.yml

    Returns:
    - None
    """

    config_yml_txt = textwrap.dedent(
    """
    # Copyright The Authors
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    chat:
      context: ""
      model: "ggml-malachite-7b-0226-Q4_K_M"
      session: ""

    generate:
      model: "ggml-malachite-7b-0226-Q4_K_M"
      num_cpus: 10
      num_instructions_to_generate: 100
      path_to_taxonomy: "./taxonomy"
      prompt_file_path: "./cli/generator/prompt.txt"
      seed_tasks_path: "./cli/generator/seed_tasks.jsonl"

    list:
      path_to_taxonomy: "./taxonomy"

    log:
      level: info

    serve:
      model_path: "./models/ggml-malachite-7b-0226-Q4_K_M.gguf"
      n_gpu_layers: -1
    """
    )
    if not os.path.isfile(config_file_name):
        if os.path.dirname(config_file_name) != "":
            os.makedirs(os.path.dirname(config_file_name), exist_ok=True)
        with open(config_file_name, "w", encoding="utf-8") as model_file:
            model_file.write(config_yml_txt)

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
    chat_config_file_name = os.path.join(
        os.path.dirname(config_file_name), "chat-cli.toml"
    )
    if not os.path.isfile(chat_config_file_name):
        if os.path.dirname(chat_config_file_name) != "":
            os.makedirs(os.path.dirname(chat_config_file_name), exist_ok=True)
        with open(chat_config_file_name, "w", encoding="utf-8") as model_file:
            model_file.write(chat_config_toml_txt)
