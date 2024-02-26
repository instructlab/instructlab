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

import logging
import os
from typing import Any, Optional
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
        return self._get_setting(_GENERATE_CATEGORY, _PATH_TO_TAX_SETTING, _DEF_TAXONOMY_PATH)

    def get_generate_num_instructions(self) -> int:
        return self._get_setting(_GENERATE_CATEGORY, _NUM_INSTS_SETTING, _DEF_NUM_INSTRS)

    def get_generate_seed_task_path(self) -> str:
        return self._get_setting(_GENERATE_CATEGORY, _SEED_TASK_PATH_SETTING, _DEF_SEED_TASK_PATH)

    def get_generate_prompt_file_path(self) -> str:
        return self._get_setting(_GENERATE_CATEGORY, _PROMPT_FILE_PATH, _DEF_PROMPT_FILE_PATH)

    def get_list_taxonomy(self) -> str:
        return self._get_setting(_LIST_CATEGORY, _PATH_TO_TAX_SETTING, _DEF_TAXONOMY_PATH)
    
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
        return self._get_setting(_SERVE_CATEGORY, _NUM_GPUS_LAY_SETTING, _DEF_NUM_GPU_LAYERS)
       

