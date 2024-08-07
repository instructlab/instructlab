# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any
from unittest.mock import patch
import logging
import os
import pathlib
import shutil

# Third Party
import platformdirs
import pytest
import yaml

# First Party
from instructlab import configuration as config
from instructlab.configuration import DEFAULTS
from instructlab.log import configure_logging


class TestConfig:
    def _assert_defaults(self, cfg: config.Config):
        # redefine defaults here instead of relyin on those in configuration.DEFAULTS
        # to catch any errors if we are doing things incorrectly over there
        package_name = "instructlab"
        internal_dirname = "internal"
        data_dir = platformdirs.user_data_dir(package_name)
        cache_dir = platformdirs.user_cache_dir(package_name)
        default_model = f"{cache_dir}/models/merlinite-7b-lab-Q4_K_M.gguf"

        assert cfg.general is not None
        assert cfg.version is not None
        assert cfg.general.log_level == "INFO"
        assert cfg.general.debug_level == 0

        assert cfg.chat is not None
        assert cfg.chat.model == default_model
        assert not cfg.chat.vi_mode
        assert cfg.chat.visible_overflow
        assert cfg.chat.context == "default"
        assert cfg.chat.session is None
        assert cfg.chat.logs_dir == f"{data_dir}/chatlogs"
        assert not cfg.chat.greedy_mode

        assert cfg.evaluate is not None

        assert cfg.general is not None
        assert cfg.general.log_level == "INFO"

        assert cfg.generate is not None
        assert cfg.generate.teacher.model_path == default_model
        assert cfg.generate.teacher.llama_cpp is not None
        assert cfg.generate.teacher.llama_cpp.gpu_layers == -1
        assert cfg.generate.teacher.llama_cpp.max_ctx_size == 4096
        assert cfg.generate.teacher.llama_cpp.llm_family == ""
        assert cfg.generate.teacher.vllm is not None
        assert cfg.generate.teacher.vllm.vllm_args == []
        assert cfg.generate.teacher.host_port == "127.0.0.1:8000"
        assert cfg.generate.teacher.backend is None
        assert cfg.generate.teacher.chat_template is None
        assert cfg.generate.pipeline == "simple"
        assert cfg.generate.model == default_model
        assert cfg.generate.taxonomy_path == f"{data_dir}/taxonomy"
        assert cfg.generate.taxonomy_base == "origin/main"
        assert cfg.generate.num_cpus == 10
        assert cfg.generate.sdg_scale_factor == 30
        assert cfg.generate.chunk_word_count == 1000
        assert cfg.generate.output_dir == f"{data_dir}/datasets"
        assert cfg.generate.prompt_file == f"{data_dir}/{internal_dirname}/prompt.txt"
        assert (
            cfg.generate.seed_file == f"{data_dir}/{internal_dirname}/seed_tasks.json"
        )

        assert cfg.serve is not None
        assert cfg.serve.model_path == default_model
        assert cfg.serve.llama_cpp is not None
        assert cfg.serve.llama_cpp.gpu_layers == -1
        assert cfg.serve.llama_cpp.max_ctx_size == 4096
        assert cfg.serve.llama_cpp.llm_family == ""
        assert cfg.serve.vllm is not None
        assert cfg.serve.vllm.vllm_args == []
        assert cfg.serve.host_port == "127.0.0.1:8000"
        assert cfg.serve.backend is None
        assert cfg.serve.chat_template is None

    def _assert_model_defaults(self, cfg):
        package_name = "instructlab"
        cache_dir = platformdirs.user_cache_dir(package_name)
        default_model = f"{cache_dir}/models/merlinite-7b-lab-Q4_K_M.gguf"

        assert cfg.chat is not None
        assert cfg.chat.model == default_model

        assert cfg.evaluate is not None
        assert cfg.evaluate.base_model == "instructlab/granite-7b-lab"

        assert cfg.generate is not None
        assert cfg.generate.model == DEFAULTS.DEFAULT_MODEL

        assert cfg.serve is not None
        assert cfg.serve.model_path == DEFAULTS.DEFAULT_MODEL

        assert cfg.train is not None
        assert cfg.train.model_path == "instructlab/granite-7b-lab"

    def test_default_config(self, cli_runner):  # pylint: disable=unused-argument
        cfg = config.get_default_config()
        assert cfg is not None
        self._assert_defaults(cfg)
        self._assert_model_defaults(cfg)

    def test_cfg_auto_fill(self, tmp_path_home):
        config_path = tmp_path_home / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as config_file:
            config_file.write(
                """general:
  log_level: INFO
"""
            )
        cfg = config.read_config(config_path)
        self._assert_defaults(cfg)
        self._assert_model_defaults(cfg)

    def test_cfg_auto_fill_with_large_config(self, tmp_path_home):  # pylint: disable=unused-argument
        config_path = tmp_path_home / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as config_file:
            config_file.write(
                """general:
  log_level: INFO
generate:
  pipeline: simple
  teacher:
    model_path: models/granite-7b-lab-Q4_K_M.gguf
    chat_template: tokenizer
    llama_cpp:
      gpu_layers: 1
      max_ctx_size: 2048
      llm_family: ''
    vllm:
      gpus: 8
"""
            )
        cfg = config.read_config(config_path)
        # make sure the generate cfg passed is preserved
        assert cfg.generate.teacher.llama_cpp.max_ctx_size == 2048
        self._assert_model_defaults(cfg)

    def test_validate_log_level_invalid(self):
        cfg = config.get_default_config()
        with pytest.raises(ValueError):
            cfg.general.validate_log_level("INVALID")

    def test_validate_log_level_valid(self):
        cfg = config.get_default_config()
        assert cfg.general.validate_log_level("DEBUG") == "DEBUG"
        assert cfg.general.validate_log_level("INFO") == "INFO"
        assert cfg.general.validate_log_level("WARNING") == "WARNING"
        assert cfg.general.validate_log_level("WARN") == "WARN"
        assert cfg.general.validate_log_level("FATAL") == "FATAL"
        assert cfg.general.validate_log_level("CRITICAL") == "CRITICAL"
        assert cfg.general.validate_log_level("ERROR") == "ERROR"
        assert cfg.general.validate_log_level("NOTSET") == "NOTSET"

    def test_expand_paths(self, tmp_path_home):
        config_path = tmp_path_home / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as config_file:
            config_file.write(
                """\
version: 1.0.0
chat:
  model: $HOME/.cache/instructlab/models/granite-7b-lab-Q4_K_M.gguf
generate:
  model: ~/.cache/instructlab/models/granite-7b-lab-Q4_K_M.gguf
  taxonomy_base: upstream/main
  taxonomy_path: mytaxonomy
  teacher:
    model_path: ~/.cache/instructlab/models/granite-7b-lab-Q4_K_M.gguf
    chat_template: tokenizer
    llama_cpp:
      gpu_layers: 1
      max_ctx_size: 2048
      llm_family: ''
    vllm:
      gpus: 8
serve:
  model_path: models/granite-7b-lab-Q4_K_M.gguf
  chat_template: tokenizer
  llama_cpp:
    gpu_layers: 1
    max_ctx_size: 2048
    llm_family: ''
  vllm:
    gpus: 8
    vllm_args:
       - --qlora-adapter-name-or-path=$HOME/qlora-adapter-name-or-path
"""
            )
        cfg = config.read_config(config_path)
        assert cfg.chat.model.startswith("/")
        assert cfg.chat.model == os.path.expandvars(
            "$HOME/.cache/instructlab/models/granite-7b-lab-Q4_K_M.gguf"
        )
        assert cfg.generate.model.startswith("/")
        assert cfg.generate.model == os.path.expanduser(
            "~/.cache/instructlab/models/granite-7b-lab-Q4_K_M.gguf"
        )
        # Validate multi level dict
        assert cfg.generate.teacher.model_path.startswith("/")
        assert cfg.generate.teacher.model_path == os.path.expanduser(
            "~/.cache/instructlab/models/granite-7b-lab-Q4_K_M.gguf"
        )
        assert cfg.serve.vllm.vllm_args[0] == os.path.expandvars(
            "--qlora-adapter-name-or-path=$HOME/qlora-adapter-name-or-path"
        )

    def test_get_model_family(self):
        good_cases = {
            # two known families
            "merlinite": "merlinite",
            "mixtral": "mixtral",
            # case insensitive
            "MERLINiTe": "merlinite",
            # mapping granite to merlinite
            "granite": "merlinite",
            # default empty value of model_family will use name of model in model path to guess model_family
            "": "merlinite",
        }
        bad_cases = [
            # unknown family
            "unknown",
        ]
        for model_name, expected_family in good_cases.items():
            model_path = os.path.join("models", f"{model_name}-7b-lab-Q4_K_M.gguf")
            assert config.get_model_family(model_name, model_path) == expected_family
            assert config.get_model_family(None, model_path) == expected_family
        for model_name in bad_cases:
            model_path = os.path.join("models", f"{model_name}-7b-lab-Q4_K_M.gguf")
            with pytest.raises(config.ConfigException):
                config.get_model_family(model_name, model_path)

    def test_config_modified_settings(self, tmp_path_home):
        config_path = tmp_path_home / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as config_file:
            config_file.write(
                """\
version: 1.0.0
chat:
  model: models/granite-7b-lab-Q4_K_M.gguf
generate:
  pipeline: simple
  model: models/granite-7b-lab-Q4_K_M.gguf
  taxonomy_base: upstream/main
  taxonomy_path: mytaxonomy
  teacher:
    model_path: models/granite-7b-lab-Q4_K_M.gguf
    chat_template: tokenizer
    llama_cpp:
      gpu_layers: 1
      max_ctx_size: 2048
      llm_family: ''
    vllm:
      gpus: 8
serve:
  model_path: models/granite-7b-lab-Q4_K_M.gguf
  chat_template: tokenizer
  llama_cpp:
    gpu_layers: 1
    max_ctx_size: 2048
    llm_family: ''
  vllm:
    gpus: 8
    vllm_args:
       - --dtype=auto
       - --enable-lora
evaluate:
  base_model: instructlab/granite-7b-lab
  gpus: 1
  mmlu:
    few_shots: 2
    batch_size: auto
  mmlu_branch:
    tasks_dir: /path/to/sdg
  mt_bench:
    judge_model: prometheus
    output_dir: /dir/to/output
    max_workers: 5
  mt_bench_branch:
    taxonomy_path: taxonomy
general:
  log_level: DEBUG
"""
            )
        cfg = config.read_config(config_path)
        assert cfg is not None
        assert cfg.chat.model == "models/granite-7b-lab-Q4_K_M.gguf"
        assert cfg.generate.pipeline == "simple"
        assert cfg.generate.model == "models/granite-7b-lab-Q4_K_M.gguf"
        assert cfg.serve.llama_cpp.gpu_layers == 1
        assert cfg.serve.llama_cpp.max_ctx_size == 2048
        assert cfg.serve.chat_template == "tokenizer"
        assert cfg.serve.vllm.vllm_args == [
            "--dtype=auto",
            "--enable-lora",
        ]
        assert cfg.general.log_level == "DEBUG"
        assert cfg.general.debug_level == 1


@pytest.mark.parametrize(
    "log_level,debug_level,root,instructlab,openai_httpx",
    [
        ("INFO", 0, logging.INFO, logging.INFO, logging.ERROR),
        ("DEBUG", 1, logging.INFO, logging.DEBUG, logging.ERROR),
        ("DEBUG", 2, logging.DEBUG, logging.DEBUG, logging.DEBUG),
        ("ERROR", 0, logging.ERROR, logging.ERROR, logging.ERROR),
    ],
)
def test_logging(log_level, debug_level, root, instructlab, openai_httpx):
    configure_logging(log_level=log_level, debug_level=debug_level)
    assert logging.getLogger("root").getEffectiveLevel() == root
    assert logging.getLogger("instructlab").getEffectiveLevel() == instructlab
    assert logging.getLogger("openai").getEffectiveLevel() == openai_httpx
    assert logging.getLogger("httpx").getEffectiveLevel() == openai_httpx


@patch.multiple(
    config.DEFAULTS,
    _cache_home="/cache/instructlab",
    _config_dir="/config/instructlab",
    _data_dir="/data/instructlab",
)
def test_compare_default_config_testdata(
    testdata_path: pathlib.Path, tmp_path: pathlib.Path, regenerate_testdata: bool
):
    assert config.DEFAULTS.CHECKPOINTS_DIR == "/data/instructlab/checkpoints"
    saved_file = testdata_path / "default_config.yaml"
    current_file = tmp_path / "current_config.yaml"

    current_cfg = config.get_default_config()
    # roundtrip to verify serialization and de-serialization
    config.write_config(current_cfg, str(current_file))
    with current_file.open(encoding="utf-8") as yamlfile:
        current_content = yaml.safe_load(yamlfile)

    if regenerate_testdata:
        shutil.copy(current_file, saved_file)

    with saved_file.open(encoding="utf-8") as yamlfile:
        saved_content = yaml.safe_load(yamlfile)

    assert current_content == saved_content, (
        "current and expected configs are different. If the change was "
        "intentional, run 'make regenerate-testdata' and commit the "
        "updated test data."
    )


@pytest.mark.parametrize(
    "lora_quantize_dtype,additional_args,raises_exception",
    [
        ("nf4", {}, False),
        (None, {}, False),
        ("nf4", None, False),
        (None, None, False),
        ("valid-for-cli-but-not-for-training-library", {}, False),
        ("nf4", {"lora_alpha": 32}, False),
    ],
)
def test_read_train_profile(
    lora_quantize_dtype: str | None,
    additional_args: dict[str, Any] | None,
    raises_exception: bool,
    tmp_path_home,
):
    # define a profile with yaml
    profile_data = {
        "model_path": "/path/to/model",
        "data_path": "/path/to/data",
        "ckpt_output_dir": "/path/to/checkpoints",
        "data_output_dir": "/dev/shm",
        "max_seq_len": 4096,
        "max_batch_len": 60_000,
        "num_epochs": 10,
        "effective_batch_size": 3840,
        "save_samples": 250_000,
        "deepspeed_cpu_offload_optimizer": True,
        "lora_rank": 4,
        "lora_quantize_dtype": lora_quantize_dtype,
        "is_padding_free": False,
        "nproc_per_node": 8,
        "additional_args": additional_args,
    }
    train_profile = pathlib.Path(tmp_path_home, "profile.yaml")
    with open(train_profile, "w", encoding="utf-8") as outfile:
        yaml.dump(profile_data, outfile)

    if raises_exception:
        with pytest.raises(config.ConfigException):
            config.read_train_profile(train_profile)
    else:
        result = config.read_train_profile(train_profile)
        assert result is not None
        for k, v in result.dict().items():
            if k not in profile_data:
                continue
            if profile_data[k] is not None:
                assert v == profile_data[k]
