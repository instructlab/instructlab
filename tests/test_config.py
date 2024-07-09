# SPDX-License-Identifier: Apache-2.0

# Standard
import os

# Third Party
import pytest

# First Party
from instructlab import configuration as config


class TestConfig:
    @pytest.fixture(autouse=True)
    def _init_tmpdir(self, tmpdir):
        self.tmpdir = tmpdir

    def _assert_defaults(self, cfg):
        assert cfg.chat is not None
        assert not cfg.chat.vi_mode
        assert cfg.chat.visible_overflow
        assert cfg.chat.context == "default"
        assert cfg.chat.session is None
        assert cfg.chat.logs_dir == "data/chatlogs"
        assert not cfg.chat.greedy_mode

        assert cfg.evaluate is not None

        assert cfg.general is not None
        assert cfg.general.log_level == "INFO"

        assert cfg.generate is not None
        assert cfg.generate.taxonomy_path == "taxonomy"
        assert cfg.generate.taxonomy_base == "origin/main"
        assert cfg.generate.num_cpus == 10
        assert cfg.generate.num_instructions == 100
        assert cfg.generate.chunk_word_count == 1000
        assert cfg.generate.output_dir == "generated"
        assert cfg.generate.prompt_file == "prompt.txt"
        assert cfg.generate.seed_file == "seed_tasks.json"

        assert cfg.serve is not None
        assert cfg.serve.llama_cpp is not None
        assert cfg.serve.llama_cpp.gpu_layers == -1
        assert cfg.serve.llama_cpp.max_ctx_size == 4096
        assert cfg.serve.llama_cpp.llm_family == ""
        assert cfg.serve.vllm is not None
        assert cfg.serve.vllm.vllm_args == []
        assert cfg.serve.host_port == "127.0.0.1:8000"
        assert cfg.serve.backend is None

    def _assert_model_defaults(self, cfg):
        assert cfg.chat is not None
        assert cfg.chat.model == "models/merlinite-7b-lab-Q4_K_M.gguf"

        assert cfg.evaluate is not None
        assert cfg.evaluate.base_model == "instructlab/merlinite-7b-lab"
        assert cfg.evaluate.model is None

        assert cfg.generate is not None
        assert cfg.generate.model == "models/merlinite-7b-lab-Q4_K_M.gguf"

        assert cfg.serve is not None
        assert cfg.serve.model_path == "models/merlinite-7b-lab-Q4_K_M.gguf"

        assert cfg.train is not None
        assert cfg.train.train_args is not None
        assert cfg.train.train_args.model_path == "instructlab/merlinite-7b-lab"

    def test_default_config(self):
        cfg = config.get_default_config()
        assert cfg is not None
        self._assert_defaults(cfg)
        self._assert_model_defaults(cfg)

    def test_minimal_config(self):
        config_path = self.tmpdir.join("config.yaml")
        with open(config_path, "w", encoding="utf-8") as config_file:
            config_file.write(
                """chat:
  model: models/merlinite-7b-lab-Q4_K_M.gguf
generate:
  model: models/merlinite-7b-lab-Q4_K_M.gguf
  taxonomy_base: origin/main
  taxonomy_path: taxonomy
serve:
  model_path: models/merlinite-7b-lab-Q4_K_M.gguf
  llama_cpp:
    gpu_layers: -1
    max_ctx_size: 4096
    llm_family: ''
  vllm:
    vllm_args: []
evaluate:
  base_model: instructlab/granite-7b-lab
  gpus: 1
  mmlu:
    few_shots: 2
    batch_size: 5
  mmlu_branch:
    sdg_path: /path/to/sdg
  mt_bench:
    judge_model: prometheus
    output_dir: /dir/to/output
    max_workers: 5
  mt_bench_branch:
    taxonomy_path: taxonomy
"""
            )
        cfg = config.read_config(config_path)
        assert cfg is not None
        self._assert_defaults(cfg)

    def test_full_config(self):
        config_path = self.tmpdir.join("config.yaml")
        with open(config_path, "w", encoding="utf-8") as config_file:
            config_file.write(
                """general:
  log_level: INFO
chat:
  context: default
  greedy_mode: false
  logs_dir: data/chatlogs
  model: models/merlinite-7b-lab-Q4_K_M.gguf
  session: null
  vi_mode: false
  visible_overflow: true
generate:
  model: models/merlinite-7b-lab-Q4_K_M.gguf
  num_cpus: 10
  num_instructions: 100
  output_dir: generated
  prompt_file: prompt.txt
  seed_file: seed_tasks.json
  taxonomy_base: origin/main
  taxonomy_path: taxonomy
  chunk_word_count: 1000
serve:
  backend: null
  host_port: 127.0.0.1:8000
  llama_cpp:
    gpu_layers: -1
    max_ctx_size: 4096
    llm_family: ''
  model_path: models/merlinite-7b-lab-Q4_K_M.gguf
  vllm:
    vllm_args: []
evaluate:
  base_model: instructlab/granite-7b-lab
  gpus: 1
  mmlu:
    few_shots: 2
    batch_size: 5
  mmlu_branch:
    sdg_path: /path/to/sdg
  mt_bench:
    judge_model: prometheus
    output_dir: /dir/to/output
    max_workers: 5
  mt_bench_branch:
    taxonomy_path: taxonomy
"""
            )
        cfg = config.read_config(config_path)
        assert cfg is not None
        self._assert_defaults(cfg)

    def test_config_unexpected_fields(self):
        print(dir(self.tmpdir))
        config_path = self.tmpdir.join("config.yaml")
        with open(config_path, "w", encoding="utf-8") as config_file:
            config_file.write(
                """chat:
  model: models/merlinite-7b-lab-Q4_K_M.gguf
generate:
  model: models/merlinite-7b-lab-Q4_K_M.gguf
  taxonomy_base: origin/main
  taxonomy_path: taxonomy
serve:
  model_path: models/merlinite-7b-lab-Q4_K_M.gguf
  llama_cpp:
    gpu_layers: -1
    max_ctx_size: 4096
    llm_family: ''
  vllm:
    vllm_args: []
evaluate:
  base_model: instructlab/granite-7b-lab
  gpus: 1
  mmlu:
    few_shots: 2
    batch_size: 5
  mmlu_branch:
    sdg_path: /path/to/sdg
  mt_bench:
    judge_model: prometheus
    output_dir: /dir/to/output
    max_workers: 5
  mt_bench_branch:
    taxonomy_path: taxonomy
unexpected:
  field: value
"""
            )
        cfg = config.read_config(config_path)
        self._assert_defaults(cfg)

    def test_config_missing_required_field_groups(self):
        config_path = self.tmpdir.join("config.yaml")
        with open(config_path, "w", encoding="utf-8") as config_file:
            config_file.write(
                """general:
  log_level: INFO
"""
            )
        with pytest.raises(
            config.ConfigException,
            match=r"""4 errors in [\/\w-]+config.yaml:
- missing chat: field required
- missing generate: field required
- missing serve: field required
- missing evaluate: field required
""",
        ):
            config.read_config(config_path)

    def test_config_missing_required_fields(self):
        config_path = self.tmpdir.join("config.yaml")
        with open(config_path, "w", encoding="utf-8") as config_file:
            config_file.write(
                """general:
  log_level: INFO
generate:
  model: models/merlinite-7b-lab-Q4_K_M.gguf
"""
            )
        with pytest.raises(
            config.ConfigException,
            match=r"""5 errors in [\/\w-]+config.yaml:
- missing chat: field required
- missing generate->taxonomy_path: field required
- missing generate->taxonomy_base: field required
- missing serve: field required
- missing evaluate: field required
""",
        ):
            config.read_config(config_path)

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

    def test_get_model_family(self):
        good_cases = {
            # two known families
            "merlinite": "merlinite",
            "mixtral": "mixtral",
            # case insensitive
            "MERLINiTe": "merlinite",
            # mapping granite to merlinite
            "granite": "merlinite",
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

    def test_config_modified_settings(self):
        config_path = self.tmpdir.join("config.yaml")
        with open(config_path, "w", encoding="utf-8") as config_file:
            config_file.write(
                """\
chat:
  model: models/granite-7b-lab-Q4_K_M.gguf
generate:
  model: models/granite-7b-lab-Q4_K_M.gguf
  taxonomy_base: upstream/main
  taxonomy_path: mytaxonomy
serve:
  model_path: models/granite-7b-lab-Q4_K_M.gguf
  llama_cpp:
    gpu_layers: 1
    max_ctx_size: 2048
    llm_family: ''
  vllm:
    vllm_args:
       - --dtype=auto
       - --enable-lora
evaluate:
  base_model: instructlab/granite-7b-lab
  gpus: 1
  mmlu:
    few_shots: 2
    batch_size: 5
  mmlu_branch:
    sdg_path: /path/to/sdg
  mt_bench:
    judge_model: prometheus
    output_dir: /dir/to/output
    max_workers: 5
  mt_bench_branch:
    taxonomy_path: taxonomy
"""
            )
        cfg = config.read_config(config_path)
        assert cfg is not None
        assert cfg.chat.model == "models/granite-7b-lab-Q4_K_M.gguf"
        assert cfg.generate.model == "models/granite-7b-lab-Q4_K_M.gguf"
        assert cfg.serve.llama_cpp.gpu_layers == 1
        assert cfg.serve.llama_cpp.max_ctx_size == 2048
        assert cfg.serve.vllm.vllm_args == [
            "--dtype=auto",
            "--enable-lora",
        ]
