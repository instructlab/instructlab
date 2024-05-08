# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from instructlab import config


class TestConfig:
    @pytest.fixture(autouse=True)
    def _init_tmpdir(self, tmpdir):
        self.tmpdir = tmpdir

    def _assert_defaults(self, cfg):
        assert cfg.general is not None
        assert cfg.general.log_level == "INFO"

        assert cfg.chat is not None
        assert cfg.chat.model == "merlinite-7b-lab-Q4_K_M"
        assert not cfg.chat.vi_mode
        assert cfg.chat.visible_overflow
        assert cfg.chat.context == "default"
        assert cfg.chat.session is None
        assert cfg.chat.logs_dir == "data/chatlogs"
        assert not cfg.chat.greedy_mode

        assert cfg.generate is not None
        assert cfg.generate.model == "merlinite-7b-lab-Q4_K_M"
        assert cfg.generate.taxonomy_path == "taxonomy"
        assert cfg.generate.taxonomy_base == "origin/main"
        assert cfg.generate.num_cpus == 10
        assert cfg.generate.num_instructions == 100
        assert cfg.generate.chunk_word_count == 1000
        assert cfg.generate.output_dir == "generated"
        assert cfg.generate.prompt_file == "prompt.txt"
        assert cfg.generate.seed_file == "seed_tasks.json"

        assert cfg.serve is not None
        assert cfg.serve.model_path == "models/merlinite-7b-lab-Q4_K_M.gguf"
        assert cfg.serve.gpu_layers == -1
        assert cfg.serve.host_port == "127.0.0.1:8000"
        assert cfg.serve.max_ctx_size == 4096

    def test_default_config(self):
        cfg = config.get_default_config()
        assert cfg is not None
        self._assert_defaults(cfg)

    def test_minimal_config(self):
        config_path = self.tmpdir.join("config.yaml")
        with open(config_path, "w", encoding="utf-8") as config_file:
            config_file.write(
                """chat:
  model: merlinite-7b-lab-Q4_K_M
generate:
  model: merlinite-7b-lab-Q4_K_M
  taxonomy_base: origin/main
  taxonomy_path: taxonomy
serve:
  model_path: models/merlinite-7b-lab-Q4_K_M.gguf
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
  model: merlinite-7b-lab-Q4_K_M
  session: null
  vi_mode: false
  visible_overflow: true
generate:
  model: merlinite-7b-lab-Q4_K_M
  num_cpus: 10
  num_instructions: 100
  output_dir: generated
  prompt_file: prompt.txt
  seed_file: seed_tasks.json
  taxonomy_base: origin/main
  taxonomy_path: taxonomy
  chunk_word_count: 1000
serve:
  gpu_layers: -1
  host_port: 127.0.0.1:8000
  max_ctx_size: 4096
  model_path: models/merlinite-7b-lab-Q4_K_M.gguf
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
  model: merlinite-7b-lab-Q4_K_M
generate:
  model: merlinite-7b-lab-Q4_K_M
  taxonomy_base: origin/main
  taxonomy_path: taxonomy
serve:
  model_path: models/merlinite-7b-lab-Q4_K_M.gguf
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
            match=r"""3 errors in [\/\w-]+config.yaml:
- missing chat: field required
- missing generate: field required
- missing serve: field required
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
  model: merlinite-7b-lab-Q4_K_M
"""
            )
        with pytest.raises(
            config.ConfigException,
            match=r"""4 errors in [\/\w-]+config.yaml:
- missing chat: field required
- missing generate->taxonomy_path: field required
- missing generate->taxonomy_base: field required
- missing serve: field required
""",
        ):
            config.read_config(config_path)
