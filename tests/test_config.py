# Standard
import tempfile
import unittest

# First Party
from cli import config

TEST_CONFIG_1 = b"""chat:
  context: default
  model: ggml-merlinite-7b-0302-Q4_K_M
  session: null
  vi_mode: false
  visible_overflow: true
  logs_dir: ./logs/
  greedy_mode: false
general:
  log_level: INFO
generate:
  model: ggml-merlinite-7b-0302-Q4_K_M
  num_cpus: 10
  num_instructions: 100
  prompt_file: prompt.txt
  seed_file: seed_tasks.json
  taxonomy_path: /tmp/instruct-lab-taxonomy
  output_dir: /tmp
serve:
  gpu_layers: -1
  model_path: models/ggml-merlinite-7b-0302-Q4_K_M.gguf
  host_port: localhost:8000
"""

TEST_CONFIG_UNEXPECTED_ARGS = b"""
unexpected:
  /this/is/an/unexpected/argument
"""

TEST_CONFIG_OLD = b"""chat:
  context: default
  model: ggml-merlinite-7b-0302-Q4_K_M
  session: null
  vi_mode: false
  visible_overflow: true
general:
  log_level: INFO
generate:
  model: ggml-merlinite-7b-0302-Q4_K_M
  num_cpus: 10
  num_instructions: 100
  prompt_file: prompt.txt
  seed_file: seed_tasks.json
  taxonomy_path: /tmp/instruct-lab-taxonomy
  output_dir: /tmp
serve:
  gpu_layers: -1
  model_path: models/ggml-merlinite-7b-0302-Q4_K_M.gguf
"""


class TestConfig(unittest.TestCase):
    def setUp(self):
        # pylint: disable=consider-using-with
        self.temp = tempfile.NamedTemporaryFile(
            prefix="config", suffix=".yaml", delete=False,
        )

    # Tests basic config parsing
    def test_config(self):
        self.temp.write(TEST_CONFIG_1)
        self.temp.flush()

        cfg = config.read_config(self.temp.name)
        assert cfg is not None
        assert cfg.serve is not None
        assert cfg.serve.gpu_layers == -1
        assert cfg.serve.model_path == "models/ggml-merlinite-7b-0302-Q4_K_M.gguf"
        assert cfg.chat.context == "default"
        assert cfg.chat.model == "ggml-merlinite-7b-0302-Q4_K_M"
        assert cfg.chat.session is None
        assert cfg.chat.vi_mode is False
        assert cfg.chat.visible_overflow is True
        assert cfg.general.log_level == "INFO"
        assert cfg.generate.model == "ggml-merlinite-7b-0302-Q4_K_M"
        assert cfg.generate.num_cpus == 10
        assert cfg.generate.num_instructions == 100
        assert cfg.generate.prompt_file == "prompt.txt"
        assert cfg.generate.seed_file == "seed_tasks.json"
        assert cfg.generate.taxonomy_path == "/tmp/instruct-lab-taxonomy"

    # Tests that additional lines in the config do not cause errors
    def test_config_unexpected_arguments(self):
        self.temp.write(TEST_CONFIG_1 + TEST_CONFIG_UNEXPECTED_ARGS)
        self.temp.flush()

        cfg = config.read_config(self.temp.name)
        assert cfg is not None
        assert cfg.serve is not None
        assert cfg.serve.gpu_layers == -1

    # Tests that removed config produces an appropriate error
    def test_config_missing_arguments(self):
        self.temp.write(TEST_CONFIG_OLD)
        self.temp.flush()

        try:
            config.read_config(self.temp.name)
            assert False  # should not be reached
        except config.ConfigException as ex:
            assert "greedy_mode" in str(ex)
            assert "missing in section" in str(ex)
