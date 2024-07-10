# SPDX-License-Identifier: Apache-2.0

# Standard
import os

# Third Party
import platformdirs
import pytest

# First Party
from instructlab import configuration as config


class TestConfig:
    @pytest.fixture(autouse=True)
    def _init_tmpdir(self, tmpdir):
        os.environ["HOME"] = str(tmpdir)
        self.tmpdir = tmpdir

    def _assert_defaults(self, cfg: config.Config):
        # redefine defaults here instead of relyin on those in configuration.DEFAULTS
        # to catch any errors if we are doing things incorrectly over there
        package_name = "instructlab"
        internal_dirname = "internal"
        data_dir = platformdirs.user_data_dir(package_name)
        default_model = f"{data_dir}/models/merlinite-7b-lab-Q4_K_M"

        assert cfg.general is not None
        assert cfg.general.log_level == "INFO"

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
        assert cfg.generate.model == default_model
        assert cfg.generate.taxonomy_path == f"{data_dir}/taxonomy"
        assert cfg.generate.taxonomy_base == "origin/main"
        assert cfg.generate.num_cpus == 10
        assert cfg.generate.num_instructions == 100
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
        assert cfg.serve.vllm.vllm_additional_args == []
        assert cfg.serve.vllm.served_model_name == "models/merlinite-7b-lab-Q4_K_M.gguf"
        assert cfg.serve.vllm.device == "cpu"
        assert cfg.serve.vllm.max_model_len == 4096
        assert cfg.serve.vllm.tensor_parallel_size == 1
        assert cfg.serve.host == "127.0.0.1"
        assert cfg.serve.port == 8000
        assert cfg.serve.backend is None

    def _assert_model_defaults(self, cfg):
        package_name = "instructlab"
        data_dir = platformdirs.user_data_dir(package_name)
        default_model = f"{data_dir}/models/merlinite-7b-lab-Q4_K_M"

        assert cfg.chat is not None
        assert cfg.chat.model == default_model

        assert cfg.evaluate is not None
        assert cfg.evaluate.base_model == "instructlab/granite-7b-lab"
        assert cfg.evaluate.model is None

        assert cfg.generate is not None
        assert cfg.generate.model == default_model

        assert cfg.serve is not None
        assert cfg.serve.model_path == default_model

        assert cfg.train is not None
        assert cfg.train.train_args is not None
        assert cfg.train.train_args.model_path == "instructlab/granite-7b-lab"

    def test_default_config(self, cli_runner):  # pylint: disable=unused-argument
        cfg = config.get_default_config()
        assert cfg is not None
        self._assert_defaults(cfg)
        self._assert_model_defaults(cfg)
