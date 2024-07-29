## v0.18

### Features

* `ilab data generate` now supports parallelized data generation across batches of the seed
   data when running with a the vLLM serving. The `--batch-size` argument can be used to
   control this behavior.
* `ilab model download` now supports downloading models from OCI registries. Repositories
   that are prefixed by "docker://" and specified against `--repository` are treated as OCI
   registries.
* `ilab` now uses dedicated directories for storing config and data files. On Linux, these
  will generally be the XDG directories: `~/.config/instructlab` for config,
  `~/.local/share/instructlab` for data, and `~/.cache` for temporary files, including downloaded
  models. On MacOS, both the config and data is located at
  `~/Library/Application Support/instructlab`.
* A new `ilab config show` command is introduced as a convenience feature, which prints out
   the contents of the ***actively loaded*** config, not just the contents of the config file.
* `ilab system`: A new command group named `ilab system` has been added which will serve as the
   basis for all system-related commands. This currently contains `ilab system info` as its only
   sub-command.
* Add [vLLM](https://github.com/vllm-project/vllm) backend to serve, chat and generate commands.
* Add `--backend` flag to `ilab model serve` command to allow for specifying the backend to use
   when serving a model. This is useful when you have multiple backends installed and want to
   specify which one to use. Currently, the only supported backend are `llama-cpp` and `vllm`.
* Update `llama-cpp-python` to latest upstream release 0.2.79 to address poor
  results of synthetic data generation and local training.
* Adding `ilab model evaluate` which uses the new backend serving functionality.  Evaluate offers
   two standard benchmarks (mt-bench and mmlu) as well as two variations (mt-bench-branch and
   mmlu-branch) which are integrated with the ilab workflow to evaluate new skills and knowledge.
   Includes --gpus option for specifying number of gpus to utilize when serving models for
   evaluation (currently applicable for vLLM only).  Also includes --merge-system-user-message
   flag to enable Mistral based judge models and a --enable-serving-output flag that
   configures whether the output of the model serving backend is suppressed.
* The `ilab` command now accepts a `-v` / `--verbose` option to enable debug logging.
  `ilab -vv` or `ilab --verbose --verbose` enables more verbose debug logging.
* `ilab model test` generic support
* Add `--chat-template` option to `ilab model serve` to support customization of the chat
   template for both vLLM and llama.cpp backends. Options include 'auto' (current behavior, ilab
   provides its own template), 'tokenizer' (uses the model's tokenizer config), and an external
   file name.
* Default log format changes to include the logger name in the logs.
* `ilab data generate` now supports a new and more extensive pipeline with the
  option `--pipeline full`. This option requires `mixtral-8x7b-instruct` as the
  teacher model.
* The `instructlab` package now uses optional dependencies for each supported hardware `cpu`,
  `cuda`, `hpu`, `mps`, and `rocm`. To install InstructLab for e.g. NVIDIA CUDA, use
  `pip install instructlab[cuda]`.
* Add a `--enable-serving-output` flag for `ilab data generate`. This flag determines whether vLLM
  will have its output suppressed when it serves the teacher model in the background.
* The `generate` section of the config now has a `teacher` section. This section configures the
  teacher model when it is automatically served in the background. This new section has the same
  values as the `serve` section of the config.
* Support for `ILAB_GLOBAL_CONFIG` environment variable: When set, this environment variable
   specifies a global configuration file that serves as the template for the
   `~/.config/instructlab/config.yaml` user space config. This bypasses the interactive mode in
   `ilab config init` and can be used to specify alternative configurations for any command,
   ensuring that defaults such as taxonomy repositories and base models are honored from the global
   config.
* `ilab model list`: a new command which lists all GGUF and Safetensor Models on the system.

### Breaking Changes

* `ilab`: **Deprecation of Python 3.9 support and withdrawal of Python 3.12 support** Due to changes to training requiring the usage of [GPTDolomite](https://github.com/instructlab/GPTDolomite), Python 3.9 is no longer supported and Python 3.12 support is currently withdrawn. If you are using either of these versions, you will need to start using either Python 3.10 or Python 3.11 to use this and subsequent versions of the CLI.
* `ilab model train`: The '--device' parameter no longer supports specifying a GPU index (e.g., 'cuda:0'). To use a specific GPU, set the visible GPU before running the train command.
* `ilab init`: With the introduction of a dedicated storage system within the `ilab` CLI,
   `ilab init` and `ilab config init` will now output and read the config file from the
   platform's config directory under the `instructlab` package.
* `ilab taxonomy` and `ilab data`: The `ilab` CLI now uses the platform's dedicated data directory to store
   the taxonomy under the `instructlab/taxonomy` directory as a default.
* `ilab data`: The default directory for new datasets is now under `instructlab/datasets` in the
   platform's dedicated data directory under the `instructlab` package.
* `ilab model`: The default location for saved and downloaded models is now under `instructlab/models`
   in the platform's dedicated data directory under the `instructlab` package. Outputted
   checkpoints now live in the `instructlab/checkpoints` directory under the platform's dedicated
   program cache directory.
* `ilab model chat`: Chatlogs are now stored under the `instructlab/checkpoints` directory in the
   platform's dedicated data directory under the `instructlab` package.
* The `--num-instructions` option to `ilab data generate` has been deprecated.
  See `--sdg-scale-factor` for an updated option providing similar
  functionality.
* `ilab model train --legacy`: Trained GGUF models are now saved in the global user checkpoints directory.
  Previously, checkpoints were always saved into a directory local to where the user called it from.

### Fixes

* `ilab config`: Fixed a bug where `ilab` didn't recognize `train.lora_quantize_dtype: null` as a valid
   value.

## v0.17

### Features

#### ilab command redesign

The ilab command redesign included in v0.17 introduces a new command structure that follows a resource group design. This means that commands that once were something like `ilab chat` now are `ilab model chat`. The new groups are model, data, taxonomy, and config. The commands that fall under these are all of the pre-existing `ilab` commands just now grouped by the resource which the command commonly deals with.

The old command structure is still aliased to work but will be removed in 0.19.0. This means for 0.17.0 and 0.18.0 the aliases will exist and work as expected.

### Breaking Changes
