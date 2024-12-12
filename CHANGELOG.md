## v0.22

### Breaking Changes

### Features

* `ilab train --pipeline=accelerated --strategy=lab-skills-only` supports training with only the skills phase (leaving out knowledge).
* Previously, System Profile auto-detection was done by reading the names of the YAML files and matching them to your hardware. We now depend on the `Metadata` class stored in the configuration file itself. Please select `y` when prompted to over-write your existing system profiles to utilize the the new auto-detection system.

## v0.21

### Breaking Changes

* train-profiles have been deprecated and replaced with system-profiles. These profiles follow the format of the config file and apply to all commands. They live in `~/.local/share/instructlab/internal/system_profiles`
* The default model has been changed from Merlinite to Granite - see <https://github.com/instructlab/instructlab/issues/2238> for more details
* Removed the `--greedy-mode` flag from `ilab model chat`. Please update any scripts or workflows relying on `--greedy-mode` to ensure compatibility.

### Features

* `ilab` now supports system profiles. These profiles apply entire configuration files tailored to specific hardware configurations. We support a set of auto-detected profiles for CPU enabled Linux machines, M-Series Apple Silicon Chips, and Nvidia GPUs, and Intel Gaudi 3. When you run `ilab config init`, one of these profiles should be selected for you. If there is not a direct match, a menu will be displayed allowing you to choose one.
* Add support for inferencing with IBM granite architecture models.
* `ilab model chat` now includes a temperature setting feature, allowing users to adjust the response generation behavior. A default temperature of 1.0 has been added to the configuration file, which can be customized for each chat session using the `--temperature` or `-t` flag. Lower values produce
more deterministic and focused responses, while higher values increase variability.
* the `full` training pipeline now fits on devices with 16 and 32 GB of RAM! If you are on a Mac, these optimizations are done for you. If you are on Linux try using `--optimize-memory`, results vary per CPU vendor.
* `ilab data generate` now has `--max-num-tokens` which defaults to 4096. This flag can be used to generate less data per SDG run. Specifying a value like `512` results in a faster run with less data generated. This works well with consumer hardware and will reduce training time.

* `ilab model download` uses the `hf_transfer` library for faster model downloads reducing the average download time by 60%. This only applies to models that are hosted on Hugging Face Hub. This can be disabled by setting the environment variable `HF_HUB_ENABLE_HF_TRANSFER` to `0`.

## v0.20

### Breaking Changes

* vLLM has been upgraded to [v0.6.2](https://github.com/opendatahub-io/vllm/releases/tag/v0.6.2) and will need to be reinstalled if you are upgrading `ilab` from an older version
* Intel Gaudi software has been updated to 1.18.0 with Python 3.11 and
  Torch 2.4.0.

## v0.19

### Breaking Changes

* InstructLab now uses XDG-based directories on macOS, similar to Linux.
  Users are advised to re-initialize their config files and remove cached models.
* Removed unused argument `--rouge-threshold` of `ilab data generate`
* Removed the following aliased commands:
  * `convert`
  * `diff`
  * `download`
  * `evaluate`
  * `init`
  * `list`
  * `sysinfo`
  * `test`
* Intel Gaudi software has been updated to 1.17.1 with Python 3.11 and
  Torch 2.3.1 support.
* `--legacy` has been removed and replaced with `--pipeline=simple` in `ilab model train`
* `ilab data generate` now defaults to `--pipeline full` and uses the  `TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_M.gguf` model as the teacher. This provides increased performance and better generated data.

### Features

* `ilab config init` now auto detects your hardware when running on Nvidia enabled systems and chooses the best train profile. It does this by checking first if your system directly matches one of our supported train profiles and then attempts to match the vRAM for each profile to the total vRAM on your system.
* Add `log_format` to the `config.yaml` file to allow for customizing the log format.
* `ilab model evaluate --max-workers=auto` is now supported and is the default option. When
  auto is specified, the optimal value is determined based on your GPUs, CPUs, and
  configuration.
* `ilab model train` now supports `--pipeline`. The supported pipelines are `simple`, `full`, and `accelerated`. Simple preserves the functionality found in `--legacy` and current MacOS training. Full introduces a new training loop optimized for CPU and MacOS performance. `accelerated` allows users with dedicated graphics cards to run the full fine tuning and multi-phase training found in our training library.
* `--device` in `ilab model train` now supports `mps` which stands for Metal Performance Shaders. This is a PyTorch device for MacOS training that allows us to utilize the same code path for Linux and MacOS.
* Multi-phase training with `ilab model train --strategy lab-multiphase` is now resumeable! In the case of a machine failure or an incidental stop, progress is tracked in a new `journal` file, rendered in yaml.
  Upon restarting training, the user can confirm whether they would like to proceed with a pre-existing training run (one that might have only evaluated a few checkpoints of the first eval phase, for instance)
  or restart from scratch.
* Allow users to pick a distributed training backend framework for GPU accelerated training between 'fsdp' and 'deepspeed'. Also add support for FSDP specific configuration options.

## v0.18.1

### Features

* `ilab data generate` and `ilab taxonomy diff` now support `--taxonomy-base=empty` to allow
  specifying that all taxonomy files in the supplied repo should be included.

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
   two standard benchmarks (mt_bench and mmlu) as well as two variations (mt_bench_branch and
   mmlu_branch) which are integrated with the ilab workflow to evaluate new skills and knowledge.
   Includes `--gpus` option for specifying number of gpus to utilize when serving models for
   evaluation (currently applicable for vLLM only).  Also includes `--merge-system-user-message`
   flag to enable Mistral based judge models and a `--enable-serving-output` flag that
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
* `ilab data list`: a new command which lists the generated datasets in the user's datasets
  directory.
* Legacy Linux training now supports the new messages format. When a dataset is provided in the
  HuggingFace messages format, `ilab` will automatically convert it back into the legacy format.
* Legacy Linux training is now compatible with the phase07 pretraining format.
* Add support for `ILAB_TRAIN_PROFILE_DIR` which will point to the template train profiles to be brought into the `train_configuration` directory.
* Add interactive prompt for users to choose their train profile.
* The `generate` section of the config now has a `pipeline` value. This value sets a default value
  and can be overridden by the `--pipeline` flag. The default for this value is 'simple'.

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
* `ilab model chat`: Fixed an issue where the default served model couldn't be resolved when running
   model besides the default `merlinite-7b-lab-Q4_K_M.gguf`.

## v0.17

### Features

#### ilab command redesign

The ilab command redesign included in v0.17 introduces a new command structure that follows a resource group design. This means that commands that once were something like `ilab chat` now are `ilab model chat`. The new groups are model, data, taxonomy, and config. The commands that fall under these are all of the pre-existing `ilab` commands just now grouped by the resource which the command commonly deals with.

The old command structure is still aliased to work but will be removed in 0.19.0. This means for 0.17.0 and 0.18.0 the aliases will exist and work as expected.

### Breaking Changes
