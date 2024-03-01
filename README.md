# Labrador 🐶 command-line interface `lab`

Labrador 🐶 is a novel synthetic data-based alignment tuning method for Large
Language Models (LLMs.) The "**lab**" in **Lab**rador 🐶 stands for **L**arge-scale **A**lignment for Chat **B**ots.

This command-line interface for Labrador 🐶 (`lab`) will allow you to create models tuned
with your data using the Labrador 🐶 method on your laptop or workstation.

_This is currently a tool that **requires an M1/M2/M3 Mac** to use; we anticipate
future support for 🐧 Linux and other operating systems as well as for
💻 additional hardware._

## Contents:

- [Labrador 🐶 command-line interface `lab`](#labrador--command-line-interface-lab)
  - [Contents:](#contents)
- [Getting `lab`](#getting-lab)
  - [📋 Requirements](#-requirements)
  - [🧰 Program Installation](#-program-installation)
    - [Installing from GitHub (I just want it to work! 🚀)](#installing-from-github-i-just-want-it-to-work-)
    - [Installing from Source (I want to start developing! 🛠️)](#installing-from-source-i-want-to-start-developing-️)
  - [🚀 Verifying `lab` installation](#-verifying-lab-installation)
- [How to use `lab`](#how-to-use-lab)
  - [🏗️ 1. Initial setup](#️-1-initial-setup)
    - [Prepare the CLI's configuration](#prepare-the-clis-configuration)
    - [Download model](#download-model)
  - [🧑‍🏫 2. Model training](#-2-model-training)
    - [Serve the model](#serve-the-model)
    - [Test the model with chat before training](#test-the-model-with-chat-before-training)
    - [Generate a dataset](#generate-a-dataset)
    - [Train the model](#train-the-model)
  - [👩🏽‍🔬 3. Testing the fine-tuned model](#-3-testing-the-fine-tuned-model)
    - [Serve the fine-tuned model](#serve-the-fine-tuned-model)
    - [Try out the new model](#try-out-the-new-model)
    - [Run tests](#run-tests)
  - [🎁 4. Submit your dataset!](#-4-submit-your-dataset)
  - [Contributing](#contributing)
  - [Other stuffs](#other-stuffs)

<a name="getting"></a>

# Getting `lab`

## 📋 Requirements

- 🐍 Python 3.9 or later (CLang dsitribution of Python: 15.0.0 from xcode)
- 🍎 macOS (14.x with an M1/M2/M3 Metal/GPU)
- 📦 A quantized model in GGUF format (or read our [guide](#model-convert-quant) on to convert
  models to GGUF format and quantize them.)
- `gh` cli: Install [Github command cli](https://cli.github.com/) for downloading models from Github

🗒️ **Note:** The steps below use [Python venv](https://docs.python.org/3/library/venv.html) for virtual environments. If you have used [pyenv](https://github.com/pyenv/pyenv),
[Conda Miniforge](https://github.com/conda-forge/miniforge), or another tool for Python version management on your laptop, then use the virtual environment with that tool instead. Otherwise, you may have issues with packages installed but modules
from that package not found as they are linked to your Python version management tool and not `venv`.

## 🧰 Program Installation

The `lab` CLI will be available from PyPI using `pip3 install lab-cli` in the future.
For now, we offer two ways to get started:

### Installing from GitHub (I just want it to work! 🚀)

Let's start at an example folder `~/Documents/github` on your computer.

We'll create a new directory called `labrador` to store the files that this CLI needs when it runs.

```ShellSession
mkdir labrador
cd labrador
python3 -m venv venv
source venv/bin/activate
pip install git+ssh://git@github.com/open-labrador/cli.git
```

These are the steps that we're executing above, in plain language:

1. Create the new `labrador` directory.
2. `cd` into that directory.
3. In the `labrador` directory, created a new Python virtual environment.
4. Turn the virtual environment on.
5. Install the latest main-branch labrador cli program from GitHub in the new virtual environment.

**NOTE**: You're free to name your new directory, that we called `labrador`, anything you want!

### Installing from Source (I want to start developing! 🛠️)

We're keeping these detailed instructions in `CONTRIBUTING.MD` to keep this `README.MD` brief.

## 🚀 Verifying `lab` installation

In order for `lab` to run correctly in your terminal (or shell) window, you'll always need the Python
virtual environment, with `lab` installed, to be turned on.

```ShellSession
source venv/bin/activate
```

See "Installing" above if you haven't completed that step already!

If `lab` is installed correctly, you should be able to run:

```ShellSession
lab
```

Congrats! You're ready to get started 😁

# How to use `lab`

**NOTE**: The following instructions assume that you've followed the "Installation" instructions above, including installing `lab` from GitHub into a Python virtual environment.

The Labrador 🐶 CLI `lab` requires a few setup steps- in these instructions, we'll guide you through getting started.
You can see a flow chart showing the order of commands in a typical workflow as well as detailed command documentation below:

![flow diagram](docs/workflow.png)

## 🏗️ 1. Initial setup

### Prepare the CLI's configuration

- Inside the `labrador` directory that we created in the installation step, run the following:

  ```shell
  lab init
  ```

  This will add a new, default `config.yaml` file, and clone the `git@github.com:open-labrador/taxonomy.git` repository into the `labrador` directory.

### Download model

- Download the model to train using the **download** command:

  ```shell
  lab download
  ```

  This will download all the pre-trained models from the latest [release](https://github.com/open-labrador/cli/releases) into the `/models` directory.

- Manually download models:

  Pop over to our [cli releases](https://github.com/open-labrador/cli/releases) to check out the list of available models and a set of instructions on how to do this manually if necessary.

  📋 **Note:** Once you have the model chunks downloaded and reassembled according to the instructions above, please move the model to a `models/` directory in the root directory of your git checkout of this project (this assumes the model is in your `Downloads/` folder):

  ```
  mkdir models
  mv ~/Downloads/ggml-labrador13B-model-Q4_K_M.gguf models
  ```

## 🧑‍🏫 2. Model training

---

📋 **Note:** By default, the serve and generate commands assuming that we're using `ggml-malachite-7b-Q4_K_M.gguf` - this is a lightweight, fast model based on [Mistral](https://mistral.ai/news/announcing-mistral-7b/) that takes about ~45 min for synthetic data generation on an M1 / 16GB mac. If you have another quantized, gguf-format model you would like to use instead, there is a `--model` argument you can add to the **serve** and **generate** commands to indicate which model to use:

- **Serve** with the `--model` argument requires indicating the directory path to the model file, e.g.:
  `lab serve --model models/ggml-malachite-7b-Q4_K_M.gguf`

- **Generate** with the `--model` argument just requires the file name of the gguf model and assumes the model is located in the `models/` subdirectory of the root `cli/` git checkout directory, e.g.:
  `lab generate --model ggml-malachite-7b-Q4_K_M.gguf`

---

### Serve the model

- Serve the downloaded model locally via the **serve** command using the
  [llama.cpp framework](#TODO) and [llama-cpp-python](#TODO) (which provides
  Python bindings for llama.cpp):

  `lab serve`

  Once the model is being served and ready (takes less than 1 minute on an M1 mac), you'll see the following output:

  ```
  Starting server process
  After application startup complete see http://127.0.0.1:8000/docs for API.
  Press CTRL+C to shutdown server.
  INFO:     Started server process [4295]
  INFO:     Waiting for application startup.
  INFO:     Application startup complete.
  INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
  ```

### Test the model with chat before training

- Before you start tuning your model, test its output to your prompts as a baseline so you can better understand if your training was effective later. You can do this live via a chat interface with **chat**:

  `lab chat`

  Once you are in the chat interface, you can type `/h` for help, which will list out all of the chat commands.

### Generate a dataset

- Generate a synthetic dataset to enhance your example data set using the
  **generate** command, in another venv-activated terminal with the server running:

  ```
  lab generate
  ```

  📋 **Note:** This takes about **~45 minutes** to complete on an M1 mac with 16 GB RAM. The synthetic data set will be a file starting with the name `generated` ending in a `.json` file extension in the directory of your taxonomy. The file name includes model used and date time of generation.

  > Tip: If you want to pickup where you left off, copy a generated JSON file into a file named `regen.json`. `regen.json` will be picked up at the start of `lab generate` when available.

### Train the model

- Train the model on your synthetic data-enhanced dataset using **train**:

  `lab train {local path to gguf-format model} {path to root directorylocation of dataset}`

## 👩🏽‍🔬 3. Testing the fine-tuned model

### Serve the fine-tuned model

- First, stop the server you have running via `ctrl+c` in the terminal it is running in.
- Serve the fine-tuned model locally via the **serve** with the `--model` argument to specify your new model.

  `lab serve --model <New model name>`

### Try out the new model

- Try the fine-tuned model out live using a chat interface, and see if the results are better than the untrained version of the model with **chat**:

  `lab chat`

  Once you are in the chat interface, you can type `/h` for help, which will list out all of the chat commands.

### Run tests

- Run tests against the model via the **test** command:

  `lab test`

## 🎁 4. Submit your dataset!

Of course the final step is - if you've improved the model - to share your new dataset by submitting it! You'll submit it via a pull-request process, which
is documented in the [taxonomy respository](#TODO).

## Contributing

Check out our [contributing](CONTRIBUTING.md) guide to learn how to contribute to the Labrador CLI.

## Other stuffs

Hosted training environment:
[This jupyter notebook hosted on Google Colab](https://colab.research.google.com/drive/1qQr7X9Js6RTuXV12mRJtDHZU-bk4WgSU?usp=sharing)

[Converting a Model to GGUF and Quantizing](https://github.com/open-labrador/cli/docs/converting_GGUF.md)
