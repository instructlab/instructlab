# InstructLab 🥼 (`lab`)

## 📖 Contents
- [❓What is `lab`](#-what-is-lab)
- [📋 Requirements](#-requirements)
- [✅ Getting started](#-getting-started)
  - [🧰 Installing`lab`](#-installing-lab)
  - [🏗️ Initialize `lab`](#%EF%B8%8F-initialize-lab)
  - [📥 Download the model](#-download-the-model)
  - [🍴 Serving the model](#-serving-the-model)
  - [📣 Chat with the model (Optional)](#-chat-with-the-model-optional)
- [💻 Creating new knowledge and training the model](#-creating-new-knowledge-and-training-the-model)
  - [🎁 Contribute knowledge or compositional skills](#-contribute-knowledge-or-compositional-skills)
  - [📜 List your new knowledge](#-list-your-new-knowledge)
  - [🚀 Generate a synthetic dataset](#-generate-a-synthetic-dataset)
  - [👩‍🏫 Train the model](#-train-the-model) 
  - [🍴 Serve the newly trained model](#-serve-the-newly-trained-model)
  - [📣 Chat with the new model (not optional this time)](#-chat-with-the-new-model-not-optional-this-time)
- [🎁 Submit your new knowledge](#-submit-your-new-knowledge)
- [📬 Contributing to Instruct-Lab CLI](#-contributing)

## ❓ What is `lab`

`lab` is a Command-Line Interface (CLI) tool that allows you to:

1. Download a pre-trained LLM (Large Laungage Model).
2. Chat with the LLM.

To add new knowledge and skills to the pre-trained LLM you have to add new information to the companion [taxonomy](https://github.com/instruct-lab/taxonomy.git) repository.
After that is done, you can:

1. Use `lab` to generate new synthetic training data based on the changes in your local `taxonomy` repository.
2. Re-train the LLM with the new training data.
3. Chat with the re-trained LLM to see the results.

## 📋 Requirements

- **🍎 Apple M1/M2/M3 Mac or 🐧 Linux system** (tested on Fedora). We anticipate support for more operating systems in the future.
- The GNU C++ compiler
- 🐍 Python 3.9 or later, including the development headers.
- Approximately 10GB of free disk space to get through the `lab generate` step.  Approximately 60GB of free disk space to fully run the entire process locally on Apple hardware.

On Fedora Linux this means installing:
```
$ sudo yum install g++ python3 python3-devel
```

## ✅ Getting started 
### 🧰 Installing `lab`

To start, create a new directory called `instruct-lab` to store the files that the `lab` CLI needs when it runs.

```
mkdir instruct-lab
cd instruct-lab
python3 -m venv venv
source venv/bin/activate
pip install git+ssh://git@github.com/instruct-lab/cli.git@stable
```
> **NOTE**: ⏳ `pip install` may take some time, depending on your internet connection.

If `lab` is installed correctly, you can test the lab command:

```
(venv) $ lab
lab [OPTIONS] COMMAND [ARGS]...

  CLI for interacting with InstructLab.

  If this is your first time running `lab`, it's best to start with `lab init`
  to create the environment

Options:
  --config PATH  Path to a configuration file.  [default: config.yaml]
  --help         Show this message and exit.

Commands:
  chat      Run a chat using the modified model
  download  Download the model(s) to train
  generate  Generates synthetic data to enhance your example data
  init      Initializes environment for InstructLab
  list      Lists taxonomy files that have changed since last commit
  serve     Start a local server
  test      Perform rudimentary tests of the model
  train     Trains model
```

**Every** `lab` command needs to be run from within your Python virtual environment. To enter the Python environment, run the following command: 

```
source venv/bin/activate
```

### 🏗️ Initialize `lab`

```
lab init
```
Initializing `lab` will: 
1. Add a new, default `config.yaml` file. 
2. Clone the `git@github.com:instruct-lab/taxonomy.git` repository into the current directory.

```
(venv) $ lab init
Welcome to InstructLab CLI. This guide will help you to setup your environment.
Please provide the following values to initiate the environment:
Path to taxonomy repo [taxonomy]: <ENTER>
`taxonomy` seems to not exists or is empty. Should I clone git@github.com:instruct-lab/taxonomy.git for you? [y/N]: y
Cloning git@github.com:instruct-lab/taxonomy.git...
Generating `config.yaml` in the current directory...
Initialization completed successfully, you're ready to start using `lab`. Enjoy!
```

`lab` will use the default configuration file unless otherwise specified.
You can override this behavior for any `lab` command with the `--config` parameter.

### 📥 Download the model

```
lab download
```

`lab download` will download a pre-trained model from HuggingFace and store it in a `models` directory:

```
(venv) $ lab download
Downloading model from ibm/merlinite-7b-GGUF@main to models...
(venv) $ ls models
merlinite-7b-Q4_K_M.gguf
```

> **NOTE** ⏳ This command can take few minutes or immediately depending on your internet connection or model is cached.

### 🍴 Serving the model

```
lab serve
```

Once the model is served and ready, you'll see the following output:

```
(venv) $ lab serve
INFO 2024-03-02 02:21:11,352 lab.py:201 Using model 'models/ggml-merlinite-7b-0302-Q4_K_M.gguf' with -1 gpu-layers
Starting server process
After application startup complete see http://127.0.0.1:8000/docs for API.
Press CTRL+C to shutdown server.
```

### 📣 Chat with the model (Optional)

Because you're serving the model in one terminal window, you likely have to create a new window and re-activate your Python virtual environment to run `lab chat`:
```
source venv/bin/activate
lab chat
```

Before you start adding new skills and knowledge to your model, you can check out its baseline performance:

```
(venv) $ lab chat
╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────── system ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Welcome to Chat CLI w/ GGML-MERLINITE-7B-0302-Q4_K_M (type /h for help)                                                                                                                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
>>> what is the capital of canada                                                                                                                                                                                                 [S][default]
╭────────────────────────────────────────────────────────────────────────────────────────────────────── ggml-merlinite-7b-0302-Q4_K_M ───────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ The capital city of Canada is Ottawa. It is located in the province of Ontario, on the southern banks of the Ottawa River in the eastern portion of southern Ontario. The city serves as the political center for Canada, as it is home to │
│ Parliament Hill, which houses the House of Commons, Senate, Supreme Court, and Cabinet of Canada. Ottawa has a rich history and cultural significance, making it an essential part of Canada's identity.                                   │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── elapsed 12.008 seconds ─╯
>>>                                                                                                                                                                                                                               [S][default]
```

## 💻 Creating new knowledge and training the model
### 🎁 Contribute knowledge or compositional skills

Locally contribute new knowledge or compositional skills to your local [taxonomy](https://github.com/instruct-lab/taxonomy.git) repository.

Detailed contribution instructions can be found on the [taxonomy github](https://github.com/instruct-lab/taxonomy/blob/main/README.md).

### 📜 List your new knowledge

```
lab list
```

To ensure `lab` is registering your new knowledge, you can run `lab list`.

The following is the expected result after adding the new compositional skill foo-lang:
```
(venv) $ lab list
compositional_skills/writing/freeform/foo-lang/foo-lang.yaml
```

### 🚀 Generate a synthetic dataset

```
lab generate
```
The next step is to generate a synthetic dataset based on your newly added knowledge set in the [taxonomy](https://github.com/instruct-lab/taxonomy.git) repository.

> **NOTE:** ⏳ This can take over **1 hour+** to complete depending on your computing resources.

> **IMPORTANT:** Do not use `lab generate` and `lab chat` in parallel, it could cause the model to crash ([#346](https://github.com/instruct-lab/cli/issues/346)).

```
(venv) $ lab generate
INFO 2024-02-29 19:09:48,804 lab.py:250 Generating model 'ggml-merlinite-7b-0302-Q4_K_M' using 10 cpus,
taxonomy: '/home/username/instruct-lab/taxonomy' and seed 'seed_tasks.json'

0%|##########| 0/100 Cannot find prompt.txt. Using default prompt.
98%|##########| 98/100 INFO 2024-02-29 20:49:27,582 generate_data.py:428 Generation took 5978.78s
```

The synthetic data set will be three files in the newly created `generated` directory that are named like: `generated*.json`, `test*.jsonl`, and `train*.jsonl`:
```
(venv) $ ls generated/
 'generated_ggml-malachite-7b-0226-Q4_K_M_2024-02-29T19 09 48.json'   'train_ggml-malachite-7b-0226-Q4_K_M_2024-02-29T19 09 48.jsonl'
 'test_ggml-malachite-7b-0226-Q4_K_M_2024-02-29T19 09 48.jsonl'
```

It is also possible to run the generate step against a different model via an
OpenAI compatible API. For example, the one spawned by `lab serve` or any remote or locally hosted LLM (e.g. via [ollama](ollama.ai/), [LM Studio](https://lmstudio.ai), etc.)

```
lab generate --endpoint-url http://localhost:8000/v1
```

### 👩‍🏫 Train the model

There are currently three options to train the model on your synthetic data-enhanced dataset.

**Every** `lab` command needs to be run from within your Python virtual environment

#### Train the model locally on Linux

```
lab train
```

> **NOTE:** ⏳ This step can take **several hours** to complete depending on your computing resources.

`lab train` outputs a brand new model that can be served in the `models` directory called `ggml-model-f16.gguf`
```
(venv) $ ls models
ggml-merlinite-7b-0302-Q4_K_M.gguf  ggml-model-f16.gguf
```

#### Train the model locally on an M-series Mac:

To train the model locally on your M-Series Mac is as easy as running:
```
lab train
```

⏳ This process will take a little while to complete (time can vary based on hardware
and output of `lab generate` but on the order of 30 minutes to two hours)

#### Train the model in Colab:

Follow the instructions in [Training](./notebooks/README.md).

> **NOTE:** ⏳ This takes about **0.5-2.5 hours** to complete in the free tier of Google Colab.

After that's done, you can play with your model directly in the Colab notebook.
The model itself is for testing your taxonomy and is not a required artifact for
a PR or any further task.

### 🍴 Serve the newly trained model

Stop the server you have running via `ctrl+c` in the terminal it is running in.

Before serving the newly trained model you will need to convert it to work with 
the `lab` cli. 

```
lab convert
```

Serve the newly trained model locally via `lab serve` with the `--model` 
argument to specify your new model:

```
lab serve --model-path <New model name>
```

But which model to serve? After running the `lab convert` command, a few files
and directories are generated. The one you will want to serve will end in `.gguf`
and will exist in a directory with the suffix `fused-pt`. For example:
`ibm-merlinite-7b-mlx-q-fused-pt/ggml-model-Q4_K_M.gguf`

## 📣 Chat with the new model (not optional this time)

Try the fine-tuned model out live using the chat interface, and see if the results are better than the untrained version of the model with chat.

```
lab chat -m <New model name>
```
## 🎁 Submit your new knowledge

Of course the final step is, if you've improved the model, to open up a a pull-request in the [taxonomy repository](https://github.com/instruct-lab/taxonomy) that includes the `qna.yaml` files with your improved data. 

## 📬 Contributing

Check out our [contributing](CONTRIBUTING/CONTRIBUTING.md) guide to learn how to contribute to the InstructLab CLI.
