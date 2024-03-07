# InstructLab 🥼 (`lab`)

## ❓ What is `lab`

`lab` is a Command-Line Interface (CLI) tool that allows you to:

1. Download a pre-trained LLM.
2. Chat with the LLM.

To add new knowledge and skills to the pre-trained LLM you have to add new information to the companion [taxonomy](https://github.com/instruct-lab/taxonomy.git) repository.
After that is done you can:

1. Use `lab` to generate new synthetic training data based on the changes to your local `taxonomy` repository.
2. Re-train the LLM that you initially downloaded with this new training data.
3. Chat with the re-trained LLM to see the results.

## 📋 Requirements

- **🍎 Apple M1/M2/M3 Mac or 🐧 Linux system** (tested on Fedora). We anticipate support for more operating systems in the future.
- The GNU C++ compiler
- 🐍 Python 3.9 or later, including the development headers.
- `gh` cli: Install [Github command cli](https://cli.github.com/) for downloading models from Github
- Approximately 10GB of free disk space to get through the `lab generate` step.  Approximately 60GB of free disk space to fully run the entire process locally on Apple hardware.

On Fedora Linux this means installing:
```
$ sudo dnf config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo
$ sudo yum install g++ gh python3 python3-devel
```

## 🧰 Installing `lab`

To start we'll create a new directory called `instruct-lab` to store the files that this CLI needs when it runs.

```
mkdir instruct-lab
cd instruct-lab
python3 -m venv venv
source venv/bin/activate
pip install git+ssh://git@github.com/instruct-lab/cli.git@stable
```
⏳ `pip install` may take some time, depending on your internet connection.

If `lab` is installed correctly, you should be able to test the lab command:

```
(venv) $ lab
lab [OPTIONS] COMMAND [ARGS]...

  CLI for interacting with InstructLab.

  If this is your first time running lab, it's best to start with `lab init`
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

**Every** `lab` command needs to be run from within your Python virtual environment:

```
source venv/bin/activate
```

## 🏗️ Initialize `lab`

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
Path to your model [models/ggml-merlinite-7b-0302-Q4_K_M.gguf]: <ENTER>
Generating `config.yaml` in the current directory...
Initialization completed successfully, you're ready to start using `lab`. Enjoy!
```

`lab` will use the default configuration file unless otherwise specified.
You can override this behavior for any `lab` command with the `--config` parameter.

## 📥 Download the model

Users should make sure they are logged in to their github accounts via the `gh` CLI and following the prompts/instructions:

```
gh auth login
```

**⁉️  Something not working?**: Please review [lab-troubleshoot.md](./lab-troubleshoot.md) for troubleshooting tips related to `gh`.

```
lab download
```

`lab download` will download a pre-trained model from GitHub and store it in a `models` directory:

```
(venv) $ lab download
Make sure the local environment has the `gh` cli: https://cli.github.com
Downloading models from https://github.com/instruct-lab/cli.git@v0.2.0 to models...
(venv) $ ls models
ggml-merlinite-7b-0302-Q4_K_M.gguf
```

⏳ This command can take 5+ minutes depending on your internet connection.

## 🍴 Serving the model

```
lab serve
```

Once the model is being served and ready, you'll see the following output:

```
(venv) $ lab serve
INFO 2024-03-02 02:21:11,352 lab.py:201 Using model 'models/ggml-merlinite-7b-0302-Q4_K_M.gguf' with -1 gpu-layers
Starting server process
After application startup complete see http://127.0.0.1:8000/docs for API.
Press CTRL+C to shutdown server.
```

## 📣 Chat with the model (optional)

Because you're serving the model in one terminal window, you'll likely have to create a new window and re-activate your Python virtual environment to run `lab chat`:
```
source venv/bin/activate
lab chat
```

Before you start adding new skills and knowledge to your knowledge, you can check out its baseline performance:

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

## 🎁 Contribute knowledge or compositional skills

Locally contribute new knowledge or compositional skills to your local [taxonomy](https://github.com/instruct-lab/taxonomy.git) repository.

Detailed contribution instructions can be found on the [taxonomy github](https://github.com/instruct-lab/taxonomy/blob/main/README.md).

## 📜 List your new knowledge

```
lab list
```

To ensure `lab` is registering your new knowledge you can run `lab list`.

Here is the expected result after adding the new compositional skill foo-lang:
```
(venv) $ lab list
compositional_skills/writing/freeform/foo-lang/foo-lang.yaml
```

## 🚀 Generate a synthetic dataset

```
lab generate
```

The next step is to generate a synthetic dataset based on your newly added knowledge set in the [taxonomy](https://github.com/instruct-lab/taxonomy.git) repository:

```
(venv) $ lab generate
INFO 2024-02-29 19:09:48,804 lab.py:250 Generating model 'ggml-merlinite-7b-0302-Q4_K_M' using 10 cpus,
taxonomy: '/home/username/instruct-lab/taxonomy' and seed 'seed_tasks.json'

0%|##########| 0/100 Cannot find prompt.txt. Using default prompt.
98%|##########| 98/100 INFO 2024-02-29 20:49:27,582 generate_data.py:428 Generation took 5978.78s
```

The synthetic data set will be three files in the `taxonomy` repository that are named like: `generated*.json`, `test*.jsonl`, and `train*.jsonl`:
```
(venv) $ ls taxonomy/
 CODE_OF_CONDUCT.md     CONTRIBUTING.md  'generated_ggml-merlinite-7b-0302-Q4_K_M_2024-02-29T19 09 48.json'   README.md                                                      'train_ggml-merlinite-7b-0302-Q4_K_M_2024-02-29T19 09 48.jsonl'
 compositional_skills   docs              MAINTAINERS.md                                                     'test_ggml-merlinite-7b-0302-Q4_K_M_2024-02-29T19 09 48.jsonl'
```

⏳ This can take over **1 hour+** to complete depending on your computing resources.

## 👩‍🏫 Train the model

### Traing the model locally on an M-series Mac

```
lab train
lab convert
```

**Every** `lab` command needs to be run from within your Python virtual environment:

### Traing the model in Co Lab

Train the model on your synthetic data-enhanced dataset by following the instructions in [Training](./notebooks/README.md)

⏳ This takes about **0.5-2.5 hours** to complete in the free tier of Google Colab.

After that's done, download the newly trained model from Google Colab and put it in the `models` directory created by the `lab download` command.

## 🍴 Serve the newly trained model

Stop the server you have running via `ctrl+c` in the terminal it is running in.
Serve the newly trained model locally via `lab serve` with the `--model` argument to specify your new model:

```
lab serve --model-path <New model name>
```

## 📣 Chat with the new model (not optional this time)

Try the fine-tuned model out live using the chat interface, and see if the results are better than the untrained version of the model with chat.

```
lab chat
```
## 🎁 Submit your new knowledge

Of course the final step is - if you've improved the model - to open up a a pull-request in the [taxonomy repository](https://github.com/instruct-lab/taxonomy).

## Contributing

Check out our [contributing](CONTRIBUTING/CONTRIBUTING.md) guide to learn how to contribute to the InstructLab CLI.
