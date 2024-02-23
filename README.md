# CLI for TBD

This is the command-line interface for TBD...

## Pre-reqs

 * Python 3.9 or later
   * CLang distribution of Python: 15.0.0 (xcode)
* MacOS
  * 14.x with M1/M2/M3 (Metal/GPU)

**Note:** The steps shown use [Python venv](https://docs.python.org/3/library/venv.html) for virtual environments. If you have used [pyenv](https://github.com/pyenv/pyenv), [Conda Miniforge](https://github.com/conda-forge/miniforge) or another tool for Python version management on your laptop, then use the virtual environment with that tool instead. Otherwise, you may have issues with packages installed but modules from that package not found as they are linked to you Python version management tool and not `venv`.

## Usage

To run from source, clone this repo.

```shell
cd <repo root>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Always be sure to activate the *venv* in your working shell.

```shell
source venv/bin/activate
```

Run CLI as follows:

```
$ python -m cli
Usage: python -m cli [OPTIONS] COMMAND [ARGS]...

  CLI for interacting with labrador

Options:
  --help  Show this message and exit.

Commands:
  chat      Run a chat using the modified model
  generate  Generates synthetic data to enhance your example data
  init      Initializes environment for labrador
  serve     Start a local server
  test      Perform rudimentary tests of the model
  train     Trains labrador model
```

The flow of commands (in order) are as follows:

1. init
2. download
3. serve
4. chat
5. generate
6. test
7. train

## Inferencing Models

This repo provides instructions on how to inference models as follows:

- [Inferencing AI Models on a Mac Laptop](./mac_inference/README.md)
