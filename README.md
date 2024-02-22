# CLI for TBD

This is the command-line interface for TBD...

## Pre-reqs

 * Python 3.9 or later
   * CLang distribution of Python: 15.0.0 (xcode)
* MacOS
  * 14.x with M1/M2/M3 (Metal/GPU)

## Quickstart

> TODO: run in a project folder using `pip this` from PyPI

To run from source, clone this repo.

```shell
cd <repo root>
python3 -m venv venv
pip install -r requirements.txt
```

Always be sure to activate the venv in your working shell.

```shell
source venv/bin/activate
```

Run the CLI:

```ShellSession
(venv) $ cli/lab.py
Usage: lab.py [OPTIONS] COMMAND [ARGS]...

  CLI for interacting with labrador

Options:
  --help  Show this message and exit.

Commands:
  chat      Run a chat using the modified model
  generate  Generates synthetic data to enhance your example data
  init      Initializes environment for labrador
  test      Perform rudimentary tests of the model
  train     Trains labrador model
```


## Inferencing Models

This repo provides instructions on how to inference models as follows:

- [Inferencing AI Models on a Mac Laptop](./mac_inference/README.md)
