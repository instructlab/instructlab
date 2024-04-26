# Welcome to InstructLab 🥼

This guide should teach you how to set up your development environment to start contributing to the `ilab` CLI tool.

**tl;dr** We're going to install `ilab` in a Python virtual environment like we did in the `README.md`. Instead of installing from GitHub, however, we'll clone the [`instructlab/instructlab`](https://github.com/instructlab/instructlab) repository from GitHub and install `ilab` from the cloned Python source code.

## Installing `ilab` from source

Here we install from the upstream repository, but you may want to fork the repository and replace the git clone URL below with the URL for your fork:

```ShellSession
git clone https://github.com/instructlab/instructlab.git
cd instructlab
python3 -m venv venv
source venv/bin/activate
pip3 install .
```

These are the steps that we're executing above, in plain language:

1. Clone the `instructlab/instructlab` repository from GitHub into the `instructlab` directory.
2. `cd` into that directory.
3. In the `instructlab` directory, create a new Python virtual environment.
4. Turn the virtual environment on.
5. Install `ilab` into your newly created Python virtual environment.

Success! 🌟 Now, when you run `ilab` commands, it's using the source code on your computer.

⚠️  **If `ilab` stops working:** Make sure to run `source venv/bin/activate` and are inside the venv (your terminal prompt should be prefixed with a `(venv)` to indicate this.

## Reinstalling `ilab` to see your changes

Changes to the `ilab` code in `instructlab/instructlab` **won't** automatically show up when you call `ilab` unless you reinstall the package from source.

Simply execute:

```shell
pip3 install .
```

This will reinstall `ilab` in your Python virtual environment with the changes that you made.
