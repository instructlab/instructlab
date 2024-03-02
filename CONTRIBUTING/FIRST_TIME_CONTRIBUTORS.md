# Welcome to Labrador 🐶!

This guide should teach you how to set up your development environment to start contributing to the `lab` CLI tool.

**tl;dr** We're going to install `lab` in a Python virtual environment like we did in the README.md. Instead of installing from GitHub, however, we'll clone the [`open-labrador/cli`](https://github.com/open-labrador/cli) repository from GitHub and install `lab` from the cloned Python source code.

# Installing `lab` from source

Here we install from the upstream repo, but you may want to fork the repo and replace the git clone URL below with the URL for your fork:

```ShellSession
git clone https://github.com/open-labrador/cli.git
cd cli
python3 -m venv venv
source venv/bin/activate
pip install .
```

These are the steps that we're executing above, in plain language:

1. Clone the `open-labrador/cli` repository from GitHub into the `cli` directory.
2. `cd` into that directory.
3. In the `cli` directory, create a new Python virtual environment.
4. Turn the virtual environment on.
5. Install `lab`, from the `cli/cli/` directory, into your newly created Python virtual environment.

Success! 🌟 Now, when you run `lab` commands, it's using the source code on your computer.

# Reinstalling `lab` to see your changes

Changes to the `lab` code in `cli/cli` **won't** automatically show up when you call `lab` unless you reinstall the package from source.

Simply execute:

```
pip3 install .
```

This will reinstall `lab` in your Python virtual environment with the changes that you made.
