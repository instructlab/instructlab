# 🏎️ Making `lab` go fast

By default, `lab` will attempt to use your GPU for inference and synthesis. This works on a wide variety of common systems, but less-common configurations may require some additional tinkering to get it enabled. This document aims to describe how you can GPU-accelerate `lab` on a variety of different environments.

`lab` relies on two Python packages that can be GPU accelerated: `torch` and `llama-cpp-python`. In short, you'll need to replace the default versions of these packages with versions that have been compiled for GPU-specific support, recompile `lab`, then run it.

### Python 3.11 (Linux only)

Unfortunately, at the time of writing, `torch` does not have GPU-specific support for the latest Python (3.12), so if you're on Linux, it's recommended to set up a Python 3.11-specific `venv` and install `lab` to that to minimize issues. (MacOS ships Python 3.9, so this step shouldn't be necessary.) Here's how to do that on Fedora with `dnf`:

```ShellSession
# Install python3.11
sudo dnf install python3.11

# Remove old venv (if it exists)
rm -r venv

# Create and activate new Python 3.11 venv
python3.11 -m venv venv
source venv/bin/activate

# Install lab (assumes a locally-cloned repo)
pip3 install .
```

With Python 3.11 installed, it's time to replace some packages!

### Nvidia/CUDA

`torch` should already ship with CUDA support, so you only have to replace `llama-cpp-python`.

Go to the project's Github to see the [supported backends](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends). Find the `cuBLAS (CUDA)` backend. You'll see a `pip3 install` command. You'll want to add a few options to ensure it gets installed over the existing package: `--force-reinstall` and `--no-cache-dir`. Your final command should look like so:

```shell
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip3 install --force-reinstall --no-cache-dir llama-cpp-python
```

Once that package is installed, recompile `lab` with `pip3 install .` and skip to the `Testing` section.

### AMD/ROCm

`torch` does not yet ship with AMD ROCm support, so you'll need to install a version compiled with support for it.

Visit [Pytorch's "Get Started Locally" page](https://pytorch.org/get-started/locally/) and use the matrix installer tool to find the ROCm package. `Stable, Linux, Pip, Python, ROCm 5.7` in the matrix installer spits out the following command:

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

You don't need `torchvision` or `torchaudio`, so get rid of those. You also want to make _very_ sure you're installing the right package, and not the old one that doesn't have GPU support, so you should add these options: `--force-reinstall` and `--no-cache-dir`. Your command should look like below. Run it to install the new version of `torch`.

```shell
pip3 install torch --force-reinstall --no-cache-dir --index-url https://download.pytorch.org/whl/rocm5.7
```

With that done, it's time to move on to `llama-cpp-python`.

Go to the project's Github to see the [supported backends](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends). There are several possible backends that may work on AMD; `CLBlast (OpenCL)` has been tested to work. It may be worth installing others to see if they work for you, but your mileage may vary.

Whichever backend you choose, you'll see a `pip3 install` command. You'll want to add a few options to ensure it gets installed over the existing package: `--force-reinstall` and `--no-cache-dir`. Your final command should look like so (this uses `CLBlast`):

```shell
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip3 install --force-reinstall --no-cache-dir llama-cpp-python
```

Once that package is installed, recompile `lab` with `pip3 install .` and skip to the `Testing` section.

### Metal/Apple Silicon

The `lab` default installation should have Metal support by default. If that isn't the case, these steps might help to enable it.

`torch` should already ship with Metal support, so you only have to replace `llama-cpp-python`. Go to the project's Github to see the [supported backends](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends). Find the `Metal` backend. You'll see a `pip3 install` command. You'll want to add a few options to ensure it gets installed over the existing package: `--force-reinstall` and `--no-cache-dir`. Your final command should look like so:

```shell
CMAKE_ARGS="-DLLAMA_METAL=on" pip3 install --force-reinstall --no-cache-dir llama-cpp-python
```

Once that package is installed, recompile `lab` with `pip3 install .` and skip to the `Testing` section.

### Testing

Test your changes by chatting to the LLM. Run `lab serve` and `lab chat` and chat to the LLM. If you notice significantly faster inference, congratulations! You've enabled GPU acceleration. You should also notice that the `lab generate` step will take significantly less time.
