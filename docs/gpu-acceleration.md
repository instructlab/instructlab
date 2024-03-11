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

Go to the project's Github to see the [supported backends](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends). There are several possible backends that may work on AMD; `CLBlast (OpenCL)` and `hipBLAS (ROCm)` have been tested to work. It may be worth installing others to see if they work for you, but your mileage may vary. Instructions for the tested backends are included below!

Whichever backend you choose, you'll see a `pip3 install` command. You'll want to add a few options to ensure it gets installed over the existing package: `--force-reinstall` and `--no-cache-dir`.

#### hipBLAS

If using hipBLAS you may need to install additional ROCm and hipBLAS Dependencies:
```
# Optionally enable repo.radeon.com repository, available through AMD documentation or Radeon Software for Linux for RHEL 9.3 at https://www.amd.com/en/support/linux-drivers
# ROCm Dependencies
sudo dnf install rocm-dev rocm-utils rocm-llvm rocminfo

# hipBLAS dependencies
sudo dnf install hipblas-devel hipblas rocblas-devel
```

With those dependencies installed, you should be able to install (and build) `llama-cpp-python`!

You can use `rocminfo | grep gfx` to find our GPU model to include in the build command - this may not be necessary in Fedora 40+ or ROCm 6.0+.  You should see something like the following if you have an AMD Integrated and Dedicated GPU:
```
$ rocminfo | grep gfx
  Name:                    gfx1100
      Name:                    amdgcn-amd-amdhsa--gfx1100
  Name:                    gfx1036
      Name:                    amdgcn-amd-amdhsa--gfx103
```

In this case, `gfx1100` is the model we're looking for (our dedicated GPU) so we'll include that in our build command as follows:
```
CMAKE_ARGS="-DLLAMA_HIPBLAS=on -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm -DAMDGPU_TARGETS=gfx1100" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Note:** This is explicitly forcing the build to use the ROCm compilers and prefix path for dependency resolution in the CMake build.  This works around an issue in the CMake and ROCm version in Fedora 39 and below and may be fixed in F40.

Once that package is installed, recompile `lab` with `pip3 install .`.  You also need to tell `HIP` which GPU to use - you can find this out via `rocminfo` although it is typically GPU 0.  To set which device is visible to HIP, we'll set `export HIP_VISIBLE_DEVICES=0` for GPU 0. Now you can skip to the `Testing` section.

#### CLBlast (OpenCL)

Your final command should look like so (this uses `CLBlast`):

```shell
CMAKE_ARGS="-DLLAMA_CLBLAST=on" pip3 install --force-reinstall --no-cache-dir llama-cpp-python
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
