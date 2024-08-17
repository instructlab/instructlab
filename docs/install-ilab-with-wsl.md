# Installing InstructLab on Windows using WSL (Windows Subsystem for Linux)

This tutorial describes the full process of running InstructLab on WSL, from installation of WSL to the initialization of InstructLab.

This tutorial will use the following software tools and packages:

- Windows PowerShell

- WSL

- Ubuntu

- `python3.11`

- `python3.11-dev`

- `build-essential`


## Installing WSL

To install WSL, first open up Powershell. The following command installs the necessary features for WSL and the Ubuntu distro as default. The default distro can be changed with `wsl --list -d <DistributionName>`.
>NOTE: As of writing, Fedora is not supported with WSL so the tutorial proceeds with using Ubuntu.

```
wsl --install
```

WSL is installed.

To run WSL, simply type the following command to set up the Linux environment in Powershell:

```
wsl
```

## Installing InstructLab 
From here, proceed with setting up InstructLab within the Linux environment. The following instructions are a mix of both the official InstructLab documentation as well as the WSL installation/setup process.

First, update and upgrade the Linux environment to ensure all of your installed packages are up-to-date.

```
sudo apt update
sudo apt upgrade
```

And install `build-essential`:

```
sudo apt install build-essential
```

Install the requisite packages:
>NOTE: Since WSL uses Ubuntu as the default distro, use `apt install` instead of `dnf install` as is stated in the official InstructLab documentation.

```
sudo apt install gcc g++ make git python3.11 python3.11-dev
```

Create a directory called `instructlab` to store the files InstructLab needs to run and cd into that directory:

```	
mkdir instructlab
cd instructlab
```

Next, 

For the sake of simplicity, install InstructLab using PyTorch without CUDA bindings or GPU acceleration. Note that we are making sure the build is done without Apple M-series GPU support because we are not using MacOS.

```
python3 -m venv --upgrade-deps venv
source venv/bin/activate
CMAKE_ARGS="-DLLAMA_METAL=off" pip install instructlab
```

## Running InstructLab
At last, we can run `instructlab`. Verify the `ilab` CLI is running, then initialize it.

```
ilab
ilab config init
```

And there you go! You have InstructLab setup on your Windows machine (using Linux) through WSL! But it doesn’t stop here. You still need to download your model, generate your synthetic test data, train your model, test it, and even talk to it. If you end up not having enough VRAM to train your models locally (like me), you can use a cloud service like Google Colab. I highly recommend checking out the [official documentation](https://github.com/instructlab/instructlab/tree/main) for these next steps. I will also be releasing more tutorials as I go through my InstructLab journey, so stay tuned!
