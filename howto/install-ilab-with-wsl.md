# Installing InstructLab on Windows using WSL (Windows Subsystem for Linux)

In this tutorial, I will be walking through the full process of running InstructLab on WSL, from installation of WSL to the initialization of InstructLab. My laptop specs are:

- CPU: Intel i7-10750H

- OS: Windows 10 Home

- RAM: 16GB

- GPU: Geforce RTX 3060 Laptop GPU (12GB VRAM)

Throughout this tutorial, we will be using the following software tools and packages:

- Windows PowerShell

- WSL

- Ubuntu

- `python3.10-venv`

- `cmake`

- `build-essential`


## Installing WSL

To install WSL, first open up Powershell. The following command installs the necessary features for WSL and the Ubuntu distro as default. The default distro can be changed with `wsl --list -d <DistributionName>`.
>NOTE: As of writing, Fedora is not supported with WSL (I could not figure out how to obtain the tar file for it) so I just proceeded with using Ubuntu.

```
wsl --install
```

WSL is installed! (Quite easy, wasn’t it?).  If you would still like to install WSL with Fedora, some links that may help are listed below:	
- https://dev.to/bowmanjd/install-fedora-on-windows-subsystem-for-linux-wsl-4b26
- https://learn.microsoft.com/en-us/windows/wsl/use-custom-distro

To run WSL, simply type the following command to set up the Linux environment in Powershell:

```
wsl
```

The expected output of this command should be similar to the following:

```
Welcome to Ubuntu 22.04.4 LTS (GNU/Linux 5.15.153.1-microsoft-standard-WSL2 x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/pro

 * Strictly confined Kubernetes makes edge and IoT secure. Learn how MicroK8s
   just raised the bar for easy, resilient and secure K8s cluster deployment.

   https://ubuntu.com/engage/secure-kubernetes-at-the-edge

This message is shown once a day. To disable it please create the
/home/user/.hushlogin file.
```

## Installing InstructLab 
From here, we can proceed with setting up InstructLab within our Linux environment. The following instructions are a mix of both the official InstructLab documentation as well as the WSL installation/setup process I went through.

Create a directory called `instructlab` to store the files InstructLab needs to run and cd into that directory:

```	
mkdir instructlab
cd instructlab
```

Next, update and upgrade the Linux environment to ensure all of your installed packages are up-to-date.

```
sudo apt update
sudo apt upgrade
```

For the sake of simplicity, we will be installing InstructLab using PyTorch without CUDA bindings or GPU acceleration. To do that, first go ahead and install the `python3.10-venv` package.

```
sudo apt install python3.10-venv
```

Then,

```
python3 -m venv --upgrade-deps venv
source venv/bin/activate
pip cache remove llama_cpp_python
```

Install `cmake`:

```
pip install cmake
```

And install `build-essential`:

```
sudo apt install build-essential
```

Finally, install the `instructlab` package. Note that we are making sure the build is done without Apple M-series GPU support because we are not using MacOS.

```
CMAKE_ARGS="-DLLAMA_METAL=off" pip install instructlab --extra-index-url=https://download.pytorch.org/whl/cpu
```

## Running InstructLab
At last, we can run `instructlab`. Verify the `ilab` CLI is running, then initialize it.

```
ilab
ilab config init
```

And there you go! You have InstructLab setup on your Windows machine (using Linux) through WSL! But it doesn’t stop here. You still need to download your model, generate your synthetic test data, train your model, test it, and even talk to it. If you end up not having enough VRAM to train your models locally (like me), you can use a cloud service like Google Colab. I highly recommend checking out the [official documentation](https://github.com/instructlab/instructlab/tree/main) for these next steps. I will also be releasing more tutorials as I go through my InstructLab journey, so stay tuned!
