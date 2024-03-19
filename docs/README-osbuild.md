# Specialized Fedora for Nvidia PCs

The use case here is, you have a **PC with an Nvidia graphics card** available and don't want
to mess with your current installation (Windows?) just for AI testing and playing around with models.

**NOTE** that your graphics card has to have
 * enough RAM to hold the AI model and has to
 * support "CUDA"

See https://developer.nvidia.com/cuda-gpus to check if your card is compatible

Now you can install [osbuild](https://osbuild.org) on any Fedora or Red Hat Enterprise Linux
```
dnf install osbuild
```

and build your **own custom operating system** to be copied onto an external USB drive.
This is then will be a **portable Fedora**, able to run AI workloads but not
touching your current installation.

You can even disconnect your current hard drive before booting
for some extra peace of mind ☺

## Adapt the image definition

Open `AIServer-nvidia.toml` in your favorite text editor and change at least
the SSH keys and password entries.

Your public SSH key might show up when you execute:
```
cat ~/.ssh/id_rsa.pub
```

and an encrypted password for the user `ollama`
can be generated like so

```
python -c "import bcrypt, getpass; print(bcrypt.hashpw(getpass.getpass('Enter a password: ').encode(), bcrypt.gensalt()).decode())"
```

You should change the password of `ollama` because the current one (being "password") is really bad 😉

## Build the image

Tell your osbuild installation where the custom repositories are:

(The files are next to this `README-osbuild.md` in this
repository and need to be on your computer)

```
composer-cli sources add nvidia-cuda.toml
composer-cli sources add nvidia-source.toml
composer-cli sources add rpmfusion.toml
```

Add the blueprint and build it

```
composer-cli blueprints push AIServer-nvidia.toml
composer-cli compose start AIServer-nvidia minimal-raw
```

## Dump the image on your USB drive

Connect a USB drive **WHICH WILL BE FULLY WIPED**!

Recommended is a drive with at least 64GB and no USB stick as
they are really slow for this type of use case.


The "IMAGE ID" is the one written after `composer-cli compose start AIServer-nvidia minimal-raw`
or you can check the IDs with `composer-cli compose status`

Check e.g. with `lsblk` where your USB drive is.
Please be careful to take the correct device and replace `/dev/sd_SOMETHING` in the command below.
WARNING! This will erase all data on the disk given!

```
export IMAGEID=YOUR_IMAGE_ID_HERE
composer-cli compose image $IMAGEID
export DISK=/dev/sd_SOMETHING
xz -dc $IMAGEID-raw.img.xz |sudo dd of=$DISK bs=100M status=progress && echo "Waiting for full data sync..." && sync
```

## Boot the image

Now just connect the USB drive to your PC with the Nvidia card and assure it starts from the USB drive.

You should be able to connect to the PC via SSH (the IP should be visible on the monitor)
with your password or ssh key.
After login, execute the scripts there in sequence.
At first boot some more ~4GB of data need to be downloaded.
Then run the scripts and follow their instructions
(currently only two)

```
cd /root
ls -la
```

or connect to `http://THE_IP:8080` and play around with "ollama-webui"

https://docs.openwebui.com/

HINT: On first startup you just "register" any user - no real E-Mail needed. The WebUI just wants
a username and password for local access (nothing is sent to the internet)
Finally download a model in the settings!

Everything will be local on your disk anyway!


## NOTES

CUDA installation seems to be strange so if you need llama-cpp-python you might want to use this command

```
CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

