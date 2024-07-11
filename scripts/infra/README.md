# cloud-instance.sh

This script is a helper for launching cloud instances and getting instructlab setup for development

## Prereqs

You'll need to setup the respective cloud CLI before using [cloud-instance.sh](cloud-instance.sh)

### AWS

- You can ensure you have all the necessary local system packages with `local-setup.sh` (see example below)
  - This includes [aws](https://aws.amazon.com/cli/) (the AWS CLI)
- A keypair created within AWS. Config defaults to `whoami` for key name.

```bash
# Ensure proper system packages are installed if your local system is Red Hat-based
# If you are running a different OS you'll need to install these packages manually
./local-setup.sh rh ensure-aws-system-pkgs
# If you are using SAML auth with the AWS CLI, you can install packages for that with the '-s` flag
./local-setup.sh rh ensure-aws-system-pkgs -s
```

### IBM

- [ibmcloud](https://cloud.ibm.com/docs/cli) (the IBM Cloud CLI)
- An sshkey created within IBM Cloud. Config defaults to `whoami` for key name.
- A floating IP created within IBM Cloud. Config defaults to `whoami` for floating IP name.

```bash
ibmcloud api cloud.ibm.com
ibmcloud plugin install is
```

## Config

You'll need to specify the `cloud-instance.sh` config before getting started. [instructlab-cloud-config-example](instructlab-cloud-config-example) is provided as a template. By default, you'll need to place your config in `~/.instructlab/cloud-config`, or you can specify the location as the environment variable `INSTRUCTLAB_CLOUD_CONFIG`

To place the config in its default location

```bash
mkdir ~/.instructlab
cp instructlab-cloud-config-example ~/.instructlab/cloud-config
```

Or, to setup the environment variable

```bash
cp instructlab-cloud-config-example <location of your choice>
export INSTRUCTLAB_CLOUD_CONFIG=<location of your choice>
```

Either way, you'll need to edit the config to reference your cloud environment(s).

## Usage

`scripts/infra/cloud-instance.sh <cloud-type> <command> [options]`

Example:

```bash
# Launch a new instance with your `EC2_INSTANCE_NAME` and `EC2_INSTANCE_TYPE` specified
# in config with the instance type including Nvidia GPU(s)
scripts/infra/cloud-instance.sh ec2 launch
# Clone instructlab onto the instance and setup the development environment
scripts/infra/cloud-instance.sh ec2 setup-rh-devenv
# Install nvidia drivers and reboot
scripts/infra/cloud-instance.sh ec2 install-rh-nvidia-drivers
scripts/infra/cloud-instance.sh ec2 ssh sudo reboot
# Install instructlab
scripts/infra/cloud-instance.sh ec2 pip-install-with-nvidia
# ssh to the instance
scripts/infra/cloud-instance.sh ec2 ssh
# Run commands on the instance through ssh
scripts/infra/cloud-instance.sh ec2 ssh ls -la
scripts/infra/cloud-instance.sh ec2 ssh "source instructlab/venv/bin/activate && ilab system info"
# Sync your local git repo to the repo on the instance
scripts/infra/cloud-instance.sh ec2 sync
# Make changes in your local git without committing
# Sync your changes to the remote instance with a temporary commit
scripts/infra/cloud-instance.sh ec2 sync -c
# When you're done, stop the instance
scripts/infra/cloud-instance.sh ec2 stop
# While stopped and not yet pruned from your account, you can restart it if needed
scripts/infra/cloud-instance.sh ec2 start
# When you are done, you can delete the instance
scripts/infra/cloud-instance.sh ec2 terminate
```