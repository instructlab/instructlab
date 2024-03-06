# Lab Troubleshooting

This document is for commonly found problems and their solutions when using `lab`.

## `lab` troubleshooting

### Long `lab generate` on MacOS issue
If you notice `lab generate` being very slow for you, several hours or more.
Check [this discussion](https://github.com/ggerganov/llama.cpp/discussions/2182#discussioncomment-7698315)
which is suggesting to tweak the GPU limit, which MacOS assigns. By default it's
around 60%-70% of your total RAM available, which is expressed as 0:
```
$ sudo sysctl iogpu.wired_limit_mb
iogpu.wired_limit_mb = 0
```
You can set it to any number, although it's advisable to leave 4-6GB for MacOS.

On a M1 with 16GB, it was tested that `lab generate` with that limit bumped to
12GB was able to finish in less than an hour.
```
sudo sysctl iogpu.wired_limit_mb=12288
```
Once done, make sure to reset the limit back to 0, which resets it to default.

**Note:** This value will reset to defaults after machine reboot.

## `gh` troubleshooting

This page has some troubleshooting techniques if you hit a github cli `gh` error described below.

### gh error during `lab download`
Some people are hitting a `gh` error while running the `lab download` step.

If you see this `error invoking gh command` there is a `quick fix` and `longer fix`

```
(venv) $ lab download
Make sure the local environment has the `gh` cli: https://cli.github.com
Downloading models from https://github.com/instruct-lab/cli.git@v0.2.0 to models...
Downloading models failed with the following error: error invoking `gh` command: Command '['gh', 'release', 'download', 'v0.2.0',
 '--repo', 'https://github.com/instruct-lab/cli.git', '--dir', 'models', '--pattern', 'ggml-merlinite-7b-0302-Q4_K_M.*']' returned non-zero exit status 4.
it is time to look at your gh settings - and make sure you can run 'gh auth login'
```

#### Quick fix
Run `gh auth login`

```
gh auth login
```

**Note:** On macOS, users can add their SSH keys to their apple-keychain by running:
```
ssh-add --apple-use-keychain ~/.ssh/[your-private-key]
```

#### Longer fix
If after `gh auth login` you are still hitting issue(s) try the following:

Up front
- git uses your SSH public key to allow https git clone (see below)
- gh uses a token

If you need to check/create a new `gh` token
- log in to your `https://github.com/` account
- find `developer settings` bottom of left hand column (sometimes tough to find)
- Go to settings -> Developer Settings -> new personal access (classic) token

Create new token checking off:
- [x] repo
- admin:org [x] read:org

Copy your new token to `mytoken.txt` which gets used below.

More info on `gh_auth_login` is at [gh_auth_login](https://cli.github.com/manual/gh_auth_login)
```
Authenticate against github.com by reading the token from a file
gh auth login --with-token < mytoken.txt
```

#### Raw steps

**NOTE:** logout may not be necessary

```
$ gh auth logout
```

Always refer to [README.md](README.md) for most/latest commands used during Installing lab:

```
gh auth login --with-token < ~/Documents/mytoken.txt
git clone https://github.com/instruct-lab/cli.git
cd cli
mkdir instruct-lab
cd instruct-lab
python3 -m venv venv
source venv/bin/activate
pip install git+ssh://git@github.com/instruct-lab/cli.git
pip install --upgrade pip
```

```
$ lab init
Welcome to InstructLab CLI. This guide will help you to setup your environment.
Please provide the following values to initiate the environment:
Path to taxonomy repo [taxonomy]: <ENTER>
`taxonomy` seems to not exists or is empty. Should I clone git@github.com:instruct-lab/taxonomy.git for you? [y/N]: y
Cloning git@github.com:instruct-lab/taxonomy.git...
Path to your model [models/ggml-merlinite-7b-0302-Q4_K_M.gguf]: <ENTER>
Generating `config.yaml` in the current directory...
Initialization completed successfully, you're ready to start using `lab`. Enjoy!
```

```
$ lab download
Make sure the local environment has the `gh` cli: https://cli.github.com
Downloading models from https://github.com/instruct-lab/cli.git@v0.2.0 to models...
```

Ensure a model is downloaded.

```
$ ls models
ggml-merlinite-7b-0302-Q4_K_M.gguf
```
