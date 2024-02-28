# Standard
import os
import re
import subprocess
import textwrap

# Third Party
import click


def download_model(
    gh_repo="https://github.com/open-labrador/cli.git",
    gh_release="latest",
    model_dir=".",
    pattern="",
):
    """
    Download and combine the model file from a GitHub repository source.

    Parameters:
    - gh_repo (str): The URL of the GitHub repository containing the model.
        Default: Open Labrador CLI repository.
    - gh_release (str): The GitHub release version of the model to download.
        Default is 'latest'.
    - model_dir(str): The local directory to download the model files into
    - pattern(str): Download only assets that match a glob pattern

    Returns:
    - None
    """

    model_file_split_keyword = ".split."

    click.secho(
        '\nMake sure the local environment has the "gh" cli. https://cli.github.com',
        fg="blue",
    )
    click.echo(
        "\nDownloading Models from %s with version %s to local directory %s ...\n"
        % (gh_repo, gh_release, model_dir)
    )

    # Download GitHub release
    download_commands = [
        "gh",
        "release",
        "download",
        gh_release,
        "--repo",
        gh_repo,
        "--dir",
        model_dir,
    ]
    if pattern != "":
        download_commands.extend(["--pattern", pattern])
    if gh_release == "latest":
        download_commands.pop(3)  # remove gh_release arg to download the latest version
        if (
            pattern == ""
        ):  # Latest release needs to specify the pattern argument to download files
            download_commands.extend(["--pattern", "*"])
    try:
        create_subprocess(download_commands)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        click.echo("%s" % e)
        click.echo("\nAn error occurred with gh. Check the traceback for details.\n")

    # Get the list of local files
    ls_commands = ["ls", model_dir]
    ls_result = create_subprocess(ls_commands)
    file_list = ls_result.stdout.decode("utf-8").split("\n")
    file_list = [os.path.join(model_dir, f) for f in file_list]

    splitted_models = {}

    # Use Regex to capture model and files names that need to combine into one file.
    for file in file_list:
        if model_file_split_keyword in file:
            model_name_regex = "".join(
                ["([^ \t\n,]+)", model_file_split_keyword, "[^ \t\n,]+"]
            )
            regex_result = re.findall(model_name_regex, file)
            if regex_result:
                model_name = regex_result[0]
                if splitted_models.get(model_name):
                    splitted_models[model_name].append(file)
                else:
                    splitted_models[model_name] = [file]

    # Use the model and file names we captured to minimize the number of bash subprocess creations.
    combined_model_list = []
    for key, value in splitted_models.items():
        cat_commands = ["cat"]
        splitted_model_files = list(value)
        cat_commands.extend(splitted_model_files)
        cat_result = create_subprocess(cat_commands)
        if cat_result.stdout not in (None, b""):
            with open(key, "wb") as model_file:
                model_file.write(cat_result.stdout)
            rm_commands = ["rm"]
            rm_commands.extend(splitted_model_files)
            create_subprocess(rm_commands)
            combined_model_list.append(key)

    click.echo("\nDownload Completed.")
    if combined_model_list:
        click.echo("\nList of combined models: ")
        for model_name in combined_model_list:
            click.echo("%s" % model_name)


def clone_taxonomy(
    gh_repo="https://github.com/open-labrador/taxonomy.git",
    gh_branch="main",
    git_filter_spec="",
    dir='taxonomy'
):
    """
    Clone the taxonomy repository from a Git repository source.

    Parameters:
    - gh_repo (str): The URL of the taxonomy Git repository.
        Default is the Open Labrador taxonomy repository.
    - gh_branch (str): The GitHub branch of the taxonomy repository. Default is main
    - git_filter_spec(str): Optional path to the git filter spec for git partial clone
    - dir(str): Optional path to clone the repo into. Defaults to taxonomy in your current working directory.

    Returns:
    - None
    """

    click.echo('\nCloning repository %s with branch "%s" ...' % (gh_repo, gh_branch))

    # Clone taxonomy repo
    git_clone_commands = ['git', 'clone', gh_repo, dir]
    if git_filter_spec != "" and os.path.exists(git_filter_spec):
        # TODO: Add gitfilterspec to sparse clone GitHub repo
        git_filter_arg = "".join(
            ["--filter=sparse:oid=", gh_branch, ":", git_filter_spec]
        )
        git_sparse_clone_flags = ["--sparse", git_filter_arg]
        git_clone_commands.extend(git_sparse_clone_flags)
    else:
        git_clone_commands.extend(["--branch", gh_branch])

    result = create_subprocess(git_clone_commands)
    if result.stderr:
        click.echo("\n%s" % result.stderr.decode("utf-8"))
    click.echo("Git clone completed.")


def create_config_file(config_file_name="./config.yml"):
    # pylint: disable=line-too-long
    """
    Create default config file.
    TODO: Remove this function after config class is updated.

    Parameters:
    - config_path (str): Path to create the default config.yml

    Returns:
    - None
    """

    config_yml_txt = textwrap.dedent(
        """
    # Copyright The Authors
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    chat:
      context: ""
      model: "ggml-malachite-7b-0226-Q4_K_M"
      session: ""

    generate:
      model: "ggml-malachite-7b-0226-Q4_K_M"
      num_cpus: 10
      num_instructions_to_generate: 100
      path_to_taxonomy: "./taxonomy"
      prompt_file_path: "./cli/generator/prompt.txt"
      seed_tasks_path: "./cli/generator/seed_tasks.jsonl"

    list:
      path_to_taxonomy: "./taxonomy"

    log:
      level: info

    serve:
      model_path: "./models/ggml-malachite-7b-0226-Q4_K_M.gguf"
      n_gpu_layers: -1
    """
    )
    if not os.path.isfile(config_file_name):
        if os.path.dirname(config_file_name) != "":
            os.makedirs(os.path.dirname(config_file_name), exist_ok=True)
        with open(config_file_name, "w", encoding="utf-8") as model_file:
            model_file.write(config_yml_txt)
        click.echo("Config file is created at %s" % config_file_name)

    chat_config_toml_txt = textwrap.dedent(
        """
    api_base = "http://localhost:8000/v1"
    api_key = "no_api_key"
    model = "malachite-7b"
    vi_mode = false
    visible_overflow = true

    [contexts]
    default = "You are Labrador, an AI language model developed by IBM DMF (Data Model Factory) Alignment Team. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
    cli_helper = "You are an expert for command line interface and know all common commands. Answer the command to execute as it without any explanation."
    dictionary = "You are a professional English-Chinese translator. Translate the input to the other language by providing its part of speech (POS) followed by up-to 5 common but distinct translations in this format: `[{POS}] {translation 1}; {translation 2}; ...`. Do not provide nonexistent results."
    """
    )
    chat_config_file_name = os.path.join(
        os.path.dirname(config_file_name), "chat-cli.toml"
    )
    if not os.path.isfile(chat_config_file_name):
        if os.path.dirname(chat_config_file_name) != "":
            os.makedirs(os.path.dirname(chat_config_file_name), exist_ok=True)
        with open(chat_config_file_name, "w", encoding="utf-8") as model_file:
            model_file.write(chat_config_toml_txt)
        click.echo("Chat config file for is created at %s" % chat_config_file_name)


def create_subprocess(commands):
    return subprocess.run(
        commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
