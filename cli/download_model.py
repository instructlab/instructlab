import click
import subprocess
import re


def download_model(gh_repo='https://github.com/open-labrador/cli.git', gh_version='v0.0.0'):
    """
    Download and combine the model file from a GitHub repository sourc.

    Parameters:
    - gh_repo (str): The URL of the GitHub repository containing the model. Default is the Open Labrador CLI repository.
    - gh_version (str): The GitHub release version of the model to download. Default is 'v0.0.0'.

    Returns:
    - None
    """

    model_file_split_keyword = '.split.'

    click.secho('\nMake sure the local environment has the "gh" cli. https://cli.github.com', fg="blue")
    click.echo("\nDownloading Models...\n")

    # Download Github release
    download_commands = ['gh', 'release', 'download', gh_version, '--repo', gh_repo]
    gh_result = create_subprocess(download_commands)
    if gh_result.stderr:
        raise Exception('gh command error occurred:\n\n %s' % gh_result.stderr.decode('utf-8'))
    
    # Get the list of local files
    ls_commands = ['ls']
    ls_result = create_subprocess(ls_commands)
    file_list = ls_result.stdout.decode('utf-8').split('\n')
    splitted_models = {}

    # Use Regex to capture model and files names that need to combine into one file.
    for file in file_list:
        if model_file_split_keyword in file:
            model_name_regex = "".join(["([^ \t\n,]+)", model_file_split_keyword, "[^ \t\n,]+"])
            regex_result = re.findall(model_name_regex, file)
            if regex_result:
                model_name = regex_result[0]
                if splitted_models.get(model_name):
                    splitted_models[model_name].append(file)
                else:
                    splitted_models[model_name] = [file]

    # Use the model and file names we captured to minimize the number of bash subprocess creations.
    for key, value in splitted_models.items():
        cat_commands = ['cat']
        splitted_model_files = [splitted_model_file for splitted_model_file in value]
        cat_commands.extend(splitted_model_files)
        cat_result = create_subprocess(cat_commands)
        if cat_result.stdout != None and cat_result.stdout != b'':
            with open(key, "wb") as model_file:
                model_file.write(cat_result.stdout)
            rm_commands = ['rm']
            rm_commands.extend(splitted_model_files)
            create_subprocess(rm_commands)

    click.echo("\nList of combined models: \n%s " % " ".join(splitted_models.keys()))

def create_subprocess(commands):
    return subprocess.run(commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
