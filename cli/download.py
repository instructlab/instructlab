import click
import subprocess
import re
import os


def download_model(gh_repo='https://github.com/open-labrador/cli.git', gh_release='latest', dir='.', pattern=''):
    """
    Download and combine the model file from a GitHub repository source.

    Parameters:
    - gh_repo (str): The URL of the GitHub repository containing the model. Default is the Open Labrador CLI repository.
    - gh_release (str): The GitHub release version of the model to download. Default is 'latest'.
    - dir(str): The local directory to download the model files into
    - pattern(str): Download only assets that match a glob pattern

    Returns:
    - None
    """

    model_file_split_keyword = '.split.'

    click.secho('\nMake sure the local environment has the "gh" cli. https://cli.github.com', fg="blue")
    click.echo("\nDownloading Models from %s with version %s to local directory %s ...\n" % (gh_repo, gh_release, dir))

    # Download Github release
    download_commands = ['gh', 'release', 'download', gh_release, '--repo', gh_repo, '--dir', dir]
    if pattern != '':
        download_commands.extend(['--pattern', pattern])
    if gh_release == 'latest':
        download_commands.pop(3)  # remove gh_release arg to download the latest version
        if pattern == '':  # Latest release needs to specify the pattern argument to download all files
            download_commands.extend(['--pattern', '*'])
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
    combined_model_list = []
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
            combined_model_list.append(key)

    click.echo("\nDownload Completed.")
    if combined_model_list:
        click.echo("\nList of combined models: ")
        for model_name in combined_model_list:
            click.echo("%s" % model_name)


def clone_taxonomy(gh_repo='https://github.com/open-labrador/taxonomy.git',
                   gh_branch='main',
                   dir_name='taxonomy',
                   git_filter_spec=''):
    """
    Clone the taxonomy repository from a Git repository source.

    Parameters:
    - gh_repo (str): The URL of the taxonomy Git repository. Default is the Open Labrador taxonomy repository.
    - gh_branch (str): The Github branch of the taxonomy repository. Default is main
    - dir_name(str): The local directory name to clone the taxonomy repository into
    - git_filter_spec(str): Optional path to the git filter spec for git partial clone

    Returns:
    - None
    """

    click.echo('\nCloning repository %s with branch "%s" to %s ...' % (gh_repo, gh_branch, dir_name))

    # Clone taxonomy repo
    git_clone_commands = ['git', 'clone', gh_repo, dir_name]
    if git_filter_spec != '' and os.path.exists(git_filter_spec):
        # TODO: Add gitfilterspec to sparse clone github repo
        git_filter_arg = ''.join(['--filter=sparse:oid=', gh_branch, ':', git_filter_spec])
        git_sparse_clone_flags = ['--sparse', git_filter_arg]
        git_clone_commands.extend(git_sparse_clone_flags)
    else:
        git_clone_commands.extend(['--branch', gh_branch])
    
    result = create_subprocess(git_clone_commands)
    if result.stderr:
        click.echo('\n%s' % result.stderr.decode('utf-8'))
    click.echo('Git clone completed.')


def create_subprocess(commands):
    return subprocess.run(commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
