# SPDX-License-Identifier: Apache-2.0

# Standard
from os import listdir
from os.path import dirname, exists

# Third Party
from git import GitError, Repo
import click

# First Party
from instructlab import clickext, utils
from instructlab.configuration import (
    DEFAULTS,
    ensure_storage_directories_exist,
    get_default_config,
    read_train_profile,
    write_config,
)


@click.command()
@click.option(
    "--interactive/--non-interactive",
    default=True,
    show_default=True,
    help="Initialize the environment assuming defaults.",
)
@click.option(
    "--model-path",
    type=click.Path(),  # we don't use utils.Pathlib() here, since this is a requested path to download a model, so the files are not guaranteed to exist yet
    default=lambda: DEFAULTS.DEFAULT_GGUF_MODEL,
    show_default="The instructlab data files location per the user's system.",
    help="Path to the model used during generation.",
)
@click.option(
    "--taxonomy-base",
    default=DEFAULTS.TAXONOMY_BASE,
    show_default=True,
    help="Base git-ref to use when listing/generating new taxonomy.",
)
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    default=lambda: DEFAULTS.TAXONOMY_DIR,
    show_default="The instructlab data files location per the user's system.",
    help="Path to where the taxonomy should be cloned.",
)
@click.option(
    "--repository",
    default=DEFAULTS.TAXONOMY_REPO,
    show_default=True,
    help="Taxonomy repository location.",
)
@click.option(
    "--min-taxonomy",
    is_flag=True,
    help="Shallow clone the taxonomy repository with minimum size. "
    "Please do not use this option if you are planning to contribute back "
    "using the same taxonomy repository. ",
)
@click.option("--train-profile", type=click.Path(), default=None)
@clickext.display_params
def init(
    interactive,
    model_path: str,
    taxonomy_path,
    taxonomy_base,
    repository,
    min_taxonomy,
    train_profile,
):
    """Initializes environment for InstructLab"""
    ensure_storage_directories_exist()
    clone_taxonomy_repo = True
    if interactive:
        if exists(DEFAULTS.CONFIG_FILE):
            overwrite = click.confirm(
                f"Found {DEFAULTS.CONFIG_FILE} in the current directory, do you still want to continue?"
            )
            if not overwrite:
                return
        click.echo(
            "Welcome to InstructLab CLI. This guide will help you to setup your environment."
        )
        click.echo(
            "Please provide the following values to initiate the "
            "environment [press Enter for defaults]:"
        )
        taxonomy_path = utils.expand_path(
            click.prompt("Path to taxonomy repo", default=taxonomy_path)
        )

    try:
        taxonomy_contents = listdir(taxonomy_path)
    except FileNotFoundError:
        taxonomy_contents = []
    if taxonomy_contents:
        clone_taxonomy_repo = False
    elif interactive:
        clone_taxonomy_repo = click.confirm(
            f"`{taxonomy_path}` seems to not exist or is empty. Should I clone {repository} for you?"
        )

    # clone taxonomy repo if it needs to be cloned
    if clone_taxonomy_repo:
        click.echo(f"Cloning {repository}...")
        clone_depth = False if not min_taxonomy else 1
        try:
            Repo.clone_from(
                repository,
                taxonomy_path,
                branch="main",
                recurse_submodules=True,
                depth=clone_depth,
            )
        except GitError as exc:
            click.secho(f"Failed to clone taxonomy repo: {exc}", fg="red")
            click.secho(f"Please make sure to manually run `git clone {repository}`")
            raise click.exceptions.Exit(1)

    # check if models dir exists, and if so ask for which model to use
    models_dir = dirname(model_path)
    if interactive and exists(models_dir):
        model_path = utils.expand_path(
            click.prompt("Path to your model", default=model_path)
        )
    click.echo(f"Generating `{DEFAULTS.CONFIG_FILE}`...")
    cfg = get_default_config()
    if train_profile is not None:
        cfg.train = read_train_profile(train_profile)
    cfg.chat.model = model_path
    cfg.generate.model = model_path
    cfg.serve.model_path = model_path
    cfg.generate.taxonomy_path = taxonomy_path
    cfg.generate.taxonomy_base = taxonomy_base
    cfg.evaluate.model = model_path
    cfg.evaluate.mt_bench_branch.taxonomy_path = taxonomy_path
    write_config(cfg)

    click.secho(
        "Initialization completed successfully, you're ready to start using `ilab`. Enjoy!",
        fg="green",
    )
