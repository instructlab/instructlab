import click
from config import config
import utils

@config.command()
@click.option(
    "--interactive/--non-interactive",
    default=True,
    show_default=True,
    help="Initialize the environment assuming defaults.",
)
@click.option(
    "--model-path",
    type=click.Path(),
    default=config.DEFAULT_MODEL_PATH,
    show_default=True,
    help="Path to the model used during generation.",
)
@click.option(
    "--taxonomy-base",
    default=config.DEFAULT_TAXONOMY_BASE,
    show_default=True,
    help="Base git-ref to use when listing/generating new taxonomy.",
)
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    default=config.DEFAULT_TAXONOMY_PATH,
    show_default=True,
    help=f"Path to {config.DEFAULT_TAXONOMY_REPO} clone.",
)
@click.option(
    "--repository",
    default=config.DEFAULT_TAXONOMY_REPO,
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
def init(
    interactive,
    model_path,
    taxonomy_path,
    taxonomy_base,
    repository,
    min_taxonomy,
):
    """Initializes environment for InstructLab"""
    if exists(config.DEFAULT_CONFIG):
        overwrite = click.confirm(
            f"Found {config.DEFAULT_CONFIG} in the current directory, do you still want to continue?"
        )
        if not overwrite:
            return

    clone_taxonomy_repo = True
    if interactive:
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
        taxonomy_contents = os.listdir(taxonomy_path)
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
        try:
            if not min_taxonomy:
                Repo.clone_from(repository, taxonomy_path, branch="main")
            else:
                Repo.clone_from(repository, taxonomy_path, branch="main", depth=1)
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
    click.echo(f"Generating `{config.DEFAULT_CONFIG}` in the current directory...")
    cfg = config.get_default_config()
    model = splitext(basename(model_path))[0]
    cfg.chat.model = model
    cfg.generate.model = model
    cfg.serve.model_path = model_path
    cfg.generate.taxonomy_path = taxonomy_path
    cfg.generate.taxonomy_base = taxonomy_base
    config.write_config(cfg)

    click.echo(
        "Initialization completed successfully, you're ready to start using `ilab`. Enjoy!"
    )
