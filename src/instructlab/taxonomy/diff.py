# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import os

# Third Party
from git import GitError
import click
import yaml

# First Party
from instructlab import clickext
from instructlab.configuration import DEFAULTS

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    help="Path to where the taxonomy is stored locally.",
    default=lambda: DEFAULTS.TAXONOMY_DIR,
    show_default="Default taxonomy location in the instructlab data directory.",
)
@click.option(
    "--taxonomy-base",
    help="Base git-ref to use for taxonomy.",
)
@click.option(
    "--yaml-rules",
    type=click.Path(),
    default=None,
    help="Custom rules file for YAML linting.",
)
@click.option(
    "--quiet",
    is_flag=True,
    help="Suppress all output. Call returns 0 if check passes, 1 otherwise.",
)
@click.pass_context
@clickext.display_params
def diff(ctx, taxonomy_path, taxonomy_base, yaml_rules, quiet):
    """
    Lists taxonomy files that have changed since <taxonomy-base>
    and checks that taxonomy is valid. Similar to 'git diff <ref>'.
    """
    # pylint: disable=import-outside-toplevel
    # Local
    from ..utils import TaxonomyReadingException, get_taxonomy_diff, validate_taxonomy

    if not taxonomy_base:
        taxonomy_base = ctx.obj.config.generate.taxonomy_base
    if not taxonomy_path:
        taxonomy_path = ctx.obj.config.generate.taxonomy_path

    if not quiet:
        is_file = os.path.isfile(taxonomy_path)
        if is_file:  # taxonomy_path is file
            click.echo(taxonomy_path)
        else:  # taxonomy_path is dir
            try:
                updated_taxonomy_files = get_taxonomy_diff(taxonomy_path, taxonomy_base)
            except (TaxonomyReadingException, GitError) as exc:
                click.secho(
                    f"Reading taxonomy failed with the following error: {exc}",
                    fg="red",
                )
                raise SystemExit(1) from exc
            for f in updated_taxonomy_files:
                click.echo(f)
    try:
        validate_taxonomy(None, taxonomy_path, taxonomy_base, yaml_rules)
    except (TaxonomyReadingException, yaml.YAMLError) as exc:
        if not quiet:
            click.secho(
                f"Reading taxonomy failed with the following error: {exc}",
                fg="red",
            )
        raise SystemExit(1) from exc
    if not quiet:
        click.secho(
            f"Taxonomy in {taxonomy_path} is valid :)",
            fg="green",
        )
