import click


def download_model():
    """Outputs instructions for downloading and combining the model files."""
    click.echo("\nTemporarily, the model's files are hosted on GitHub, split in parts, and require to be downloaded manually and combined.")
    click.echo("\nTo download and prepare the model, please follow the instructions below:")
    click.echo("\n1. You can download the files manually from the latest release on GitHub or use the `gh` command line tool:")
    click.secho('\ngh release download v0.0.0 --repo "https://github.com/open-labrador/cli.git"', fg="blue")
    click.echo("\n2. Once the files are downloaded, use `cat` to combine them:")
    click.secho('\ncat ggml-labrador13B-model-Q4_K_M.gguf.split.* > ggml-labrador13B-model-Q4_K_M.gguf && rm ggml-labrador13B-model-Q4_K_M..gguf.split.*', fg="blue")
