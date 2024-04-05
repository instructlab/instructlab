from model import model
import click
from prettytable import ALL, PrettyTable
from pathlib import  Path
import time
import pandas

@model.command()
@click.option(
    "--model-dir", 
    help="Base directory where models are stored.",
    default="models",
    show_default=True,
)
def tag(model_path: str, tag: str, *args, **kwargs):
    """tag a model with a local name"""
    # need to get local storage dir from config
    # we need to also keep track of tags locally somewhere, probably in model dir. 
    
    
