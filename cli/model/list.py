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
def list(model_dir: str, *args, **kwargs):
    """lists models"""
    # list from model dir
    # default is models/
    t = PrettyTable(['Model Name', "Model Age", "Model Size", "Valid"], hrules=ALL)
    for f in Path(model_dir).iterdir():
        # validate model, if not valid, incude but a printer column should list that
        # for now, we are assuming all models are valid
        stat = Path(f.absolute()).stat(follow_symlinks=False)
        if f.is_file():
            magnitudes = ['KB', 'MB', 'GB']
            magnitude = 'B'
            size = stat.st_size
            for mag in magnitudes:
                if size >= 1024:
                    magnitude = mag
                    size /= 1024
                else:
                    break
            
            age = pandas.to_timedelta(time.time_ns()-stat.st_ctime_ns)
            t.add_row([f.name, age, f'{size} {magnitude}', True])
    print(t)
