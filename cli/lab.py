import click
import llama_cpp.llama_chat_format as llama_chat_format
import llama_cpp.server.app as llama_app
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings
import uvicorn

from click_didyoumean import DYMGroup
from .generator.generate_data import generate_data
from .chat.chat import chat_cli
from .download_model import download_model


@click.group(cls=DYMGroup)
def cli():
    """CLI for interacting with labrador"""
    pass


@cli.command()
def init():
    """Initializes environment for labrador"""
    click.echo("# init TBD")


@cli.command()
@click.option("--model", default="./models/ggml-labrador13B-model-Q4_K_M.gguf", show_default=True)
@click.option("--n_gpu_layers", default=-1, show_default=True)
def serve(model, n_gpu_layers):
    """Start a local server"""
    settings = Settings(model=model, n_gpu_layers=n_gpu_layers)
    app = create_app(settings=settings)
    llama_app._llama_proxy._current_model.chat_handler = llama_chat_format.Jinja2ChatFormatter(
        template="{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n' + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}", eos_token="<|endoftext|>", bos_token=""
    ).to_chat_handler()
    click.echo("Starting server process")
    click.echo("After application startup complete see http://127.0.0.1:8000/docs for API.")
    click.echo("Press CTRL+C to shutdown server.")
    uvicorn.run(app)  # TODO: host params, etc...


@cli.command()
@click.option("--model", default="ggml-labrador13B-model-Q4_K_M", show_default=True)
@click.option("--num_cpus", default=10, show_default=True)
@click.option("--taxonomy", default="../taxonomy", show_default=True, type=click.Path())
@click.option("--seed_file", default="./cli/generator/seed_tasks.jsonl", show_default=True, type=click.Path())
def generate(model, num_cpus, taxonomy, seed_file):
    """Generates synthetic data to enhance your example data"""
    generate_data(model_name=model, num_cpus=num_cpus, taxonomy=taxonomy, seed_tasks_path=seed_file)


@cli.command()
def train():
    """Trains labrador model"""
    click.echo("# train TBD")


@cli.command()
def test():
    """Perform rudimentary tests of the model"""
    click.echo("# test TBD")


@cli.command()
@click.argument(
    "question", nargs=-1, type=click.UNPROCESSED
)
@click.option(
    "-m", "--model", "model", help="Model to use"
)
@click.option(
    "-c", "--context", "context", help="Name of system context in config file", default="default"
)
@click.option(
    "-s", "--session", "session", help="Filepath of a dialog session file", type=click.File("r")
)
@click.option(
    "-qq", "--quick-question", "qq", help="Exit after answering question", is_flag=True
)
def chat(question, model, context, session, qq):
    """Run a chat using the modified model"""
    chat_cli(question, model, context, session, qq)


@cli.command()
def download():
    """Download the model(s) to train"""
    download_model()
