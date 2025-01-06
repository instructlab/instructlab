# SPDX-License-Identifier: Apache-2.0

# Standard
import datetime
import json
import logging
import os
import pathlib
import sys
import time
import traceback

# Third Party
from openai import OpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
import click
import httpx
import openai

# First Party
from instructlab import clickext
from instructlab import client_utils as ilabclient
from instructlab import configuration as cfg
from instructlab import log
from instructlab.client_utils import HttpClientParams

# Local
from ..client_utils import http_client
from ..utils import get_cli_helper_sysprompt, get_model_arch, get_sysprompt

logger = logging.getLogger(__name__)

HELP_MD = """
Help / TL;DR
- `/q`: **q**uit
- `/h`: show **h**elp
- `/a assistant`: **a**mend **a**ssistant (i.e., model)
- `/c context`: **c**hange **c**ontext (available contexts: default, cli_helper)
- `/lc`: **l**ist **c**ontexts
- `/m`: toggle **m**ultiline (for the next session only)
- `/M`: toggle **m**ultiline
- `/n`: **n**ew session
- `/N`: **n**ew session (ignoring loaded)
- `/d <int>`: **d**isplay previous response based on input, if passed 1 then previous, if 2 then second last response and so on.
- `/p <int>`: previous response in **p**lain text based on input, if passed 1 then previous, if 2 then second last response and so on.
- `/md <int>`: previous response in **M**ark**d**own based on input, if passed 1 then previous, if 2 then second last response and so on.
- `/s filepath`: **s**ave current session to `filepath`
- `/l filepath`: **l**oad `filepath` and start a new session
- `/L filepath`: **l**oad `filepath` (permanently) and start a new session

Press Alt (or Meta) and Enter or Esc Enter to end multiline input.
"""

CONTEXTS = {
    "default": get_sysprompt,
    "cli_helper": get_cli_helper_sysprompt,
}

PROMPT_HISTORY_FILEPATH = os.path.expanduser("~/.local/chat-cli.history")

PROMPT_PREFIX = ">>> "

DEFAULT_ENDPOINT = cfg._serve().api_base()


def is_openai_server_and_serving_model(
    endpoint: str, api_key: str, http_params: HttpClientParams
) -> bool:
    """
    Given an endpoint, returns whether or not the server is OpenAI-compatible
    and is actively serving at least one model.
    """
    try:
        models = ilabclient.list_models(
            endpoint, api_key=api_key, http_client=http_client(http_params)
        )
        return len(models.data) > 0
    except ilabclient.ClientException:
        return False


@click.command()
@click.argument(
    "question",
    nargs=-1,
    type=click.UNPROCESSED,
)
@click.option(
    "-m",
    "--model",
    cls=clickext.ConfigOption,
    required=True,  # default from config
)
@click.option(
    "-c",
    "--context",
    cls=clickext.ConfigOption,
    required=True,  # default from config
)
@click.option(
    "-s",
    "--session",
    type=click.File("r"),
    cls=clickext.ConfigOption,
)
@click.option(
    "-qq",
    "--quick-question",
    is_flag=True,
    help="Exit after answering question.",
)
@click.option(
    "--max-tokens",
    type=click.INT,
    cls=clickext.ConfigOption,
)
@click.option(
    "--endpoint-url",
    type=click.STRING,
    help="Custom URL endpoint for OpenAI-compatible API. Defaults to the `ilab model serve` endpoint.",
)
@click.option(
    "--api-key",
    type=click.STRING,
    default=cfg.DEFAULTS.API_KEY,  # Note: do not expose default API key
    help="API key for API endpoint. [default: config.DEFAULT_API_KEY]",
)
@click.option(
    "--tls-insecure",
    is_flag=True,
    help="Disable TLS verification.",
)
@click.option(
    "--tls-client-cert",
    type=click.Path(),
    default="",
    show_default=True,
    help="Path to the TLS client certificate to use.",
)
@click.option(
    "--tls-client-key",
    type=click.Path(),
    default="",
    show_default=True,
    help="Path to the TLS client key to use.",
)
@click.option(
    "--tls-client-passwd",
    type=click.STRING,
    default="",
    help="TLS client certificate password.",
)
@click.option(
    "--model-family",
    help="Force model family to use when picking a chat template",
)
@click.option(
    "--serving-log-file",
    type=click.Path(path_type=pathlib.Path),
    required=False,
    help="Log file path to write server logs to.",
)
@click.option(
    "-t",
    "--temperature",
    cls=clickext.ConfigOption,
)
@click.option(
    "-pf",
    "--prompt-file",
    type=click.Path(exists=True),
    help="Path to a file containing test prompts (used with -of). Each line is treated as a separate input.",
)
@click.option(
    "-of",
    "--output-file",
    type=click.Path(writable=True),
    help="Path to save model responses (used with -pf).",
)
@click.pass_context
@clickext.display_params
def chat(
    ctx,
    question,
    model,
    context,
    session,
    quick_question,
    max_tokens,
    endpoint_url,
    api_key,
    tls_insecure,
    tls_client_cert,
    tls_client_key,
    tls_client_passwd,
    model_family,
    serving_log_file,
    temperature,
    prompt_file,
    output_file,
):
    """Runs a chat using the modified model"""
    # pylint: disable=import-outside-toplevel
    # First Party
    from instructlab.model.backends.common import is_temp_server_running

    serve_server = ctx.obj.config.serve.server
    users_endpoint_url = cfg.get_api_base(serve_server.host, serve_server.port)

    # we prefer the given endpoint when one is provided, else we check if the user
    # is actively serving something before falling back to serving our own model
    backend_instance = None
    if endpoint_url:
        api_base = endpoint_url
    elif is_openai_server_and_serving_model(
        users_endpoint_url,
        api_key,
        http_params={
            "tls_client_cert": tls_client_cert,
            "tls_client_key": tls_client_key,
            "tls_client_passwd": tls_client_passwd,
            "tls_insecure": tls_insecure,
        },
    ):
        api_base = users_endpoint_url
        if serving_log_file:
            logger.warning(
                "Setting serving log file (--serving-log-file) is not supported when the server is already running"
            )
    else:
        # First Party
        from instructlab.model.backends import backends

        # If a log file is specified, write logs to the file
        root_logger = logging.getLogger()
        if serving_log_file:
            log.add_file_handler_to_logger(root_logger, serving_log_file)

        ctx.obj.config.serve.llama_cpp.llm_family = model_family
        backend_instance = backends.select_backend(
            ctx.obj.config.serve,
            model_path=model,
            log_file=serving_log_file,
        )
        try:
            # Run the llama server
            api_base = backend_instance.run_detached(http_client(ctx.params))
        except Exception as exc:
            click.secho(f"Failed to start server: {exc}", fg="red")
            raise click.exceptions.Exit(1)

    # if only the chat is running (`ilab model chat`) and the temp server is not, the chat interacts
    # in server mode (`ilab model serve` is running somewhere, or we are talking to another
    # OpenAI compatible endpoint).
    if not is_temp_server_running():
        # Try to get the model name right if we know we're talking to a local `ilab model serve`.
        #
        # If the model from the CLI and the one in the config are the same, use the one from the
        # server if they are different else let's use what the user provided
        #
        # 'model' will always get a value and never be None so it's hard to distinguish whether
        # the value came from the user input or the default value.
        # We can only assume that if the value is the same as the default value and the value
        # from the config is the same as the default value, then the user didn't provide a value
        # we then compare it with the value from the server to see if it's different
        if (
            # We need to get the base name of the model because the model path is a full path and
            # the once from the config is just the model name
            os.path.basename(model) == cfg.DEFAULTS.GRANITE_GGUF_MODEL_NAME
            and os.path.basename(ctx.obj.config.chat.model)
            == cfg.DEFAULTS.GRANITE_GGUF_MODEL_NAME
            and api_base == ctx.obj.config.serve.api_base()
        ):
            logger.debug(
                "No model was provided by the user as a CLI argument or in the config, will use the model from the server"
            )
            try:
                models = ilabclient.list_models(
                    api_base=api_base,
                    http_client=http_client(ctx.params),
                )

                # Currently, we only present a single model so we can safely assume that the first model
                server_model = models.data[0].id if models is not None else None

                # override 'model' with the first returned model if not provided so that the chat print
                # the model used by the server
                model = (
                    server_model
                    if server_model is not None
                    and server_model != ctx.obj.config.chat.model
                    else model
                )
                logger.debug(f"Using model from server {model}")
            except ilabclient.ClientException as exc:
                click.secho(
                    f"Failed to list models from {api_base}. Please check the API key and endpoint.",
                    fg="red",
                )
                # Right now is_temp_server() does not check if a subprocessed vllm is up
                # shut it down just in case an exception is raised in the try
                # TODO: revise is_temp_server to check if a vllm server is running
                if backend_instance is not None:
                    backend_instance.shutdown()
                raise click.exceptions.Exit(1) from exc

    try:
        chat_cli(
            ctx,
            api_base=api_base,
            config=ctx.obj.config.chat,
            question=question,
            model=model,
            context=context,
            session=session,
            qq=quick_question,
            max_tokens=max_tokens,
            temperature=temperature,
            prompt_file=prompt_file,
            output_file=output_file,
        )
    except ChatException as exc:
        click.secho(f"Executing chat failed with: {exc}", fg="red")
        raise click.exceptions.Exit(1)
    finally:
        if backend_instance is not None:
            backend_instance.shutdown()


class ChatException(Exception):
    """An exception raised during chat step."""


class ChatQuitException(Exception):
    """A quit command was executed during chat."""


# TODO Autosave chat history
class ConsoleChatBot:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        model,
        client,
        vi_mode=False,
        prompt=True,
        vertical_overflow="ellipsis",
        loaded=None,
        log_file=None,
        max_tokens=None,
        temperature=1.0,
    ):
        self.client = client
        self.model = model
        self.vi_mode = vi_mode
        self.vertical_overflow = vertical_overflow
        self.loaded = loaded
        self.log_file = log_file
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.console = Console()

        self.input = None
        if prompt:
            os.makedirs(os.path.dirname(PROMPT_HISTORY_FILEPATH), exist_ok=True)
            self.input = PromptSession(history=FileHistory(PROMPT_HISTORY_FILEPATH))
        self.multiline = False
        self.multiline_mode = 0

        self.info = {}
        self._reset_session()

    def _reset_session(self, hard=False):
        if hard:
            self.loaded = {}
        self.info["messages"] = (
            []
            if hard or ("messages" not in self.loaded)
            else [*self.loaded["messages"]]
        )

    def _sys_print(self, *args, **kwargs):
        self.console.print(Panel(*args, title="system", **kwargs))

    def log_message(self, msg):
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as fp:
                fp.write(msg)

    def greet(self, help=False, new=False, session_name="new session"):  # pylint: disable=redefined-builtin
        side_info_str = (" (type `/h` for help)" if help else "") + (
            f" ({session_name})" if new else ""
        )
        self._sys_print(
            Markdown(
                f"Welcome to InstructLab Chat w/ **{self.model_name.upper()}**"
                + side_info_str
            )
        )

    @property
    def model_name(self):
        return os.path.basename(os.path.normpath(self.model))

    @property
    def _right_prompt(self):
        return FormattedText(
            [
                (
                    "#3f7cac bold",
                    f"[{'M' if self.multiline else 'S'}]",
                ),  # info blue for multiple
                *(
                    [("bold", f"[{self.loaded['name']}]")]
                    if "name" in self.loaded
                    else []
                ),  # loaded context/session file
                # TODO: Fix openai package to fix the openai error.
                # *([] if openai.proxy is None else [('#d08770 bold', "[proxied]")]), # indicate prox
            ]
        )

    def _handle_quit(self, _):
        raise ChatQuitException

    def _handle_help(self, _):
        self._sys_print(Markdown(HELP_MD))
        raise KeyboardInterrupt

    def _handle_multiline(self, content):
        temp = content == "/m"  # soft multiline only for next prompt
        self.multiline = not self.multiline
        self.multiline_mode = 1 if not temp else 2
        raise KeyboardInterrupt

    def _handle_amend(self, content):
        cs = content.split()
        if len(cs) < 2:
            self._sys_print(
                Markdown(
                    "**WARNING**: The second argument `assistant` is missing in the `/a assistant` command."
                )
            )
            raise KeyboardInterrupt
        self.model = cs[1]
        self._reset_session()
        self.greet(new=True)
        raise KeyboardInterrupt

    def _handle_context(self, content):
        if CONTEXTS is None:
            self._sys_print(
                Markdown("**WARNING**: No contexts loaded from the config file.")
            )
            raise KeyboardInterrupt
        cs = content.split()
        if len(cs) < 2:
            self._sys_print(
                Markdown(
                    "**WARNING**: The second argument `context` is missing in the `/c context` command."
                )
            )
            raise KeyboardInterrupt
        context = cs[1]
        if context not in CONTEXTS:
            available_contexts = ", ".join(CONTEXTS.keys())
            self._sys_print(
                Markdown(
                    f"**WARNING**: Context `{context}` not found. "
                    f"Available contexts: `{available_contexts}`"
                )
            )
            raise KeyboardInterrupt
        self.loaded["name"] = context

        sys_prompt = CONTEXTS.get(context, "default")(get_model_arch(self.model))
        self.loaded["messages"] = [{"role": "system", "content": sys_prompt}]
        self._reset_session()
        self.greet(new=True)
        raise KeyboardInterrupt

    def _handle_new_session(self, content):
        hard = content == "/N"  # hard new ignores loaded context/session
        self._reset_session(hard=hard)
        self.greet(new=True)
        raise KeyboardInterrupt

    def __handle_replay(self, content, display_wrapper=lambda x: x):
        # if the history is empty, then return
        if (
            len(self.info["messages"]) == 1
            and self.info["messages"][0]["role"] == "system"
        ):
            raise KeyboardInterrupt
        cs = content.split()
        try:
            i = 1 if len(cs) == 1 else int(cs[1]) * 2 - 1
            if abs(i) >= len(self.info["messages"]):
                raise IndexError
        except (IndexError, ValueError) as exc:
            self.console.print(
                display_wrapper("Invalid index: " + content), style="bold red"
            )
            raise KeyboardInterrupt from exc
        if len(self.info["messages"]) > abs(i):
            self.console.print(display_wrapper(self.info["messages"][-i]["content"]))
        raise KeyboardInterrupt

    def _handle_display(self, content):
        return self.__handle_replay(content, display_wrapper=lambda x: Panel(x))  # pylint: disable=unnecessary-lambda

    def _load_session_history(self, content=None):
        data = self.info["messages"]
        if content is not None:
            data = content["messages"]
        for m in data:
            if m["role"] == "user":
                self.console.print(
                    "\n" + PROMPT_PREFIX + m["content"], style="dim grey0"
                )
            else:
                self.console.print(Panel(m["content"]), style="dim grey0")

    def _handle_plain(self, content):
        return self.__handle_replay(content)

    def _handle_markdown(self, content):
        return self.__handle_replay(
            content,
            display_wrapper=(
                lambda x: Panel(
                    Markdown(x), subtitle_align="right", subtitle="rendered as Markdown"
                )
            ),
        )

    def _handle_save_session(self, content):
        cs = content.split()
        if len(cs) < 2:
            self._sys_print(
                Markdown(
                    "**WARNING**: The second argument `filepath` is missing in the `/s filepath` command."
                )
            )
            raise KeyboardInterrupt
        filepath = cs[1]
        with open(filepath, "w", encoding="utf-8") as outfile:
            json.dump(self.info["messages"], outfile, indent=4)
        raise KeyboardInterrupt

    def _handle_load_session(self, content):
        cs = content.split()
        if len(cs) < 2:
            self._sys_print(
                Markdown(
                    "**WARNING**: The second argument `filepath` is missing in the `/l filepath` or `/L filepath` command."
                )
            )
            raise KeyboardInterrupt
        filepath = cs[1]
        if not os.path.exists(filepath):
            self._sys_print(
                Markdown(
                    f"**WARNING**: File `{filepath}` specified in the `/l filepath` or `/L filepath` command does not exist."
                )
            )
            raise KeyboardInterrupt
        with open(filepath, "r", encoding="utf-8") as session:
            messages = json.loads(session.read())
        if content[:2] == "/L":
            self.loaded["name"] = filepath
            self.loaded["messages"] = messages
            self._reset_session()
            self.greet(new=True)
        else:
            self._reset_session()
            self.info["messages"] = [*messages]
            self.greet(new=True, session_name=filepath)

        # now load session's history
        self._load_session_history()
        raise KeyboardInterrupt

    def _handle_empty(self):
        raise KeyboardInterrupt

    def _update_conversation(self, content, role):
        assert role in ("user", "assistant")
        message = {"role": role, "content": content}
        self.info["messages"].append(message)

    def _handle_list_contexts(self, _):
        # reconstruct contexts dict based on values passed at runtime
        context_dict = dict.fromkeys(CONTEXTS, None)
        context_dict["default"] = get_sysprompt(
            get_model_arch(pathlib.Path(self.model))
        )
        context_dict["cli_helper"] = get_cli_helper_sysprompt()

        context_list = "\n\n".join(
            [f"**{key}**:\n{value}" for key, value in context_dict.items()]
        )
        self._sys_print(Markdown(f"**Available contexts:**\n\n{context_list}"))
        raise KeyboardInterrupt

    def start_prompt(
        self,
        logger,  # pylint: disable=redefined-outer-name
        content=None,
        box=True,
    ):
        handlers = {
            "/q": self._handle_quit,
            "quit": self._handle_quit,
            "exit": self._handle_quit,
            "/h": self._handle_help,
            "/a": self._handle_amend,
            "/c": self._handle_context,
            "/m": self._handle_multiline,
            "/n": self._handle_new_session,
            "/d": self._handle_display,
            "/p": self._handle_plain,
            "/md": self._handle_markdown,
            "/s": self._handle_save_session,
            "/l": self._handle_load_session,
            "/lc": self._handle_list_contexts,
        }

        if content is None:
            content = self.input.prompt(
                PROMPT_PREFIX,
                rprompt=self._right_prompt,
                vi_mode=True,
                multiline=self.multiline,
            )

        # Handle empty
        if content.strip() == "":
            raise KeyboardInterrupt

        # Handle commands
        handler = handlers.get(content.split()[0].lower(), None)
        if handler is not None:
            handler(content)

        self.log_message(PROMPT_PREFIX + content + "\n\n")

        # Update message history and token counters
        self._update_conversation(content, "user")

        # Deal with temp multiline
        if self.multiline_mode == 2:
            self.multiline_mode = 0
            self.multiline = not self.multiline

        # Temperature parameters
        create_params = {}
        # https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature
        create_params["temperature"] = self.temperature

        if self.max_tokens:
            create_params["max_tokens"] = self.max_tokens

        # Get and parse response
        try:
            while True:
                # Loop to catch situations where we need to retry, such as context length exceeded
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.info["messages"],
                        stream=True,
                        **create_params,
                    )
                except openai.BadRequestError as e:
                    logger.debug(f"BadRequestError: {e}")
                    if e.code == "context_length_exceeded":
                        if len(self.info["messages"]) > 1:
                            # Trim the oldest entry in our message history
                            logger.debug(
                                "Trimming message history to attempt to fit context length"
                            )
                            self.info["messages"] = self.info["messages"][1:]
                            continue
                        # We only have a single message and it's still to big.
                        self.console.print(
                            "Message too large for context size.", style="bold red"
                        )
                        self.info["messages"].pop()
                        raise KeyboardInterrupt from e
                except openai.InternalServerError as e:
                    logger.debug(f"InternalServerError: {e}")
                    self.info["messages"].clear()
                    raise KeyboardInterrupt from e
                assert (
                    next(response).choices[0].delta.role == "assistant"
                ), 'first response should be {"role": "assistant"}'
                break
        except openai.AuthenticationError as e:
            self.console.print(
                "Invalid API Key. Please set it in your config file.", style="bold red"
            )
            raise ChatException("API Key Error") from e
        except openai.RateLimitError as e:
            self.console.print(
                "Rate limit or maximum monthly limit exceeded", style="bold red"
            )
            self.info["messages"].pop()
            raise ChatException("Rate limit exceeded") from e
        except openai.APIConnectionError as e:
            logger.debug("Connection error, try again...", exc_info=True)
            self.console.print("Connection error, try again...", style="red bold")
            self.info["messages"].pop()
            raise KeyboardInterrupt from e
        except KeyboardInterrupt as e:
            raise e
        except httpx.RemoteProtocolError as e:
            logger.debug("Connection to the server was closed", exc_info=True)
            self.console.print("Connection to the server was closed", style="bold red")
            self.info["messages"].pop()
            raise ChatException("Connection to the server was closed") from e
        except Exception as e:
            if logger.getEffectiveLevel() == logging.DEBUG:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                message = f"Unknown error: {''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))}"
            else:
                message = f"Unknown error: {sys.exc_info()[0]}"

            self.console.print("Unknown error", style="bold red")
            raise ChatException(message) from e

        response_content = Text()
        panel = (
            Panel(
                response_content,
                title=self.model_name,
                subtitle_align="right",
            )
            if box
            else response_content
        )
        subtitle = None
        with Live(
            panel,
            console=self.console,
            refresh_per_second=5,
            vertical_overflow=self.vertical_overflow,
        ):
            start_time = time.time()
            for chunk in response:
                chunk_message = chunk.choices[0].delta
                if chunk_message.content:
                    response_content.append(chunk_message.content)

                if box:
                    panel.subtitle = f"elapsed {time.time() - start_time:.3f} seconds"
            subtitle = f"elapsed {time.time() - start_time:.3f} seconds"

        # Update chat logs
        if subtitle is not None:
            self.log_message("- " + subtitle + " -\n")
        self.log_message(response_content.plain + "\n\n")
        # Update message history and token counters
        self._update_conversation(response_content.plain, "assistant")


def chat_cli(
    ctx,
    api_base,
    config,
    question,
    model,
    context,
    session,
    qq,
    max_tokens,
    temperature,
    prompt_file,
    output_file,
):
    """Starts a CLI-based chat with the server"""
    client = OpenAI(
        base_url=api_base,
        api_key=ctx.params["api_key"],
        timeout=cfg.DEFAULTS.CONNECTION_TIMEOUT,
        http_client=http_client(ctx.params),
    )
    # ensure the model specified exists on the server. with backends like vllm, this is crucial.
    try:
        model_list = client.models.list().data
    except openai.OpenAIError as exc:
        raise ChatException(f"Is the server running? {exc}") from exc
    model_ids = []
    for m in model_list:
        model_ids.append(m.id)
    # if a model is already being served, ignore whatever model may have been supplied with the chat command
    if (len(model_ids) > 0) and (not any(model == m for m in model_ids)):
        # assuming chatting with first model in list if there are multiple. Temp fix
        logger.info(
            f"Requested model {model} is not served by the server. Proceeding to chat with served model: {model_ids[0]}"
        )
        model = model_ids[0]

    # Load context/session
    loaded = {}

    # Context config file
    # global CONTEXTS
    # CONTEXTS = config["contexts"]
    if context not in CONTEXTS:
        logger.info(f"Context {context} not found in the config file. Using default.")
        context = "default"
    loaded["name"] = context
    sys_prompt = CONTEXTS.get(context, "default")(get_model_arch(pathlib.Path(model)))
    loaded["messages"] = [{"role": "system", "content": sys_prompt}]

    # Session from CLI
    if session is not None:
        loaded["name"] = os.path.basename(session.name).strip(".json")
        try:
            loaded["messages"] = json.loads(session.read())
        except json.JSONDecodeError as exc:
            raise ChatException(
                f"Session file {session.name} is not a valid JSON file."
            ) from exc

    log_file = None
    if config.logs_dir:
        date_suffix = (
            datetime.datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
        )
        os.makedirs(config.logs_dir, exist_ok=True)
        log_file = f"{config.logs_dir}/chat_{date_suffix}.log"

    # Initialize chat bot
    ccb = ConsoleChatBot(
        config.model if model is None else model,
        client=client,
        vi_mode=config.vi_mode,
        log_file=log_file,
        prompt=not qq,
        vertical_overflow=("visible" if config.visible_overflow else "ellipsis"),
        loaded=loaded,
        temperature=(temperature if temperature is not None else config.temperature),
        max_tokens=(max_tokens if max_tokens else config.max_tokens),
    )

    # Handle prompts from file and output
    if prompt_file:
        process_prompts_from_file(ccb, prompt_file, output_file)
        return

    if not qq and session is None:
        # Greet
        ccb.greet(help=True)

    # Use the input question to start with
    if len(question) > 0:
        question = " ".join(question)
        if not qq:
            print(f"{PROMPT_PREFIX}{question}")
        try:
            ccb.start_prompt(logger, content=question, box=not qq)
        except ChatException as exc:
            raise ChatException(f"API issue found while executing chat: {exc}") from exc
        except (ChatQuitException, KeyboardInterrupt, EOFError):
            return

    if qq:
        return

    # load the history
    if session is not None:
        ccb._load_session_history(loaded)

    # Start chatting
    while True:
        try:
            ccb.start_prompt(logger)
        except KeyboardInterrupt:
            continue
        except ChatException as exc:
            raise ChatException(f"API issue found while executing chat: {exc}") from exc
        except httpx.RemoteProtocolError as exc:
            raise ChatException("Connection to the server was closed") from exc
        except (ChatQuitException, EOFError):
            return


def process_prompts_from_file(ccb, prompt_file, output_file):
    """Processes prompts from a file and outputs to a file or console"""
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as file:
        prompts = [line.strip() for line in file if line.strip()]

    if not prompts:
        click.secho(
            f"The prompt file '{prompt_file}' is empty. Please provide a valid file with prompts (one question per line).",
            fg="red",
        )
        return

    click.echo("Start processing prompts...")
    total_prompts = len(prompts)

    if output_file:
        # Save to file
        with open(output_file, "a", encoding="utf-8") as output:
            output.write(
                f"--- Processing started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n"
            )
            process_and_output(ccb, prompts, output, total_prompts)
            click.echo(f"Finished processing prompts. Results saved to {output_file}.")

            click.echo(f"Results from {output_file}:")
            with open(output_file, "r", encoding="utf-8") as f:
                click.echo(f.read())
    else:
        # If no output file, print to console
        process_and_output(ccb, prompts, None, total_prompts)
        click.echo("Finished processing prompts.")


def process_prompt(ccb, prompt, output=None):
    messages = [{"role": "system", "content": ccb.info["messages"][0]["content"]}]
    messages.append({"role": "user", "content": prompt})

    response_content = ccb.client.chat.completions.create(
        model=ccb.model,
        messages=messages,
        temperature=ccb.temperature,
        max_tokens=ccb.max_tokens,
    )
    response = response_content.choices[0].message.content

    if output:
        output.write(f"Q: {prompt}\nA: {response}\n\n")
        output.flush()

    return response


def process_and_output(ccb, prompts, output, total_prompts):
    for idx, prompt in enumerate(prompts, start=1):
        try:
            click.echo(f"Processing prompt {idx}/{total_prompts}...")
            response = process_prompt(ccb, prompt, output)
            if not output:
                click.echo(f"Q: {prompt}\nA: {response}\n")
            ccb._reset_session()
        except ChatException as exc:
            logger.error(f"Error processing prompt '{prompt}': {exc}")
            continue
