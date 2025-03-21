# SPDX-License-Identifier: Apache-2.0

# Standard
from subprocess import CalledProcessError
from typing import List
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
import httpx
import openai
import requests

# First Party
from instructlab import client_utils as ilabclient
from instructlab import configuration as cfg
from instructlab import log
from instructlab.client_utils import HttpClientParams
from instructlab.model.download import resolve_model_path

# Local
from ..client_utils import http_client
from ..feature_gates import FeatureGating, FeatureScopes, GatedFeatures
from ..rag.document_store import DocumentStoreRetriever
from ..rag.document_store_factory import create_document_retriever
from ..utils import get_cli_helper_sysprompt, get_model_arch, get_sysprompt
from .backends import backends
from .backends.common import CHAT_TEMPLATE_TOKENIZER

logger = logging.getLogger(__name__)

# ANSI escape sequence for red text
RED = "\033[31m"
RESET = "\033[0m"

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
    "cli_helper": lambda _: get_cli_helper_sysprompt(),
}

PROMPT_HISTORY_FILEPATH = os.path.expanduser("~/.local/chat-cli.history")

PROMPT_PREFIX = ">>> "


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
        retriever=None,
        vi_mode=False,
        prompt=True,
        vertical_overflow="ellipsis",
        loaded=None,
        log_file=None,
        max_tokens=None,
        max_ctx_size=None,
        temperature=1.0,
        backend_type="",
        box=True,
    ):
        self.client = client
        self.retriever: DocumentStoreRetriever | None = retriever
        self.model = model
        self.vi_mode = vi_mode
        self.vertical_overflow = vertical_overflow
        self.loaded = loaded
        self.log_file = log_file
        self.max_tokens = max_tokens
        self.max_ctx_size = max_ctx_size
        self.temperature = temperature
        self.backend_type = backend_type
        self.box = box

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
        if self.box:
            self.console.print(Panel(*args, title="system", **kwargs))
        else:
            self.console.print(*args)

    def log_message(self, msg):
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as fp:
                fp.write(msg)

    def greet(self, help=False, new=False, session_name="new session"):  # pylint: disable=redefined-builtin
        side_info_str = (" (type `/h` for help)" if help else "") + (
            f" ({session_name})" if new else ""
        )
        message = (
            f"Welcome to InstructLab Chat w/ **{self.model_name.upper()}**"
            + side_info_str
        )
        if self.box:
            self._sys_print(Markdown(message))
        else:
            self.console.print(message)

    @property
    def model_name(self):
        return os.path.basename(os.path.normpath(self.model))

    @property
    def _right_prompt(self):
        if not self.box:
            return None
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

        sys_prompt = CONTEXTS.get(context, "default")(
            get_model_arch(pathlib.Path(self.model))
        )
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
        return self.__handle_replay(
            content,
            display_wrapper=lambda x: Panel(x) if self.box else x,  # pylint: disable=unnecessary-lambda
        )

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
                if self.box:
                    self.console.print(Panel(m["content"]), style="dim grey0")
                else:
                    self.console.print(m["content"], style="dim grey0")

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

        # if RAG is enabled, fetch context and insert into session
        # TODO: what if context is already too long? note that current retriever implementation concatenates all docs
        # TODO: better way to check whether we should perform retrieval?
        if self.retriever is not None:
            context = self.retriever.augmented_context(user_query=content)
            self._update_conversation(context, "assistant")

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
                # We need to catch these errors before they hit the server or else it will crash.
                # as of llama_cpp_python 0.3.z, BadRequestErrors cause the server to become unavailable
                if self.backend_type == backends.LLAMA_CPP:
                    total_context_size = 0
                    # this handling can apply to llama-cpp-python only. The exception handling below should still exist for vLLM.
                    while True:
                        # we need to loop this. If we only remove 1 message, but it isn't enough to fit our new message in, we can hit the error
                        # if you have 3 messages in the list, and the last one is 127 tokens with a 128 context window, you need to drop the first two in order to fit the third
                        for msg in self.info["messages"]:
                            total_context_size += len(msg["content"])
                        if (
                            self.max_ctx_size is not None
                            and self.max_ctx_size < total_context_size
                            and len(self.info["messages"]) > 1
                        ):
                            self.info["messages"] = self.info["messages"][1:]
                            logger.debug(
                                "Message too large for context size. Dropping from queue."
                            )
                        else:
                            break
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.info["messages"],
                        stream=True,
                        **create_params,
                    )
                except openai.BadRequestError as e:
                    # we still want to handle this the same for vLLM
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
            if self.box
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

                if self.box:
                    panel.subtitle = f"elapsed {time.time() - start_time:.3f} seconds"
            subtitle = f"elapsed {time.time() - start_time:.3f} seconds"

        # Update chat logs
        if subtitle is not None:
            self.log_message("- " + subtitle + " -\n")
        self.log_message(response_content.plain + "\n\n")
        # Update message history and token counters
        self._update_conversation(response_content.plain, "assistant")


def chat_model(
    question,
    model,
    model_id: str | None,
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
    rag_enabled,
    document_store_uri,
    collection_name,
    embedding_model_path,
    top_k,
    no_decoration,
    backend_type,
    host,
    port,
    current_max_ctx_size,
    params,
    backend_name,
    chat_template,
    api_base,
    gpu_layers,
    max_ctx_size,
    vllm_model_family,
    vllm_args,
    max_startup_attempts,
    logs_dir,
    vi_mode,
    visible_overflow,
    models_config: List[cfg._model_config],
):
    """Runs a chat using the modified model"""
    if rag_enabled and not FeatureGating.feature_available(GatedFeatures.RAG):
        logger.error(
            f'This functionality is experimental; run "export {FeatureGating.env_var_name}={FeatureScopes.DevPreviewNoUpgrade.value}" command to enable.'
        )
        return

    # set these variables here so we can override them if `--model-id` was provided
    if model_id:
        # try to load the model from the list
        try:
            model_cfg = cfg.resolve_model_id(model_id, models_config)
            if not model_cfg:
                raise ValueError(
                    f"Base model with ID '{model_id}' not found in the configuration."
                )
        except ValueError as ve:
            logger.error(f"failed to load model from ID: {ve}")
            raise

        model = resolve_model_path(model_cfg.path)

        # if loading a specific model from the list, we should only be loading from the tokenizer config
        chat_template = CHAT_TEMPLATE_TOKENIZER
        model_family = ""

    # pylint: disable=import-outside-toplevel
    # First Party
    from instructlab.model.backends.common import is_temp_server_running

    users_endpoint_url = cfg.get_api_base(host, port)

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
        max_ctx_size = current_max_ctx_size
    else:
        # If a log file is specified, write logs to the file
        root_logger = logging.getLogger()
        if serving_log_file:
            log.add_file_handler_to_logger(root_logger, serving_log_file)

        backend_instance = backends.get_backend_from_values(
            host=host,
            port=port,
            model_path=model,
            backend_name=backend_name,
            chat_template=chat_template,
            api_base=api_base,
            gpu_layers=gpu_layers,
            max_ctx_size=max_ctx_size,
            model_family=model_family,
            log_file=serving_log_file,
            vllm_model_family=vllm_model_family,
            vllm_args=vllm_args,
            max_startup_attempts=max_startup_attempts,
        )

        backend_type = backend_instance.get_backend_type()
        try:
            # Run the llama server
            api_base = backend_instance.run_detached(http_client(params))
        except (requests.RequestException, CalledProcessError, OSError) as exc:
            print(f"Failed to start server: {exc}", file=sys.stderr)
            sys.exit(1)

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
            and os.path.basename(model) == cfg.DEFAULTS.GRANITE_GGUF_MODEL_NAME
        ):
            logger.debug(
                "No model was provided by the user as a CLI argument or in the config, will use the model from the server"
            )
            try:
                models = ilabclient.list_models(
                    api_base=api_base,
                    http_client=http_client(params),
                )

                # Currently, we only present a single model so we can safely assume that the first model
                server_model = models.data[0].id if models is not None else None

                # override 'model' with the first returned model if not provided so that the chat print
                # the model used by the server
                model = (
                    server_model
                    if server_model is not None and server_model != model
                    else model
                )
                logger.debug(f"Using model from server {model}")
            except ilabclient.ClientException:
                print(
                    f"Failed to list models from {api_base}. Please check the API key and endpoint.",
                    file=sys.stderr,
                )
                # Right now is_temp_server() does not check if a subprocessed vllm is up
                # shut it down just in case an exception is raised in the try
                # TODO: revise is_temp_server to check if a vllm server is running
                if backend_instance is not None:
                    backend_instance.shutdown()
                sys.exit(1)

    try:
        chat_cli(
            api_base=api_base,
            logs_dir=logs_dir,
            vi_mode=vi_mode,
            visible_overflow=visible_overflow,
            question=question,
            model=model,
            context=context,
            session=session,
            qq=quick_question,
            max_tokens=max_tokens,
            max_ctx_size=max_ctx_size,
            temperature=temperature,
            rag_enabled=rag_enabled,
            document_store_uri=document_store_uri,
            collection_name=collection_name,
            embedding_model_path=embedding_model_path,
            top_k=top_k,
            backend_type=backend_type,
            params=params,
            no_decoration=no_decoration,
        )
    except ChatException as exc:
        print(f"{RED}Executing chat failed with: {exc}{RESET}")
        sys.exit(1)
    finally:
        if backend_instance is not None:
            backend_instance.shutdown()


def chat_cli(
    api_base,
    question,
    model,
    context,
    session,
    qq,
    max_tokens,
    max_ctx_size,
    temperature,
    backend_type,
    rag_enabled,
    document_store_uri,
    collection_name,
    embedding_model_path,
    top_k,
    logs_dir,
    vi_mode,
    visible_overflow,
    params,
    no_decoration,
):
    """Starts a CLI-based chat with the server"""
    client = OpenAI(
        base_url=api_base,
        api_key=params["api_key"],
        timeout=cfg.DEFAULTS.CONNECTION_TIMEOUT,
        http_client=http_client(params),
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

    # Instantiate retriever if RAG is enabled
    if rag_enabled:
        logger.debug("RAG enabled for chat; initializing retriever")
        retriever: DocumentStoreRetriever | None = create_document_retriever(
            document_store_uri=document_store_uri,
            document_store_collection_name=collection_name,
            top_k=top_k,
            embedding_model_path=embedding_model_path,
        )
    else:
        logger.debug("RAG not enabled for chat; skipping retrieval setup")
        retriever: DocumentStoreRetriever | None = None

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
    if logs_dir:
        date_suffix = (
            datetime.datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
        )
        os.makedirs(logs_dir, exist_ok=True)
        log_file = f"{logs_dir}/chat_{date_suffix}.log"

    # Initialize chat bot
    ccb = ConsoleChatBot(
        model if model is None else model,
        client=client,
        retriever=retriever,
        vi_mode=vi_mode,
        log_file=log_file,
        prompt=not qq,
        vertical_overflow=("visible" if visible_overflow else "ellipsis"),
        loaded=loaded,
        temperature=(temperature if temperature is not None else temperature),
        max_tokens=(max_tokens if max_tokens else max_tokens),
        max_ctx_size=max_ctx_size,
        backend_type=backend_type,
        box=not no_decoration,
    )

    if not qq and session is None:
        # Greet
        ccb.greet(help=True)

    # Use the input question to start with
    if len(question) > 0:
        question = " ".join(question)
        if not qq:
            print(f"{PROMPT_PREFIX}{question}")
        try:
            ccb.start_prompt(logger, content=question)
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
