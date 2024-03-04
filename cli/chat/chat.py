#!/usr/bin/env python

# Standard
import atexit
import json
import os
import sys
import time

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
import openai

HELP_MD = """
Help / TL;DR
- `/q`: **q**uit
- `/h`: show **h**elp
- `/a assistant`: **a**mend **a**ssistant (i.e., model)
- `/c context`: **c**hange **c**ontext
- `/m`: toggle **m**ultiline (for the next session only)
- `/M`: toggle **m**ultiline
- `/n`: **n**ew session
- `/N`: **n**ew session (ignoring loaded)
- `/d [1]`: **d**isplay previous response
- `/p [1]`: previous response in **p**lain text
- `/md [1]`: previous response in **M**ark**d**own
- `/s filepath`: **s**ave current session to `filepath`
- `/l filepath`: **l**oad `filepath` and start a new session
- `/L filepath`: **l**oad `filepath` (permanently) and start a new session

Press Meta+Enter or Esc Enter to end multiline input.
"""

CONTEXTS = {
    "default": "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.",
    "cli_helper": "You are an expert for command line interface and know all common commands. Answer the command to execute as it without any explanation.",
}

PROMPT_HISTORY_FILEPATH = os.path.expanduser("~/.local/chat-cli.history")

PRICING_RATE = {
    "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
    "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
}

PROMPT_PREFIX = ">>> "


class ChatException(Exception):
    """An exception raised during chat step."""


# TODO Autosave chat history
class ConsoleChatBot:
    def __init__(
        self,
        model,
        client,
        vi_mode=False,
        prompt=True,
        vertical_overflow="ellipsis",
        loaded={},
    ):
        self.client = client
        self.model = model
        self.vi_mode = vi_mode
        self.vertical_overflow = vertical_overflow
        self.loaded = loaded

        self.console = Console()
        self.input = (
            PromptSession(history=FileHistory(PROMPT_HISTORY_FILEPATH))
            if prompt
            else None
        )
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

    def greet(self, help=False, new=False, session_name="new session"):
        side_info_str = (" (type `/h` for help)" if help else "") + (
            f" ({session_name})" if new else ""
        )
        self._sys_print(
            Markdown(f"Welcome to Chat CLI w/ **{self.model.upper()}**" + side_info_str)
        )

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

    def _handle_quit(self, content):
        raise EOFError

    def _handle_help(self, content):
        self._sys_print(Markdown(HELP_MD))
        raise KeyboardInterrupt

    def _handle_multiline(self, content):
        temp = content == "/m"  # soft multilien only for next prompt
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
        self.loaded["name"] = context
        self.loaded["messages"] = [{"role": "system", "content": CONTEXTS[context]}]
        self._reset_session()
        self.greet(new=True)
        raise KeyboardInterrupt

    def _handle_new_session(self, content):
        hard = content == "/N"  # hard new ignores loaded context/session
        self._reset_session(hard=hard)
        self.greet(new=True)
        raise KeyboardInterrupt

    def __handle_replay(self, content, display_wrapper=(lambda x: x)):
        cs = content.split()
        i = 1 if len(cs) == 1 else int(cs[1]) * 2 - 1
        if len(self.info["messages"]) > i:
            self.console.print(display_wrapper(self.info["messages"][-i]["content"]))
        raise KeyboardInterrupt

    def _handle_display(self, content):
        return self.__handle_replay(content, display_wrapper=(lambda x: Panel(x)))

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
        with open(filepath, "w") as outfile:
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
        with open(filepath, "r") as session:
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
        raise KeyboardInterrupt

    def _handle_empty():
        raise KeyboardInterrupt

    def _update_conversation(self, content, role):
        assert role in ("user", "assistant")
        message = {"role": role, "content": content}
        self.info["messages"].append(message)

    def start_prompt(self, content=None, box=True):
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

        # Update message history and token counters
        self._update_conversation(content, "user")

        # Deal with temp multiline
        if self.multiline_mode == 2:
            self.multiline_mode = 0
            self.multiline = not self.multiline

        # Get and parse response
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=self.info["messages"], stream=True
            )
            assert (
                next(response).choices[0].delta.role == "assistant"
            ), 'first response should be {"role": "assistant"}'
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
            self.console.print("Connection error, try again...", style="red bold")
            self.info["messages"].pop()
            raise KeyboardInterrupt
        except:
            self.console.print("Unknown error", style="bold red")
            raise ChatException(f"Unknown error: {sys.exc_info()[0]}")

        response_content = Text()
        panel = (
            Panel(response_content, title=self.model, subtitle_align="right")
            if box
            else response_content
        )
        with Live(
            panel,
            console=self.console,
            refresh_per_second=5,
            vertical_overflow=self.vertical_overflow,
        ) as live:
            start_time = time.time()
            for chunk in response:
                chunk_message = chunk.choices[0].delta
                if chunk_message.content:
                    response_content.append(chunk_message.content)

                if box:
                    panel.subtitle = f"elapsed {time.time() - start_time:.3f} seconds"

        # Update message history and token counters
        self._update_conversation(response_content.plain, "assistant")


def chat_cli(config, logger, question, model, context, session, qq) -> None:
    assert (context is None) or (
        session is None
    ), "Cannot load context and session in the same time"

    if not config.api_key:
        raise ChatException("Invalid API Key. Please set it in your config file.")

    # proxy = config.get("proxy")
    # if proxy is not None:
    # TODO: The 'openai.proxy' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(proxy={"http": proxy, "https": proxy})'
    # openai.proxy = {"http": proxy, "https": proxy}

    client = OpenAI(base_url=config.api_base, api_key=config.api_key)

    # Load context/session
    loaded = {}

    # Context config file
    # global CONTEXTS
    # CONTEXTS = config["contexts"]
    if context not in CONTEXTS:
        logger.info(f"Context {context} not found in the config file. Using default.")
        context = "default"
    loaded["name"] = context
    loaded["messages"] = [{"role": "system", "content": CONTEXTS[context]}]

    # Session from CLI
    # TODO Print history in session when loaded
    if session is not None:
        loaded["name"] = os.path.basename(session.name).strip(".json")
        loaded["messages"] = json.loads(session.read())

    # Initialize chat bot
    ccb = ConsoleChatBot(
        config.model if model is None else model,
        client=client,
        vi_mode=config.vi_mode,
        prompt=not qq,
        vertical_overflow=("visible" if config.visible_overflow else "ellipsis"),
        loaded=loaded,
    )

    if not qq:
        # Greet
        ccb.greet(help=True)

    # Use the input question to start with
    if len(question) > 0:
        question = " ".join(question)
        if not qq:
            print(f"{PROMPT_PREFIX}{question}")
        try:
            ccb.start_prompt(question, box=(not qq))
        except ChatException as exc:
            raise ChatException(f"API issue found while executing chat: {exc}")
        except KeyboardInterrupt:
            return

    if qq:
        return

    # Start chatting
    while True:
        try:
            ccb.start_prompt()
        except KeyboardInterrupt:
            continue
        except ChatException as exc:
            raise ChatException(f"API issue found while executing chat: {exc}")
