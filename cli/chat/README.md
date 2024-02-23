# CLI Chat

## Prerequsites

1. Copy [chat-cli.toml](/cli/chat/chat-cli.toml) to `~/.config/chat-cli.toml` (or `~/.chat-cli.toml`)
2. Update `api_key` in the file ([guide](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key) to get the key)

## Usage

### Command line interface

```sh
❯ python -m cli chat --help
```

```
Usage: chat.py [OPTIONS] [QUESTION]...

Options:
  -m, --model TEXT        Model to use
  -c, --context TEXT      Name of system context in config file
  -s, --session FILENAME  Filepath of a dialog session file
  -qq, --quick-question   Exit after answering question
  --help                  Show this message and exit.
```

### Start chatting

```sh
❯ python -m cli chat
```

```
╭───────────────────────────────────────── system ─────────────────────────────────────────╮
│ Welcome to Chat CLI w/ GPT-3.5-TURBO (type /h for help)                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
```

### Commands in chat
```
>>> /h                                                                       [0][S][default]
```

```
╭───────────────────────────────────────── system ─────────────────────────────────────────╮
│ Help / TL;DR                                                                             │
│                                                                                          │
│  • /q: quit                                                                              │
│  • /h: show help                                                                         │
│  • /a model: amend model                                                                 │
│  • /m: toggle multiline (for the next session only)                                      │
│  • /M: toggle multiline                                                                  │
│  • /n: new session                                                                       │
│  • /N: new session (ignoring loaded)                                                     │
│  • /d [1]: display previous response                                                     │
│  • /p [1]: previous response in plain text                                               │
│  • /md [1]: previous response in Markdown                                                │
│  • /s filepath: save current session to filepath                                         │
│  • /l filepath: load filepath and start a new session                                    │
│  • /L filepath: load filepath (permanently) and start a new session                      │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
```

### ACL: Ask Command Line

1. Make `./bin/chat` and `./bin/ask-cl` avaiable from your PATH (e.g. by making symlinks)
2. `ask-cl some natural langauge` will convert the natural language to the command line
3. Optionally, if you are using the Fish shell, add below to your config file
```fish
function acl
    if isatty stdin
        set cmd (ask-cl $argv)
    else
        read -l -z pipin
        set cmd (echo $pipin | ask-cl $argv)
    end
    commandline -i $cmd
end
```
and you can use `acl some natural langauge` which will do the same *AND* copy paste the command to your shell for you.
4. Both `ask-cl` or `acl` accepts stdin so you can pipe commands to them, e.g.
``` fish
ls | acl back up the TOML file with a prefix "old"
```

## Acknowledgements

- Heavily inspired by https://github.com/marcolardera/chatgpt-cli
- Tested by Pinzhen and Bing
