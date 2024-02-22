#!/bin/bash

# Required parameters:
# @raycast.schemaVersion 1
# @raycast.title Chat
# @raycast.mode silent

# Optional parameters:
# @raycast.icon 🤖
# @raycast.argument1 { "type": "text", "placeholder": "query", "optional": true}
# @raycast.packageName Chat CLI

# Documentation:
# @raycast.description Chat in Terminal
# @raycast.author Kai Xu
# @raycast.authorURL xuk.ai

# Option 1: Creating a new icon in the dock
#wezterm --config "font_size=20" start $HOME/bin/chat "$1"
# Option 2: Reusing the exisiting icon in the dock
WEZTERM_UNIX_SOCKET=~/.local/share/wezterm/default-org.wezfurlong.wezterm wezterm cli spawn --new-window $HOME/bin/chat "$1"
