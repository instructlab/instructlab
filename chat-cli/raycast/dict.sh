#!/bin/bash

# Required parameters:
# @raycast.schemaVersion 1
# @raycast.title Chat Dict
# @raycast.mode compact

# Optional parameters:
# @raycast.icon 📚
# @raycast.argument1 { "type": "text", "placeholder": "query", "optional": true}
# @raycast.packageName Chat Dict

# Documentation:
# @raycast.description Look up in Chat
# @raycast.author Kai Xu
# @raycast.authorURL xuk.ai

$HOME/bin/chat -qq --context dictionary "$1"
