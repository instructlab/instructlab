#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: The InstructLab Authors
# SPDX-License-Identifier: Apache-2.0
#
# Simple script to add SPDX copyright notice header lines to new files
# using reuse.
#
# The script expects at least one argument (i.e., the FILE).
# It can accept more arguments, that will be passed as CLI options
# to reuse, but does not check for the validity of any argument.
# Therefore wrong arguments may trigger reuse errors.
#
# Default options:
# - Use 'The InstructLab Authors' as default Copyright holder.
# - Apache-2.0 as default license for new files in the project.
# - Use cli-specific notice template.
# - Exclude years, as they are not required and makes it easier to mantain.
#
# Usage:
#
#   scripts/add-spdx-header.sh [OPTIONS] FILE
#
# Examples:
#
#   scripts/add-spdx-header.sh new_file.py
#   scripts/add-spdx-header.sh --style html new_file
#   scripts/add-spdx-header.sh --explicit-license new_file.txt


if [ -z "$1" ]; then
    echo "ERROR: File missing"
    echo "Usage:"
    echo "  $0 [OPTIONS] FILE"
    exit 1
fi

reuse annotate \
    --copyright="The InstructLab Authors" \
    --license="Apache-2.0" \
    --template=cli.jinja2 \
    --exclude-year \
    "$@"