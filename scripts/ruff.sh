#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -e

# wrapper to combine ruff check, ruff format, and isort
#
# "ruff.sh fix" runs fixes and reformats the code
# "ruff.sh check" checks style, format, and isort
# "ruff.sh <args>" passes abitrary args to ruff

if [ -z "$1" ]; then
    echo "USAGE: $0 [check|fix|<args>]" >&2
    exit 2
fi

run() {
    declare -i err

    echo "RUN: '$*'"
    "$@"
    err=$?
    echo
    return $err
}

case $1 in
    "check")
        declare -i exitcode=0

        set +e
        run ruff check .
        exitcode=$(( exitcode + $? ))

        run ruff format --diff .
        exitcode=$(( exitcode + $? ))

        run isort --check --diff .
        exitcode=$(( exitcode + $? ))
        set -e

        if [ $exitcode -ne 0 ]; then
            echo "ERROR: one or more checks have failed." >&2
            echo "Run 'tox -e ruff' to auto-correct all fixable errors." >&2
            exit 3
        fi
        ;;
    "fix")
        run ruff check --fix .
        run ruff format .
        run isort .
        ;;
    *)
        ruff "$@"
esac
