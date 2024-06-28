#!/usr/bin/env python

# Standard
import subprocess
import sys

CMDS = {
    "config": [
        "init",
    ],
    "data": [
        "generate",
    ],
    "model": [
        "chat",
        "convert",
        "download",
        "serve",
        "test",
        "train",
    ],
    "sysinfo": [],
    "taxonomy": [
        "diff",
    ],
}


def heading(text, level):
    h = "#" * level
    print(f"{h} {text}\n", flush=True)


def run(cmd, final=False):
    print(f"```text", flush=True)
    subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)
    print("```%s" % ("" if final else "\n"), flush=True)


def main(argv=None):
    if argv is None:
        argv = sys.argv

    heading("`ilab` Command Reference", level=1)
    run(["ilab", "--help"])

    groups = sorted(CMDS.keys())

    # Table of Contents
    for group in groups:
        cmds = CMDS[group]
        print(
            f"* [`ilab {group}` Commands](#ilab-{group.lower()}-commands)", flush=True
        )
        for cmd in cmds:
            print(
                f"  * [`ilab {group} {cmd}` Command](#ilab-{group.lower()}-{cmd.lower()}-command)",
                flush=True,
            )
    print()

    for group in groups:
        cmds = CMDS[group]
        heading(f"`ilab {group}` Commands", level=2)
        run(["ilab", group, "--help"])
        for cmd in cmds:
            heading(f"`ilab {group} {cmd}` Command", level=3)
            run(
                ["ilab", group, cmd, "--help"],
                final=True if group == groups[-1] and cmd == cmds[-1] else False,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
