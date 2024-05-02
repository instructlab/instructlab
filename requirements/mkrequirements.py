#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create requirements.txt from base.txt and constraints.txt"""

# Standard
import pathlib
import typing

# Third Party
from packaging.requirements import Requirement

HERE = pathlib.Path(__file__).parent.absolute()
ROOT = HERE.parent
CONSTRAINTS = HERE / "constraints.txt"

FILE_MAP: typing.Dict[pathlib.Path, pathlib.Path] = {
    HERE / "base.txt":  ROOT / "requirements.txt",
    HERE / "hpu.txt": ROOT / "requirements-hpu.txt",
}

Parsed = typing.Iterable[typing.Tuple[str, typing.Optional[Requirement]]]
Constraints = typing.Dict[str, typing.List[Requirement]]


def parse_file(path: pathlib.Path) -> Parsed:
    """Parse a requirements or constraints text file"""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("-"):
                # ignore lines like -c, -r, --extra-index-url
                continue
            elif stripped and not stripped.startswith("#"):
                yield line, Requirement(stripped)
            else:
                # comment or empty line
                yield line, None


def parse_constraints(path: pathlib.Path) -> Constraints:
    """Parse and check constraints"""
    constraints: Constraints = {}
    for _, con in parse_file(path):
        if con is not None:
            if con.extras:
                raise ValueError(f"Extras belong into requirements: {con}")
            constraints.setdefault(con.name, []).append(con)
    return constraints


def mkrequirements(
    src: pathlib.Path, constraints: Constraints
) -> typing.Iterable[str]:
    """Add constraints to requirements"""
    for line, req in parse_file(src):
        if req is not None:
            if req.specifier:
                raise ValueError(
                    f"Version specifiers belong into constraints.txt: {req}"
                )
            for con in constraints.get(req.name, ()):
                if req.marker and con.marker:
                    # Let's figure this out when we really, really need it.
                    raise NotImplementedError(
                        f"constraint and requirement have markers: {req}, {con}"
                    )
                # copy environment markers and extra dependencies
                if req.marker:
                    con.marker = req.marker
                con.extras = req.extras
                yield str(con)
        else:
            yield line.strip()


def main():
    constraints = parse_constraints(CONSTRAINTS)
    for src, dest in FILE_MAP.items():
        with dest.open("w", encoding="utf-8") as f:
            for line in mkrequirements(src, constraints):
                f.write(line)
                f.write("\n")


if __name__ == "__main__":
    main()
