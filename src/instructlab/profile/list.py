# Standard
from typing import Any, Dict, List
import os


def list_profiles(startpath: str) -> Dict[str, Any]:
    profiles: Dict[str, Any] = {}

    for root, _, files in os.walk(startpath):
        if root == startpath:
            continue

        path_parts = os.path.relpath(root, startpath).split(os.sep)

        current = profiles
        for part in path_parts:
            if part not in current:
                current[part] = {}
            current = current[part]

        if files:
            if "files" not in current:
                current["files"] = []
            current["files"].extend(files)

    return profiles


def format_output(profiles: Dict[str, Any], indent: str = "") -> List[str]:
    result: List[str] = []

    items = sorted(profiles.items(), key=lambda x: x[0])

    for idx, (category, content) in enumerate(items):
        is_last_item = idx == len(items) - 1

        if category == "files":
            result.append(f"{indent}└── {' '.join(content)}")
        else:
            result.append(f"{indent}{'└── ' if is_last_item else '├── '}{category}")
            child_indent = indent + ("    " if is_last_item else "│   ")
            result.extend(format_output(content, child_indent))

    return result
