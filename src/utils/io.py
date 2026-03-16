"""I/O utility functions."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Dict[str, Any]:
    """Read JSON file."""
    with open(path) as f:
        return json.load(f)


def write_json(data: Dict[str, Any], path: str | Path, indent: int = 2) -> None:
    """Write data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def get_file_size_mb(path: str | Path) -> float:
    """Get file size in megabytes."""
    return Path(path).stat().st_size / (1024 * 1024)


def copy_tree(src: str | Path, dst: str | Path) -> None:
    """Copy directory tree."""
    shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
