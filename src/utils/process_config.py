import json
import os.path as osp
from pathlib import Path

# Repo root = three levels up from this file (src/utils/process_config.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_relative_path(path: str) -> str:
    """
    Resolves a possibly-relative path against the repository root.

    Absolute paths (and user paths like ``~/data``) are returned unchanged so
    that users can still point ``storage_dir`` at an arbitrary location. Relative
    paths are anchored to the repo root so they resolve consistently regardless
    of the current working directory the script/notebook is launched from.
    """
    expanded = osp.expanduser(path)
    if osp.isabs(expanded):
        return expanded
    return str((REPO_ROOT / expanded).resolve())


def load_config(config_path: str) -> dict:
    """
    Loads a configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration.
    """
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    storage_dir = config.get("storage_dir")
    if isinstance(storage_dir, dict) and "path" in storage_dir:
        storage_dir["path"] = _resolve_relative_path(storage_dir["path"])

    return config
