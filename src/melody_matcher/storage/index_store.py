"""Persistence layer for the melody interval inverted index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Schema:
#   {
#     "[2, 2, 1, 2]": [{"file": "song.mid", "segment_index": 0}, ...],
#     ...
#   }
Index = dict[str, list[dict[str, Any]]]

_DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[4] / "data" / "index"
_DEFAULT_INDEX_FILE = _DEFAULT_INDEX_DIR / "melody_index.json"


def save_index(index: Index, path: str | Path = _DEFAULT_INDEX_FILE) -> Path:
    """Serialize the inverted index to a JSON file.

    The parent directory is created automatically if it does not exist.

    Args:
        index: The inverted index mapping stringified interval signatures to
            a list of occurrence dicts (``{"file": ..., "segment_index": ...}``).
        path: Destination file path. Defaults to ``data/index/melody_index.json``
            relative to the repository root.

    Returns:
        The resolved ``Path`` of the written file.

    Raises:
        TypeError: If ``index`` is not a dict.
        OSError: If the file cannot be written (permissions, disk full, etc.).
    """
    if not isinstance(index, dict):
        raise TypeError(f"'index' must be a dict, got {type(index).__name__}.")

    dest = Path(path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    with dest.open("w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)

    return dest


def load_index(path: str | Path = _DEFAULT_INDEX_FILE) -> Index:
    """Load the inverted index from a JSON file.

    Args:
        path: Source file path. Defaults to ``data/index/melody_index.json``
            relative to the repository root.

    Returns:
        The deserialized inverted index dict. Returns an empty dict if the
        file does not exist yet (first-run scenario).

    Raises:
        ValueError: If the file exists but cannot be parsed as valid JSON.
        OSError: If the file exists but cannot be read.
    """
    src = Path(path).resolve()

    if not src.exists():
        return {}

    try:
        with src.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Index file contains invalid JSON: {src}") from exc

    return data
