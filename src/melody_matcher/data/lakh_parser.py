"""Parser for the LMD-matched subset of the Lakh MIDI Dataset.

Expected directory layout::

    lmd_matched/
        A/A/A/TRAAAAV128F421A322/
            5c5e7099b7dbac0f00ff2b7b.mid
            a7b3f1c2d4e5a6b7c8d9e0f1.mid
        A/A/B/TRAABXG128F429B4A9/
            3d4e5f6a7b8c9d0e1f2a3b4c.mid
        ...

Each leaf folder is a Million Song Dataset track ID (``TR`` + 16 uppercase
alphanumerics).  It may contain **one or more** MIDI files, each named by
its md5 hash.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_MSD_ID_RE: re.Pattern[str] = re.compile(r"^TR[A-Z0-9]{16}$")

_MIDI_EXTENSIONS: frozenset[str] = frozenset({".mid", ".midi"})

# Files to skip unconditionally (beyond the hidden-file rule).
_SKIP_NAMES: frozenset[str] = frozenset({"Thumbs.db", ".DS_Store"})

# Directories whose contents are always skipped.
_SKIP_DIRS: frozenset[str] = frozenset({"__MACOSX"})


class LakhMatchedParser:
    """Scan an LMD-matched directory tree and yield per-file metadata dicts.

    Args:
        root: Path to the top-level ``lmd_matched/`` directory.

    Raises:
        NotADirectoryError: If ``root`` does not exist or is not a directory.
    """

    def __init__(self, root: str | Path) -> None:
        resolved = Path(root).resolve()
        if not resolved.is_dir():
            raise NotADirectoryError(
                f"LMD root is not a directory (or does not exist): {resolved}\n"
                "Download lmd_matched.tar.gz from https://colinraffel.com/projects/lmd/ "
                "and extract it there."
            )
        self._root: Path = resolved
        self._skipped_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def iter_files(self) -> Iterator[dict[str, Any]]:
        """Yield one metadata dict per valid MIDI file found under ``root``.

        The generator never materialises the full file list in memory; it
        traverses the tree lazily using :py:meth:`pathlib.Path.rglob`.

        Yields:
            A dict with the following keys:

            * ``msd_id`` (str): The MSD track ID derived from the parent
              folder name (e.g. ``"TRAAAAV128F421A322"``).
            * ``midi_path`` (Path): Absolute, symlink-resolved path to the
              MIDI file.
            * ``md5`` (str): The md5 hash from the filename stem (e.g.
              ``"5c5e7099b7dbac0f00ff2b7b"``).

        Raises:
            Nothing — all per-file errors are logged at WARNING level and
            the file is skipped.
        """
        self._skipped_count = 0

        for path in self._root.rglob("*"):
            if not path.is_file():
                continue

            # Only process MIDI candidates; everything else is ignored silently.
            if path.suffix.lower() not in _MIDI_EXTENSIONS:
                continue

            # --- Hidden files and known system artifacts ---
            if path.name.startswith(".") or path.name in _SKIP_NAMES:
                self._skipped_count += 1
                continue

            # --- macOS archive artifacts anywhere in the path ---
            if _SKIP_DIRS.intersection(path.parts):
                self._skipped_count += 1
                continue

            # --- Symlink-escape guard ---
            try:
                resolved = path.resolve()
            except OSError as exc:
                self._skipped_count += 1
                log.warning("Could not resolve %s (%s) — skipping.", path, exc)
                continue

            if not resolved.is_relative_to(self._root):
                self._skipped_count += 1
                log.warning(
                    "Symlink %s escapes root (%s) — skipping.", path, self._root
                )
                continue

            # --- MSD ID validation ---
            msd_id = path.parent.name
            if not _MSD_ID_RE.fullmatch(msd_id):
                self._skipped_count += 1
                log.warning(
                    "Non-MSD parent folder %r for file %s — skipping.", msd_id, path
                )
                continue

            yield {
                "msd_id": msd_id,
                "midi_path": resolved,
                "md5": path.stem,
            }

    def group_by_msd_id(self) -> dict[str, list[Path]]:
        """Drain :meth:`iter_files` and group MIDI paths by MSD track ID.

        Returns:
            A dict mapping each MSD track ID to the list of absolute
            :class:`~pathlib.Path` objects for its MIDI files.  The dict
            and each inner list preserve insertion order (i.e. the
            filesystem traversal order of :meth:`iter_files`).

        Note:
            Calling this method resets :attr:`skipped_count` as a side
            effect of re-running the generator.
        """
        grouped: dict[str, list[Path]] = {}
        for entry in self.iter_files():
            grouped.setdefault(entry["msd_id"], []).append(entry["midi_path"])
        return grouped

    @property
    def skipped_count(self) -> int:
        """Number of MIDI-extension files skipped during the last :meth:`iter_files` traversal."""
        return self._skipped_count
