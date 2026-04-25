"""Loaders for the Phase 0 frozen evaluation datasets.

MIR-QBSH expected layout
--------------------------
    data/v2/eval/mir_qbsh/
        QBSHdata/
            <song_id>/          # folder name is the ground-truth song ID
                <query_id>.wav  # one or more hummed queries per song

The ground-truth song ID is derived from the immediate parent folder of each
query file, so the extraction layout must be preserved.

A query entry has the shape::

    {
        "query_audio_path": Path,   # absolute path to the .wav file
        "ground_truth_song_id": str # parent folder name
    }

The loader never raises on an empty or missing directory; it logs a warning
and returns an empty list so downstream code can decide whether to abort.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

# Supported audio extensions for query files.
_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".wav", ".mp3", ".m4a", ".flac"})
# Supported symbolic extensions (for MIDI-only corpora / future use).
_MIDI_EXTENSIONS: frozenset[str] = frozenset({".mid", ".midi"})
_ALL_EXTENSIONS: frozenset[str] = _AUDIO_EXTENSIONS | _MIDI_EXTENSIONS


@dataclass(frozen=True)
class EvalQuery:
    """A single evaluation query with its ground-truth label."""

    query_audio_path: Path
    ground_truth_song_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "query_audio_path": self.query_audio_path,
            "ground_truth_song_id": self.ground_truth_song_id,
        }


class MIR_QBSH_Loader:
    """Load hummed-query entries from an extracted MIR-QBSH dataset directory.

    Parameters
    ----------
    dataset_root:
        Path to ``data/v2/eval/mir_qbsh/``.  The loader descends into the
        ``QBSHdata/`` sub-directory automatically when it is present; if the
        directory is empty or missing it logs a warning and returns no entries.
    strict:
        When ``True``, raises ``FileNotFoundError`` if the directory does not
        exist.  When ``False`` (default) it logs a warning and returns ``[]``.
    """

    # MIR-QBSH stores queries under this sub-directory after extraction.
    _QBSH_SUBDIR = "QBSHdata"

    def __init__(self, dataset_root: str | Path, *, strict: bool = False) -> None:
        self._root = Path(dataset_root).resolve()
        self._strict = strict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> list[dict[str, object]]:
        """Return a list of query dicts, one per audio file found.

        Each dict has the shape::

            {
                "query_audio_path": Path,
                "ground_truth_song_id": str,
            }

        Returns an empty list (with a warning) when no files are found.
        """
        return [q.as_dict() for q in self._iter_queries()]

    def load_typed(self) -> list[EvalQuery]:
        """Same as :meth:`load` but returns :class:`EvalQuery` dataclasses."""
        return list(self._iter_queries())

    def __len__(self) -> int:
        return sum(1 for _ in self._iter_query_files())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_root(self) -> Path | None:
        """Resolve the directory that actually contains song-ID sub-folders.

        Prefers ``<root>/QBSHdata/`` when it exists; falls back to ``<root>``
        itself so the loader works with both the standard archive layout and a
        flat custom layout.
        """
        if not self._root.exists():
            msg = "MIR-QBSH dataset directory not found: %s"
            if self._strict:
                raise FileNotFoundError(msg % self._root)
            log.warning(msg, self._root)
            log.warning(
                "Download the dataset and extract it there.  "
                "Run `python scripts/v2/setup_eval.py` for instructions."
            )
            return None

        preferred = self._root / self._QBSH_SUBDIR
        if preferred.is_dir():
            return preferred

        # Fall back: if the root has immediate sub-directories that contain
        # audio files, treat the root itself as the songs directory.
        has_song_dirs = any(
            child.is_dir() and self._dir_has_audio(child)
            for child in self._root.iterdir()
        )
        if has_song_dirs:
            log.debug("Using %s as song root (no QBSHdata sub-dir found).", self._root)
            return self._root

        log.warning(
            "MIR-QBSH directory exists but appears empty or has an unexpected layout: %s",
            self._root,
        )
        log.warning(
            "Expected: %s/<song_id>/<query>.wav  or  %s/QBSHdata/<song_id>/<query>.wav",
            self._root,
            self._root,
        )
        return None

    def _iter_queries(self) -> Iterator[EvalQuery]:
        effective_root = self._effective_root()
        if effective_root is None:
            return

        found_any = False
        for query_path in self._iter_query_files(effective_root):
            song_id = query_path.parent.name
            if not song_id:
                log.warning("Could not determine song ID for %s — skipping.", query_path)
                continue
            found_any = True
            yield EvalQuery(
                query_audio_path=query_path,
                ground_truth_song_id=song_id,
            )

        if not found_any:
            log.warning(
                "No query files (%s) found under %s.",
                ", ".join(sorted(_ALL_EXTENSIONS)),
                effective_root,
            )
            log.warning(
                "If you haven't downloaded the dataset yet, run: "
                "python scripts/v2/setup_eval.py"
            )

    def _iter_query_files(self, root: Path | None = None) -> Iterator[Path]:
        """Yield all audio/MIDI files one level below the song-ID folders."""
        search_root = root or self._effective_root()
        if search_root is None:
            return

        for song_dir in sorted(search_root.iterdir()):
            if not song_dir.is_dir():
                continue
            for candidate in sorted(song_dir.iterdir()):
                if candidate.is_file() and candidate.suffix.lower() in _ALL_EXTENSIONS:
                    yield candidate

    @staticmethod
    def _dir_has_audio(directory: Path) -> bool:
        return any(
            f.suffix.lower() in _ALL_EXTENSIONS
            for f in directory.iterdir()
            if f.is_file()
        )
