"""Baseline melody search engine (Phase 2).

Architecture
------------
Loading
    The corpus (``songs.parquet``) is read into a Pandas DataFrame once at
    construction time.  Per-song interval delta arrays are pre-decoded from
    JSON into ``numpy`` float32 arrays so the DTW scoring loop never touches
    JSON parsing.  A Parsons n-gram inverted index (n=8) is built over the
    full corpus for fast candidate pre-filtering.

Search pipeline
    1. **Normalise** — :func:`~melody_matcher.preprocessing.normalizer.compute_features`
       converts the query :class:`MelodyNote` list into a semitone-delta array
       and a Parsons contour string using the exact same math as the corpus
       build step.
    2. **Pre-filter** — the Parsons 8-gram inverted index scores every corpus
       song by how many length-8 Parsons substrings it shares with the query.
       The top ``max_candidates`` songs (default 500) are forwarded to DTW.
       This reduces the DTW workload by ~97 % (500 / 16 739).
    3. **DTW scoring** — :func:`fastdtw.fastdtw` computes the normalised
       dynamic time-warping distance between the query's semitone-delta
       sequence and each candidate's.  The distance is normalised by
       ``min(len_query, len_candidate)`` and converted to a similarity
       score in (0, 1] via ``1 / (1 + dist)``.
    4. **Combined ranking** — ``combined = 0.75 × dtw_sim + 0.25 × popularity``
       (popularity already normalised to [0, 1] in the corpus).  Results are
       sorted descending and the top ``top_k`` are returned.

Why n=8 for the Parsons index?
    The Parsons alphabet has only three symbols (U / D / R).  Short n-grams
    (n ≤ 5) are nearly non-discriminative — n=5 has an average posting-list
    size of ~29 000 across a 16 000-song corpus.  At n=8 (3^8 = 6 561 unique
    grams) the average posting list shrinks to ~900, making the overlap score
    a meaningful signal.  A 27-character query contributes 20 distinct 8-grams;
    the whole scoring step takes microseconds with ``numpy.bincount``.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from tqdm import tqdm

from melody_matcher.preprocessing.melody_extractor import MelodyNote
from melody_matcher.preprocessing.normalizer import compute_features

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CORPUS: Path = _REPO_ROOT / "data" / "v2" / "processed" / "songs.parquet"

_NGRAM_N: int = 8            # Parsons n-gram length for inverted index
_DEFAULT_MAX_CANDIDATES: int = 500   # songs forwarded from pre-filter to DTW
_MIN_QUERY_NOTES: int = 2    # below this, search returns immediately

_W_DTW: float = 0.75         # weight for DTW similarity in combined score
_W_POP: float = 0.25         # weight for popularity in combined score

_EMPTY_IDX: np.ndarray = np.array([], dtype=np.int32)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """One ranked search result.

    Attributes:
        rank: 1-based position in the result list.
        song_id: MSD track ID (``TRXXXXXXXXXXXXXXXX``).
        popularity_log: Normalised log-playcount in [0.0, 1.0].
        dtw_similarity: Melodic similarity in (0.0, 1.0]; higher is better.
        combined_score: Weighted combination of similarity and popularity.
        parsons_code: Corpus song's full Parsons contour (useful for debug).
        source_midi_path: Absolute path to the source MIDI file.
    """

    rank: int
    song_id: str
    popularity_log: float
    dtw_similarity: float
    combined_score: float
    parsons_code: str
    source_midi_path: str


# ---------------------------------------------------------------------------
# Search engine
# ---------------------------------------------------------------------------

class SearchEngine:
    """Baseline melody search engine over a pre-built ``songs.parquet`` corpus.

    Args:
        corpus_path: Path to ``songs.parquet``.  Defaults to the project's
            standard location under ``data/v2/processed/``.
        max_candidates: Maximum number of pre-filtered candidates forwarded to
            the DTW scoring loop (default 500).

    Raises:
        FileNotFoundError: If *corpus_path* does not exist.
    """

    def __init__(
        self,
        corpus_path: Path | str = _DEFAULT_CORPUS,
        *,
        max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    ) -> None:
        corpus_path = Path(corpus_path)
        if not corpus_path.is_file():
            raise FileNotFoundError(
                f"Corpus not found: {corpus_path}\n"
                "Run scripts/v2/build_corpus.py first."
            )

        log.info("Loading corpus from %s …", corpus_path)
        self._df: pd.DataFrame = pd.read_parquet(corpus_path)
        self._max_candidates = max_candidates

        log.info("Decoding %d interval sequences …", len(self._df))
        # Precompute semitone-delta arrays (first element of each interval pair).
        self._delta_arrays: list[np.ndarray] = [
            np.array([pair[0] for pair in json.loads(s)], dtype=np.float32)
            for s in self._df["melody_intervals_json"]
        ]

        log.info("Building Parsons %d-gram inverted index …", _NGRAM_N)
        self._ngram_index: dict[str, np.ndarray] = self._build_ngram_index()
        log.info(
            "SearchEngine ready: %d songs, %d unique %d-grams.",
            len(self._df), len(self._ngram_index), _NGRAM_N,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query_melody: list[MelodyNote],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search the corpus for songs melodically similar to *query_melody*.

        Args:
            query_melody: Ordered monophonic melody from
                :class:`~melody_matcher.preprocessing.melody_extractor.MelodyExtractor`.
            top_k: Number of results to return (default 5).

        Returns:
            Up to *top_k* :class:`SearchResult` objects sorted by combined
            score (descending).  Returns an empty list if the query is too
            short or yields no usable features.
        """
        if len(query_melody) < _MIN_QUERY_NOTES:
            log.warning(
                "Query has only %d notes (minimum %d) — returning empty.",
                len(query_melody), _MIN_QUERY_NOTES,
            )
            return []

        # --- Normalise query ---
        intervals, query_parsons = compute_features(query_melody)
        if not intervals:
            log.warning("compute_features produced no intervals — returning empty.")
            return []

        query_arr = np.array([iv[0] for iv in intervals], dtype=np.float32)

        # --- Pre-filter via Parsons n-gram index ---
        candidate_idx = self._filter_candidates(query_parsons)
        log.debug(
            "Parsons pre-filter: %d candidates from %d songs.",
            len(candidate_idx), len(self._df),
        )

        if len(candidate_idx) == 0:
            log.warning("No candidates after Parsons pre-filter.")
            return []

        # --- DTW scoring ---
        scored: list[tuple[int, float]] = []
        for idx in tqdm(
            candidate_idx,
            desc="DTW scoring",
            unit="song",
            leave=False,
            dynamic_ncols=True,
        ):
            cand_arr = self._delta_arrays[idx]
            if len(cand_arr) == 0:
                continue
            sim = self._dtw_similarity(query_arr, cand_arr)
            scored.append((idx, sim))

        if not scored:
            return []

        # --- Combined ranking ---
        ranked: list[SearchResult] = []
        for idx, dtw_sim in scored:
            pop = float(self._df.at[idx, "popularity_log"])
            combined = _W_DTW * dtw_sim + _W_POP * pop
            ranked.append(
                SearchResult(
                    rank=0,   # filled in below
                    song_id=str(self._df.at[idx, "song_id"]),
                    popularity_log=pop,
                    dtw_similarity=dtw_sim,
                    combined_score=combined,
                    parsons_code=str(self._df.at[idx, "parsons_code"]),
                    source_midi_path=str(self._df.at[idx, "source_midi_path"]),
                )
            )

        ranked.sort(key=lambda r: r.combined_score, reverse=True)
        for i, result in enumerate(ranked[:top_k], start=1):
            result.rank = i

        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_ngram_index(self) -> dict[str, np.ndarray]:
        """Build a Parsons n-gram inverted index over the corpus.

        Returns:
            A dict mapping each n-gram string to a ``numpy.int32`` array of
            corpus row indices that contain that n-gram at least once.  Each
            index appears at most once per n-gram (set semantics).
        """
        n = _NGRAM_N
        # Collect posting lists as lists first, then convert to numpy arrays.
        raw: dict[str, list[int]] = defaultdict(list)
        seen: set[tuple[str, int]] = set()   # (gram, row_idx) → avoid duplication

        for row_idx, code in enumerate(self._df["parsons_code"]):
            code_len = len(code)
            for j in range(code_len - n + 1):
                gram = code[j: j + n]
                key = (gram, row_idx)
                if key not in seen:
                    seen.add(key)
                    raw[gram].append(row_idx)

        return {gram: np.array(indices, dtype=np.int32) for gram, indices in raw.items()}

    # ------------------------------------------------------------------
    # Pre-filtering
    # ------------------------------------------------------------------

    def _filter_candidates(self, query_parsons: str) -> np.ndarray:
        """Return candidate corpus indices sorted by Parsons 8-gram overlap.

        Scores each corpus song by how many of the query's length-8 Parsons
        substrings it contains (set-counted per song, not multiset).  The top
        ``max_candidates`` by score are returned.

        For queries shorter than ``_NGRAM_N``, falls back to an exact-substring
        scan so that short queries still produce candidates.

        Args:
            query_parsons: Parsons contour string for the query melody.

        Returns:
            Array of corpus row indices (dtype int32), ordered by descending
            n-gram overlap count, length ≤ ``max_candidates``.
        """
        n = _NGRAM_N

        if len(query_parsons) < n:
            return self._filter_by_substring(query_parsons)

        # Collect posting arrays for every query n-gram.
        parts: list[np.ndarray] = []
        for j in range(len(query_parsons) - n + 1):
            gram = query_parsons[j: j + n]
            arr = self._ngram_index.get(gram)
            if arr is not None:
                parts.append(arr)

        if not parts:
            # Zero query n-grams found in the index — fall back to substring scan.
            log.debug("No query n-grams in index; falling back to substring scan.")
            return self._filter_by_substring(query_parsons)

        # Count how many query n-grams each corpus song matches.
        all_hits = np.concatenate(parts)
        counts = np.bincount(all_hits, minlength=len(self._df))

        # Return top-k by count (all with count > 0).
        top = int(min(self._max_candidates, (counts > 0).sum()))
        if top == 0:
            return _EMPTY_IDX

        # argpartition is O(n) and faster than full sort for large arrays.
        if top < len(counts):
            top_idx = np.argpartition(counts, -top)[-top:]
        else:
            top_idx = np.where(counts > 0)[0]

        # Re-sort the selected candidates by descending count.
        order = np.argsort(counts[top_idx])[::-1]
        return top_idx[order].astype(np.int32)

    def _filter_by_substring(self, query_parsons: str) -> np.ndarray:
        """Fallback pre-filter: exact-substring scan over all Parsons codes.

        Used when the query is shorter than ``_NGRAM_N``.

        Args:
            query_parsons: Parsons contour to search for.

        Returns:
            Array of matching corpus row indices (dtype int32).
        """
        if not query_parsons:
            return _EMPTY_IDX
        matches = [
            i
            for i, code in enumerate(self._df["parsons_code"])
            if query_parsons in code
        ]
        return np.array(matches[: self._max_candidates], dtype=np.int32)

    # ------------------------------------------------------------------
    # DTW scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _dtw_similarity(query: np.ndarray, candidate: np.ndarray) -> float:
        """Compute normalised FastDTW similarity between two delta sequences.

        The raw DTW distance is normalised by ``min(len_query, len_candidate)``
        to make scores comparable across sequence pairs of different lengths.
        The normalised distance is then mapped to a similarity in (0, 1] via
        ``1 / (1 + distance)``.

        Args:
            query: 1-D float32 array of semitone deltas.
            candidate: 1-D float32 array of semitone deltas.

        Returns:
            Similarity score in (0.0, 1.0]; 1.0 indicates a perfect match.
        """
        raw_dist, _ = fastdtw(
            query, candidate,
            dist=lambda a, b: abs(float(a) - float(b)),
        )
        normalised = raw_dist / min(len(query), len(candidate))
        return 1.0 / (1.0 + normalised)
