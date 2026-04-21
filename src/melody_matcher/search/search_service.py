"""Exact and fuzzy search over the melody inverted index."""

from __future__ import annotations

from typing import Any

from melody_matcher.features.interval_encoder import encode_intervals
from melody_matcher.preprocessing.segmenter import create_segments

Index = dict[str, list[dict[str, Any]]]

# A result dict always has these keys plus an added "score" (0.0–1.0).
SearchResult = dict[str, Any]


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def _levenshtein(a: list[int], b: list[int]) -> int:
    """Standard dynamic-programming Levenshtein distance on integer sequences."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    # Two-row rolling array — O(min(la,lb)) space.
    if la < lb:
        a, b, la, lb = b, a, lb, la

    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr

    return prev[lb]


def _similarity(a: list[int], b: list[int]) -> float:
    """Normalised similarity in [0.0, 1.0]; 1.0 means identical."""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - _levenshtein(a, b) / max_len


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_exact_match(query_notes: list[str], index: Index) -> list[SearchResult]:
    """Return all index entries whose interval signature exactly matches the query.

    Args:
        query_notes: Ordered pitch strings (e.g. ``["C4", "D4", "E4"]``).
        index: Inverted index from ``index_store.load_index``.

    Returns:
        List of match dicts (``{"file", "segment_index"}``), unranked.
        Empty list when there is no exact match or fewer than 2 notes.

    Raises:
        TypeError: Wrong argument types.
        ValueError: Invalid note string.
    """
    if not isinstance(query_notes, list):
        raise TypeError(f"'query_notes' must be a list, got {type(query_notes).__name__}.")
    if not isinstance(index, dict):
        raise TypeError(f"'index' must be a dict, got {type(index).__name__}.")

    intervals = encode_intervals(query_notes)
    if not intervals:
        return []

    return list(index.get(str(intervals), []))


def search_fuzzy_match(
    query_notes: list[str],
    index: Index,
    *,
    top_n: int = 20,
    min_score: float = 0.5,
    segment_length: int | None = None,
) -> list[SearchResult]:
    """Search the index with fuzzy interval matching and confidence scoring.

    Strategy
    --------
    1. If the query is longer than ``segment_length``, it is split into
       overlapping sub-queries via ``create_segments``; each sub-query is
       searched independently. This mirrors how the index was built so that
       long hummed phrases still find partial hits.
    2. Every index key (an interval list) is compared against each sub-query
       using normalised Levenshtein similarity.
    3. Results are de-duplicated by ``(file, segment_index)`` — keeping only
       the highest score for each location — then sorted descending by score.

    Args:
        query_notes: Ordered pitch strings.
        index: Inverted index from ``index_store.load_index``.
        top_n: Maximum number of results to return.
        min_score: Minimum similarity threshold in ``[0, 1]``; hits below
            this are discarded.  Default ``0.5`` (50 %).
        segment_length: Window size used when building the index.  When
            ``None``, the most common key length in the index is inferred
            automatically.

    Returns:
        List of result dicts, each with keys ``"file"``, ``"segment_index"``,
        and ``"score"`` (float 0–1). Sorted by score descending.

    Raises:
        TypeError: Wrong argument types.
        ValueError: Invalid note strings.
    """
    if not isinstance(query_notes, list):
        raise TypeError(f"'query_notes' must be a list, got {type(query_notes).__name__}.")
    if not isinstance(index, dict):
        raise TypeError(f"'index' must be a dict, got {type(index).__name__}.")

    if not index:
        return []

    # --- Infer segment length from index when not supplied ---
    if segment_length is None:
        # The most frequent key length approximates the build-time segment_length
        # minus 1 (each key is a list of *intervals*, so one shorter than notes).
        from collections import Counter
        length_counts: Counter[int] = Counter()
        for key in index:
            try:
                length_counts[len(_parse_key(key))] += 1
            except ValueError:
                pass
        if not length_counts:
            return []
        inferred_interval_len = length_counts.most_common(1)[0][0]
        segment_length = inferred_interval_len + 1  # notes = intervals + 1

    # --- Build sub-queries ---
    if len(query_notes) <= segment_length:
        sub_queries = [query_notes]
    else:
        sub_queries = create_segments(query_notes, segment_length=segment_length)

    # Encode sub-queries; skip any that fail (e.g. bad note strings propagate
    # as ValueError from encode_intervals, so we let the first one raise).
    encoded_queries: list[list[int]] = []
    for i, sq in enumerate(sub_queries):
        intervals = encode_intervals(sq)
        if intervals:
            encoded_queries.append(intervals)

    if not encoded_queries:
        return []

    # --- Score every index key against every sub-query ---
    # best_score[(file, seg_idx)] = highest score seen so far
    best: dict[tuple[str, int], float] = {}

    parsed_index: list[tuple[list[int], list[dict[str, Any]]]] = []
    for raw_key, entries in index.items():
        try:
            parsed_index.append((_parse_key(raw_key), entries))
        except ValueError:
            continue

    for idx_intervals, entries in parsed_index:
        for q_intervals in encoded_queries:
            score = _similarity(q_intervals, idx_intervals)
            if score < min_score:
                continue
            for entry in entries:
                loc = (entry["file"], entry["segment_index"])
                if score > best.get(loc, -1.0):
                    best[loc] = score

    # --- Assemble, sort, truncate ---
    results: list[SearchResult] = [
        {"file": file, "segment_index": seg_idx, "score": score}
        for (file, seg_idx), score in best.items()
    ]
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_n]


def _parse_key(key: str) -> list[int]:
    """Parse a stringified interval list back to ``list[int]``.

    Raises:
        ValueError: If ``key`` is not a valid ``"[int, ...]"`` string.
    """
    stripped = key.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        raise ValueError(f"Unexpected index key format: {key!r}")
    inner = stripped[1:-1].strip()
    if not inner:
        return []
    try:
        return [int(x.strip()) for x in inner.split(",")]
    except ValueError as exc:
        raise ValueError(f"Non-integer element in index key: {key!r}") from exc