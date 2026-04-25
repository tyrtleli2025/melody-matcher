"""Phase 0.3 — Evaluation metrics for query-by-humming retrieval.

All functions operate on a *ranked list* of retrieved song IDs (index 0 = top
result) and a *ground-truth song ID* (string).  The corpus may contain multiple
acceptable answers for one query in future, but the current QbH benchmark has
exactly one correct song per query.

Metric reference
----------------
MRR@k  (Mean Reciprocal Rank)
    Primary metric.  For a single query, the reciprocal rank is 1/r if the
    correct song appears at rank r ≤ k, or 0 otherwise.  Average over all
    queries to get MRR@k.

Hit@k
    Binary: 1 if the correct song appears anywhere in the top-k results.
    Average over all queries to get Hit@k.

Popularity-weighted Hit@10
    Hit@10 restricted to queries whose ground-truth song falls in the bottom
    50 % of the corpus by popularity.  Detects a ranker that "cheats" by
    always returning popular songs.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Optional


def reciprocal_rank(ranked_list: Sequence[str], ground_truth: str, k: int = 10) -> float:
    """Return the reciprocal rank of *ground_truth* in the top-k results.

    Args:
        ranked_list: Retrieved song IDs ordered by descending relevance.
        ground_truth: The single correct song ID for this query.
        k: Rank cutoff; results beyond position k are ignored.

    Returns:
        ``1 / r`` where ``r`` is the 1-based rank of *ground_truth*, or
        ``0.0`` if it does not appear in the top-k results or if
        *ranked_list* is empty.
    """
    if not ranked_list or not ground_truth:
        return 0.0

    for rank, song_id in enumerate(ranked_list[:k], start=1):
        if song_id == ground_truth:
            return 1.0 / rank

    return 0.0


def hit(ranked_list: Sequence[str], ground_truth: str, k: int) -> float:
    """Return 1.0 if *ground_truth* appears in the top-k results, else 0.0.

    Args:
        ranked_list: Retrieved song IDs ordered by descending relevance.
        ground_truth: The single correct song ID for this query.
        k: Rank cutoff.

    Returns:
        ``1.0`` on a hit, ``0.0`` on a miss.  Returns ``0.0`` for empty
        inputs rather than raising.
    """
    if not ranked_list or not ground_truth or k < 1:
        return 0.0

    return 1.0 if ground_truth in ranked_list[:k] else 0.0


# ---------------------------------------------------------------------------
# Corpus-level aggregation
# ---------------------------------------------------------------------------


def mrr_at_k(
    ranked_lists: Sequence[Sequence[str]],
    ground_truths: Sequence[str],
    k: int = 10,
) -> float:
    """Mean Reciprocal Rank at k over a set of queries.

    Args:
        ranked_lists: One ranked result list per query.
        ground_truths: Parallel sequence of correct song IDs.
        k: Rank cutoff applied to every query.

    Returns:
        Mean of per-query reciprocal ranks in [0.0, 1.0].  Returns ``0.0``
        when *ranked_lists* is empty.

    Raises:
        ValueError: If *ranked_lists* and *ground_truths* differ in length.
    """
    _check_parallel(ranked_lists, ground_truths)
    if not ranked_lists:
        return 0.0

    scores = [
        reciprocal_rank(rl, gt, k=k)
        for rl, gt in zip(ranked_lists, ground_truths)
    ]
    return statistics.mean(scores)


def hit_at_k(
    ranked_lists: Sequence[Sequence[str]],
    ground_truths: Sequence[str],
    k: int,
) -> float:
    """Hit rate at k over a set of queries.

    Args:
        ranked_lists: One ranked result list per query.
        ground_truths: Parallel sequence of correct song IDs.
        k: Rank cutoff.

    Returns:
        Fraction of queries for which the correct song is in the top-k
        results.  Returns ``0.0`` when *ranked_lists* is empty.

    Raises:
        ValueError: If *ranked_lists* and *ground_truths* differ in length.
    """
    _check_parallel(ranked_lists, ground_truths)
    if not ranked_lists:
        return 0.0

    scores = [hit(rl, gt, k=k) for rl, gt in zip(ranked_lists, ground_truths)]
    return statistics.mean(scores)


def popularity_weighted_hit_at_10(
    ranked_lists: Sequence[Sequence[str]],
    ground_truths: Sequence[str],
    popularity_scores: dict[str, float],
) -> float:
    """Hit@10 restricted to queries whose ground truth is in the bottom 50 % by popularity.

    This metric reveals whether the ranker is secretly cheating by returning
    hits — a system that always surfaces popular songs will score well on
    standard Hit@10 but badly here.

    Popularity scores must be in [0, 1] (normalised log-scores as stored in
    ``songs.parquet``).  Songs absent from *popularity_scores* are assumed to
    have popularity 0.0 (treated as unpopular).

    Args:
        ranked_lists: One ranked result list per query.
        ground_truths: Parallel sequence of correct song IDs.
        popularity_scores: Mapping of song ID → normalised popularity in [0, 1].

    Returns:
        Hit@10 computed only over the unpopular-ground-truth queries.  Returns
        ``0.0`` when no such queries exist or *ranked_lists* is empty.

    Raises:
        ValueError: If *ranked_lists* and *ground_truths* differ in length.
    """
    _check_parallel(ranked_lists, ground_truths)
    if not ranked_lists:
        return 0.0

    if popularity_scores:
        threshold = statistics.median(popularity_scores.values())
    else:
        threshold = 0.0

    filtered: list[tuple[Sequence[str], str]] = [
        (rl, gt)
        for rl, gt in zip(ranked_lists, ground_truths)
        if popularity_scores.get(gt, 0.0) <= threshold
    ]

    if not filtered:
        return 0.0

    scores = [hit(rl, gt, k=10) for rl, gt in filtered]
    return statistics.mean(scores)


def compute_all(
    ranked_lists: Sequence[Sequence[str]],
    ground_truths: Sequence[str],
    popularity_scores: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Convenience wrapper — compute the full standard metric suite at once.

    Args:
        ranked_lists: One ranked result list per query.
        ground_truths: Parallel sequence of correct song IDs.
        popularity_scores: Optional mapping used for the popularity-weighted
            metric.  When ``None``, that metric is omitted from the output.

    Returns:
        Dict with keys ``mrr@10``, ``hit@1``, ``hit@5``, ``hit@10``, and
        (conditionally) ``pop_weighted_hit@10``.

    Raises:
        ValueError: If *ranked_lists* and *ground_truths* differ in length.
    """
    _check_parallel(ranked_lists, ground_truths)

    results: dict[str, float] = {
        "mrr@10": mrr_at_k(ranked_lists, ground_truths, k=10),
        "hit@1": hit_at_k(ranked_lists, ground_truths, k=1),
        "hit@5": hit_at_k(ranked_lists, ground_truths, k=5),
        "hit@10": hit_at_k(ranked_lists, ground_truths, k=10),
    }

    if popularity_scores is not None:
        results["pop_weighted_hit@10"] = popularity_weighted_hit_at_10(
            ranked_lists, ground_truths, popularity_scores
        )

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_parallel(a: Sequence[object], b: Sequence[object]) -> None:
    if len(a) != len(b):
        raise ValueError(
            f"ranked_lists and ground_truths must have the same length, "
            f"got {len(a)} and {len(b)}."
        )
