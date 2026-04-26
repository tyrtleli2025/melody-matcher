"""Key- and tempo-invariant melody normalisation.

This module is the single source of truth for the feature computation that
converts a raw :class:`~melody_matcher.preprocessing.melody_extractor.MelodyNote`
sequence into the representation stored in ``songs.parquet`` and used at
query time by :class:`~melody_matcher.retrieval.search_engine.SearchEngine`.

Both the corpus build script (``scripts/v2/build_corpus.py``) and the search
engine must call :func:`compute_features` so that query and corpus features are
always computed identically.
"""

from __future__ import annotations

import statistics
from melody_matcher.preprocessing.melody_extractor import MelodyNote


def compute_features(
    notes: list[MelodyNote],
) -> tuple[list[list[float]], str]:
    """Normalise a melody to key- and tempo-invariant features.

    Produces two representations from an ordered monophonic note sequence:

    **Intervals** — a list of ``[semitone_delta, normalised_IOI]`` pairs with
    one entry per consecutive note-pair (length ``N-1`` for ``N`` notes):

    * *semitone_delta*: ``pitch[i+1] - pitch[i]``.  Key-invariant because
      only relative pitch motion is stored; the absolute starting pitch is
      discarded.
    * *normalised IOI*: ``(onset[i+1] - onset[i]) / median_IOI``.
      Tempo-invariant because we divide by the per-melody median
      inter-onset interval.  The median is computed with
      :func:`statistics.median` (average of the two middle values for
      even-length lists).

    **Parsons code** — a string of ``U`` (up), ``D`` (down), ``R`` (repeat)
    characters derived from the sign of each semitone delta (also length
    ``N-1``).

    Args:
        notes: Ordered monophonic melody from
            :class:`~melody_matcher.preprocessing.melody_extractor.MelodyExtractor`.
            Must have at least 2 notes; returns empty results for shorter inputs.

    Returns:
        ``(intervals, parsons_code)`` where *intervals* is a list of
        ``[float, float]`` pairs and *parsons_code* is a string of U/D/R
        characters.  Both have length ``N-1``; both are empty for melodies
        with fewer than 2 notes.
    """
    if len(notes) < 2:
        return [], ""

    # Inter-onset intervals in beats.
    iois: list[float] = [
        notes[i + 1].start_beat - notes[i].start_beat
        for i in range(len(notes) - 1)
    ]

    median_ioi: float = statistics.median(iois)
    if median_ioi <= 0.0:
        median_ioi = 1.0

    norm_iois: list[float] = [ioi / median_ioi for ioi in iois]

    deltas: list[int] = [
        notes[i + 1].pitch - notes[i].pitch
        for i in range(len(notes) - 1)
    ]

    parsons: str = "".join(
        "U" if d > 0 else "D" if d < 0 else "R" for d in deltas
    )

    intervals: list[list[float]] = [
        [float(d), float(ioi)] for d, ioi in zip(deltas, norm_iois)
    ]

    return intervals, parsons
