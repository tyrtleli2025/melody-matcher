"""Phase 1.5/1.6/1.7 — Build the processed melody corpus (songs.parquet).

Pipeline
--------
For each MSD ID in lmd_manifest.json:

1.  Try each associated MIDI file with MelodyExtractor.  Use the first
    successful extraction that yields ≥ ``--min-notes`` notes.
2.  **Phase 1.5 — Normalise** the note sequence to key- and tempo-invariant
    features::

        intervals  = [(semitone_delta_i, normalised_IOI_i), ...]   # N-1 pairs
        parsons    = "UDUDR..."                                     # N-1 chars

    *Semitone delta*: ``pitch[i+1] - pitch[i]`` — key-invariant because only
    relative intervals are stored.
    *Normalised IOI*: ``(onset[i+1] - onset[i]) / median_IOI`` — tempo-invariant
    because we divide by the per-song median inter-onset interval.
    *Parsons code*: ``U`` (up), ``D`` (down), ``R`` (repeat) from the sign of
    each semitone delta — a coarse melodic contour.

3.  **Phase 1.6 — Deduplicate** using a two-stage pipeline:

    * *Exact*: songs whose serialised interval sequences are identical are
      collapsed immediately (keeps the entry with the higher popularity score).
    * *Near-duplicate*: within each Parsons-code group, pairwise DTW distance
      (normalised by sequence length) is computed; groups below
      ``--dtw-threshold`` are merged, keeping the most popular representative.
      Groups larger than ``_DTW_GROUP_CAP`` skip DTW to avoid O(n²) cost —
      they almost certainly contain genuinely distinct melodies.

4.  **Phase 1.7 — Persist** the result as ``songs.parquet`` with columns:

    =====================  =================================================
    ``song_id``            MSD track ID (``TRXXXXXXXXXXXXXXXX``)
    ``popularity_log``     Normalised log-playcount in [0.0, 1.0]
    ``melody_notes_json``  JSON array of ``[pitch, start_beat, dur_beats]``
    ``melody_intervals_json``  JSON array of ``[semitone_delta, norm_ioi]``
    ``parsons_code``       ``"UDUDR…"`` contour string
    ``source_midi_path``   Absolute path to the source MIDI file
    =====================  =================================================

Usage
-----
    python scripts/v2/build_corpus.py [options]

Options
-------
    --manifest FILE     lmd_manifest.json  (default: data/v2/processed/)
    --popularity FILE   popularity_scores.json  (default: same dir)
    --out FILE          Output parquet  (default: data/v2/processed/songs.parquet)
    --workers N         Parallel workers  (default: cpu_count - 1, min 1)
    --min-notes N       Drop songs with fewer melody notes  (default: 5)
    --dtw-threshold F   Normalised DTW distance for near-duplicate merging
                        (default: 0.5)
    --log-level LEVEL   DEBUG / INFO / WARNING / ERROR  (default: INFO)

Exit codes: 0 success · 1 internal error · 2 missing inputs
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statistics
from fastdtw import fastdtw
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path bootstrap — must run in both the main process AND spawned workers.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from melody_matcher.preprocessing.melody_extractor import (  # noqa: E402
    MelodyExtractionError,
    MelodyExtractor,
    MelodyNote,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_MANIFEST = _REPO_ROOT / "data" / "v2" / "processed" / "lmd_manifest.json"
_DEFAULT_POPULARITY = _REPO_ROOT / "data" / "v2" / "processed" / "popularity_scores.json"
_DEFAULT_OUT = _REPO_ROOT / "data" / "v2" / "processed" / "songs.parquet"

# Groups larger than this skip pairwise DTW to bound worst-case O(n²) time.
_DTW_GROUP_CAP = 50

# Fallback popularity score for songs not found in the scores file.
_DEFAULT_POPULARITY_SCORE = 0.5


# ---------------------------------------------------------------------------
# Phase 1.5 — Normalisation (pure, called inside workers)
# ---------------------------------------------------------------------------

def _compute_features(
    notes: list[MelodyNote],
) -> tuple[list[list[float]], str]:
    """Normalise a melody to key- and tempo-invariant features.

    Args:
        notes: Ordered monophonic melody from :class:`MelodyExtractor`.

    Returns:
        A 2-tuple of:

        * ``intervals`` — list of ``[semitone_delta, normalised_ioi]`` pairs
          (length ``N-1`` for an ``N``-note melody).
        * ``parsons_code`` — contour string of ``U`` / ``D`` / ``R`` characters
          (also length ``N-1``).
    """
    if len(notes) < 2:
        return [], ""

    # Inter-onset intervals in beats (gap between consecutive note onsets).
    iois: list[float] = [
        notes[i + 1].start_beat - notes[i].start_beat
        for i in range(len(notes) - 1)
    ]

    # Tempo-invariant normalisation: divide by the per-song median IOI.
    median_ioi: float = statistics.median(iois)
    if median_ioi <= 0.0:
        median_ioi = 1.0
    norm_iois = [ioi / median_ioi for ioi in iois]

    # Semitone deltas — key-invariant because only relative motion is stored.
    deltas: list[int] = [
        notes[i + 1].pitch - notes[i].pitch
        for i in range(len(notes) - 1)
    ]

    # Parsons code: U = up, D = down, R = repeat.
    parsons: str = "".join(
        "U" if d > 0 else "D" if d < 0 else "R" for d in deltas
    )

    intervals: list[list[float]] = [
        [float(d), float(ioi)] for d, ioi in zip(deltas, norm_iois)
    ]

    return intervals, parsons


# ---------------------------------------------------------------------------
# Worker (must be a top-level function to survive multiprocessing spawn)
# ---------------------------------------------------------------------------

def _process_song(task: tuple[str, list[str], float, int]) -> dict[str, Any] | None:
    """Extract and normalise the melody for one MSD ID.

    Tries each MIDI path in order; returns the first successful result with
    at least ``min_notes`` melody notes, or ``None`` if every path fails.

    Args:
        task: ``(msd_id, midi_paths, popularity_log, min_notes)`` where
            *midi_paths* are absolute path strings.

    Returns:
        A row dict ready for a DataFrame, or ``None`` on failure.
    """
    msd_id, midi_paths, popularity_log, min_notes = task
    extractor = MelodyExtractor()

    for midi_path_str in midi_paths:
        midi_path = Path(midi_path_str)
        try:
            notes: list[MelodyNote] = extractor.extract(midi_path)
        except MelodyExtractionError:
            continue

        if len(notes) < min_notes:
            continue

        intervals, parsons = _compute_features(notes)

        return {
            "song_id": msd_id,
            "popularity_log": popularity_log,
            "melody_notes_json": json.dumps(
                [[n.pitch, round(n.start_beat, 6), round(n.duration_beats, 6)]
                 for n in notes]
            ),
            "melody_intervals_json": json.dumps(intervals),
            "parsons_code": parsons,
            "source_midi_path": str(midi_path.resolve()),
        }

    return None


# ---------------------------------------------------------------------------
# Phase 1.6 — Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(
    rows: list[dict[str, Any]],
    dtw_threshold: float,
) -> tuple[list[dict[str, Any]], int, int]:
    """Remove duplicate songs in two stages.

    Stage 1 — **Exact**: identical serialised interval sequences are collapsed
    immediately; the entry with the higher popularity score is kept.

    Stage 2 — **Near-duplicate**: within each Parsons-code group (bounded to
    ``_DTW_GROUP_CAP`` songs), pairwise FastDTW identifies songs whose melodic
    intervals are below ``dtw_threshold`` distance per note. Clusters are
    merged, keeping the most popular representative.

    Args:
        rows: Extracted corpus rows.
        dtw_threshold: Normalised DTW distance threshold.

    Returns:
        ``(deduplicated_rows, exact_removed, dtw_removed)``
    """
    # ------------------------------------------------------------------
    # Stage 1: exact interval-sequence deduplication
    # ------------------------------------------------------------------
    exact_index: dict[str, int] = {}   # intervals_json → index in deduped
    deduped: list[dict[str, Any]] = []

    for row in rows:
        key: str = row["melody_intervals_json"]
        if key in exact_index:
            existing_idx = exact_index[key]
            if row["popularity_log"] > deduped[existing_idx]["popularity_log"]:
                deduped[existing_idx] = row
                # index key stays the same
        else:
            exact_index[key] = len(deduped)
            deduped.append(row)

    exact_removed = len(rows) - len(deduped)

    # ------------------------------------------------------------------
    # Stage 2: DTW near-duplicate within Parsons-code groups
    # ------------------------------------------------------------------
    parsons_groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(deduped):
        parsons_groups[row["parsons_code"]].append(i)

    keep: set[int] = set()

    for parsons_code, indices in parsons_groups.items():
        if len(indices) == 1:
            keep.add(indices[0])
            continue

        if len(indices) > _DTW_GROUP_CAP:
            # Group is too large for O(n²) pairwise DTW.  These are almost
            # certainly distinct songs that happen to share a coarse contour
            # (e.g. "UUUUUU" is common in many genuinely different melodies).
            keep.update(indices)
            continue

        # Build delta sequences for DTW (use semitone deltas only — IOI is
        # already captured by the Parsons grouping step).
        seqs: list[np.ndarray] = []
        for idx in indices:
            pairs = json.loads(deduped[idx]["melody_intervals_json"])
            seqs.append(np.array([p[0] for p in pairs], dtype=np.float32))

        # Union-Find for clustering near-duplicates.
        parent = list(range(len(indices)))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]   # path compression
                x = parent[x]
            return x

        def _union(x: int, y: int) -> None:
            parent[_find(x)] = _find(y)

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                si, sj = seqs[i], seqs[j]
                if len(si) == 0 or len(sj) == 0:
                    continue
                dist, _ = fastdtw(si, sj, dist=lambda a, b: abs(float(a) - float(b)))
                norm_dist = dist / min(len(si), len(sj))
                if norm_dist <= dtw_threshold:
                    _union(i, j)

        # Collect clusters; keep the most popular representative of each.
        clusters: dict[int, list[int]] = defaultdict(list)
        for local_i in range(len(indices)):
            clusters[_find(local_i)].append(local_i)

        for cluster_members in clusters.values():
            best_local = max(
                cluster_members,
                key=lambda li: deduped[indices[li]]["popularity_log"],
            )
            keep.add(indices[best_local])

    result = [deduped[i] for i in sorted(keep)]
    dtw_removed = len(deduped) - len(result)

    return result, exact_removed, dtw_removed


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build songs.parquet: extract, normalise, deduplicate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST, metavar="FILE")
    p.add_argument("--popularity", type=Path, default=_DEFAULT_POPULARITY, metavar="FILE")
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT, metavar="FILE")
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        metavar="N",
        help="Parallel worker processes (default: cpu_count - 1)",
    )
    p.add_argument(
        "--min-notes",
        type=int,
        default=5,
        metavar="N",
        help="Minimum melody notes to accept a song (default: 5)",
    )
    p.add_argument(
        "--dtw-threshold",
        type=float,
        default=0.5,
        metavar="F",
        help="Normalised DTW distance for near-duplicate merging (default: 0.5)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------

def _run(args: argparse.Namespace) -> None:
    log = logging.getLogger(__name__)

    # --- Guard: inputs must exist ---
    for label, path in [("manifest", args.manifest), ("popularity", args.popularity)]:
        if not path.is_file():
            print(f"{label} file not found: {path}", file=sys.stderr)
            sys.exit(2)

    # --- Load inputs ---
    with args.manifest.open(encoding="utf-8") as fh:
        manifest: dict[str, Any] = json.load(fh)

    midi_root = Path(manifest["midi_root"])
    entries: dict[str, list[str]] = manifest["entries"]
    log.info("Manifest: %d MSD IDs, midi_root=%s", len(entries), midi_root)

    with args.popularity.open(encoding="utf-8") as fh:
        pop_data: dict[str, Any] = json.load(fh)

    pop_scores: dict[str, float] = pop_data.get("scores", {})
    fallback_score = pop_data.get("stats", {}).get("median_score", _DEFAULT_POPULARITY_SCORE)
    log.info(
        "Popularity scores: %d entries loaded (fallback=%.4f).",
        len(pop_scores), fallback_score,
    )

    # --- Build task list ---
    tasks: list[tuple[str, list[str], float, int]] = []
    for msd_id, rel_paths in entries.items():
        abs_paths = [str((midi_root / rp).resolve()) for rp in rel_paths]
        pop = pop_scores.get(msd_id, fallback_score)
        tasks.append((msd_id, abs_paths, pop, args.min_notes))

    log.info(
        "Starting extraction: %d songs, %d workers, min_notes=%d.",
        len(tasks), args.workers, args.min_notes,
    )

    # --- Phase 1.4+1.5: parallel extraction + normalisation ---
    rows: list[dict[str, Any]] = []
    skipped = 0

    with multiprocessing.Pool(processes=args.workers) as pool:
        with tqdm(total=len(tasks), desc="Extracting", unit="song", dynamic_ncols=True) as pbar:
            for result in pool.imap_unordered(_process_song, tasks, chunksize=8):
                if result is not None:
                    rows.append(result)
                else:
                    skipped += 1
                pbar.update(1)
                pbar.set_postfix(kept=len(rows), skipped=skipped, refresh=False)

    log.info(
        "Extraction complete: %d kept, %d skipped (corrupt / too few notes).",
        len(rows), skipped,
    )

    if not rows:
        log.error("No songs extracted — check --midi-root and MIDI files.")
        sys.exit(2)

    # --- Phase 1.6: deduplication ---
    log.info(
        "Deduplicating %d songs (DTW threshold=%.2f, group cap=%d) …",
        len(rows), args.dtw_threshold, _DTW_GROUP_CAP,
    )
    rows, exact_removed, dtw_removed = _deduplicate(rows, args.dtw_threshold)
    log.info(
        "Deduplication: removed %d exact duplicates, %d near-duplicates. %d songs remain.",
        exact_removed, dtw_removed, len(rows),
    )

    # --- Phase 1.7: save parquet ---
    df = pd.DataFrame(
        rows,
        columns=[
            "song_id",
            "popularity_log",
            "melody_notes_json",
            "melody_intervals_json",
            "parsons_code",
            "source_midi_path",
        ],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False, engine="pyarrow")

    log.info(
        "Saved %d songs → %s  (%.1f MB)",
        len(df),
        args.out,
        args.out.stat().st_size / 1_048_576,
    )

    # --- Summary ---
    print(
        f"\n{'Corpus build summary':=<55}\n"
        f"  MSD IDs in manifest   : {len(entries):>8,}\n"
        f"  Extraction failures   : {skipped:>8,}\n"
        f"  Kept after extraction : {len(rows) + exact_removed + dtw_removed:>8,}\n"
        f"  Exact duplicates rm'd : {exact_removed:>8,}\n"
        f"  Near-dupes rm'd (DTW) : {dtw_removed:>8,}\n"
        f"  Final corpus size     : {len(df):>8,}\n"
        f"  Output                : {args.out}\n"
    )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    log = logging.getLogger(__name__)
    try:
        _run(args)
    except SystemExit:
        raise
    except Exception:
        log.error("Internal error:\n%s", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()   # required for PyInstaller on Windows
    main()
