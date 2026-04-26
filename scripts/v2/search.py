"""Query the melody search engine from a MIDI file.

Usage
-----
    python scripts/v2/search.py --midi path/to/query.mid
    python scripts/v2/search.py --midi path/to/query.mid --top-k 10
    python scripts/v2/search.py --midi path/to/query.mid --corpus data/v2/processed/songs.parquet

Exit codes: 0 success · 1 extraction/search error · 2 bad arguments
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from melody_matcher.preprocessing.melody_extractor import (  # noqa: E402
    MelodyExtractionError,
    MelodyExtractor,
)
from melody_matcher.retrieval.search_engine import (  # noqa: E402
    SearchEngine,
    SearchResult,
    _DEFAULT_CORPUS,
    _MIN_QUERY_NOTES,
)

_MIN_NOTES_HARD = 5   # refuse to search with fewer notes than this


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Search the melody corpus using a query MIDI file.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--midi", required=True, type=Path, metavar="FILE",
                   help="Query MIDI file (.mid / .midi)")
    p.add_argument("--top-k", type=int, default=5, metavar="N",
                   help="Number of results to display (default: 5)")
    p.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS, metavar="FILE",
                   help=f"songs.parquet path (default: {_DEFAULT_CORPUS})")
    p.add_argument("--max-candidates", type=int, default=500, metavar="N",
                   help="Parsons pre-filter candidate cap (default: 500)")
    p.add_argument("--log-level", default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _print_results(results: list[SearchResult], query_path: Path) -> None:
    """Render a clean table of search results to stdout."""
    print(f"\n{'Search results':=<72}")
    print(f"  Query : {query_path}")
    print(f"  Hits  : {len(results)}\n")

    if not results:
        print("  (no matches found)")
        return

    # Column widths
    col_rank   = 4
    col_id     = 20
    col_comb   = 13
    col_dtw    = 12
    col_pop    = 12

    header = (
        f"  {'#':>{col_rank}}  "
        f"{'Song ID':<{col_id}}  "
        f"{'Combined':>{col_comb}}  "
        f"{'DTW sim':>{col_dtw}}  "
        f"{'Popularity':>{col_pop}}"
    )
    sep = "  " + "-" * (col_rank + col_id + col_comb + col_dtw + col_pop + 8)

    print(header)
    print(sep)

    for r in results:
        print(
            f"  {r.rank:>{col_rank}}  "
            f"{r.song_id:<{col_id}}  "
            f"{r.combined_score:>{col_comb}.4f}  "
            f"{r.dtw_similarity:>{col_dtw}.4f}  "
            f"{r.popularity_log:>{col_pop}.4f}"
        )

    print()

    # Show MIDI path for the top hit as a convenience
    top = results[0]
    print(f"  Top match MIDI: {top.source_midi_path}")
    print(f"  Parsons prefix: {top.parsons_code[:40]}{'…' if len(top.parsons_code) > 40 else ''}")
    print()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(levelname)-8s  %(name)s: %(message)s",
    )

    # --- Validate inputs ---
    if not args.midi.is_file():
        print(f"Error: MIDI file not found: {args.midi}", file=sys.stderr)
        sys.exit(2)

    if not args.corpus.is_file():
        print(
            f"Error: corpus not found: {args.corpus}\n"
            "Run scripts/v2/build_corpus.py first.",
            file=sys.stderr,
        )
        sys.exit(2)

    # --- Extract query melody ---
    print(f"\nExtracting melody from {args.midi.name} …", flush=True)
    try:
        notes = MelodyExtractor().extract(args.midi)
    except MelodyExtractionError as exc:
        print(f"Error: extraction failed — {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  Extracted {len(notes)} melody notes.")

    if len(notes) < _MIN_NOTES_HARD:
        print(
            f"Error: query too short ({len(notes)} notes; minimum {_MIN_NOTES_HARD}).\n"
            "Try a longer MIDI file or lower --min-notes in build_corpus.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Load engine and search ---
    print(f"Loading corpus ({args.corpus.name}) …", flush=True)
    try:
        engine = SearchEngine(args.corpus, max_candidates=args.max_candidates)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"Searching …", flush=True)
    results = engine.search(notes, top_k=args.top_k)

    # --- Display ---
    _print_results(results, args.midi)


if __name__ == "__main__":
    main()
