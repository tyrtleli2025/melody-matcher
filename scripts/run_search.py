"""Interactive CLI for melody search (exact + fuzzy).

Usage:
    python scripts/run_search.py [--index PATH] [--top N] [--min-score FLOAT]

Type a space-separated note sequence at the prompt, e.g.:
    > C4 D4 E4 F4 G4
    > C D E            (octave 4 assumed when omitted)
    > quit / exit      to stop
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from melody_matcher.features.interval_encoder import encode_intervals
from melody_matcher.search.search_service import search_exact_match, search_fuzzy_match
from melody_matcher.storage.index_store import load_index

_DEFAULT_INDEX = _REPO_ROOT / "data" / "index" / "melody_index.json"
_DEFAULT_OCTAVE = "4"
_NOTE_WITHOUT_OCTAVE = re.compile(r"^([A-Ga-g][#b]?)$")

# ANSI colour helpers (gracefully degraded on Windows)
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"


def _colourise(score: float) -> str:
    """Return an ANSI-coloured confidence string."""
    pct = f"{score * 100:.1f}%"
    if score >= 0.9:
        return f"{_GREEN}{_BOLD}{pct}{_RESET}"
    if score >= 0.7:
        return f"{_YELLOW}{pct}{_RESET}"
    return f"{_RED}{pct}{_RESET}"


def _normalise_notes(raw: str) -> list[str]:
    """Split input and append default octave to bare note names."""
    normalised: list[str] = []
    for token in raw.split():
        if _NOTE_WITHOUT_OCTAVE.match(token):
            normalised.append(token + _DEFAULT_OCTAVE)
        else:
            normalised.append(token)
    return normalised


def _print_results(
    results: list[dict],
    query_notes: list[str],
    exact_count: int,
) -> None:
    # Show the encoded signature so the user can see what was actually searched.
    try:
        sig = encode_intervals(query_notes)
        print(f"  Interval signature : {sig}")
    except Exception:
        pass

    if not results:
        print("  No matches found.\n")
        return

    print(f"  {len(results)} result(s)  [{exact_count} exact]\n")
    print(f"  {'#':>3}  {'Score':>7}  {'Segment':>7}  File")
    print(f"  {'-'*3}  {'-'*7}  {'-'*7}  {'-'*40}")

    for rank, hit in enumerate(results, start=1):
        score = hit.get("score", 1.0)
        marker = " *" if score == 1.0 else "  "
        print(
            f"  {rank:>3}{marker} {_colourise(score):>7}  "
            f"#{hit['segment_index']:<6}  {hit['file']}"
        )
    print()
    if exact_count:
        print(f"  (* = exact match)\n")


def run(index_path: Path, top_n: int, min_score: float) -> None:
    print("Loading index...", end=" ", flush=True)
    try:
        index = load_index(index_path)
    except Exception as exc:
        print(f"\n[ERROR] Could not load index: {exc}")
        sys.exit(1)

    sig_count = len(index)
    entry_count = sum(len(v) for v in index.values())
    print(f"done.  ({sig_count:,} signatures · {entry_count:,} entries)\n")

    print(f'Enter notes separated by spaces, e.g. "C4 D4 E4 F4 G4".')
    print(f'Octave defaults to 4 when omitted.  "quit" / "exit" to stop.\n')

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw:
            continue
        if raw.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        notes = _normalise_notes(raw)
        print(f"  Query : {notes}")

        if len(notes) < 2:
            print("  Please enter at least 2 notes.\n")
            continue

        try:
            exact = search_exact_match(notes, index)
            fuzzy = search_fuzzy_match(notes, index, top_n=top_n, min_score=min_score)
        except ValueError as exc:
            print(f"  [ERROR] {exc}\n")
            continue

        # Merge: exact hits are guaranteed score=1.0; fuzzy may already include
        # them. De-duplicate by (file, segment_index) so exact hits always show
        # with score 1.0 and aren't doubled.
        exact_keys = {(h["file"], h["segment_index"]) for h in exact}

        # Promote exact hits to score 1.0 inside the fuzzy list.
        for hit in fuzzy:
            if (hit["file"], hit["segment_index"]) in exact_keys:
                hit["score"] = 1.0

        # Add any exact hits the fuzzy scan missed (score below min_score
        # threshold but still a true hit).
        seen = {(h["file"], h["segment_index"]) for h in fuzzy}
        for hit in exact:
            key = (hit["file"], hit["segment_index"])
            if key not in seen:
                fuzzy.insert(0, {**hit, "score": 1.0})

        fuzzy.sort(key=lambda r: r["score"], reverse=True)
        merged = fuzzy[:top_n]

        _print_results(merged, notes, exact_count=len(exact_keys & seen | exact_keys))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive melody search CLI.")
    parser.add_argument(
        "--index",
        type=Path,
        default=_DEFAULT_INDEX,
        help=f"Path to the JSON index (default: {_DEFAULT_INDEX.name})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Maximum results to display (default: 20)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Minimum similarity score 0–1 (default: 0.5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.index, top_n=args.top, min_score=args.min_score)