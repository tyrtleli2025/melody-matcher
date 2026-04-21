"""Build the melody inverted index from a folder of MIDI files.

Usage:
    python scripts/build_index.py [--raw-dir PATH] [--out PATH] [--segment-length N]

Defaults:
    --raw-dir       data/raw/
    --out           data/index/melody_index.json
    --segment-length  5
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# Allow running as a top-level script without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from melody_matcher.features.interval_encoder import encode_intervals
from melody_matcher.io.midi_reader import extract_notes_from_midi
from melody_matcher.preprocessing.segmenter import create_segments
from melody_matcher.storage.index_store import load_index, save_index


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the melody inverted index.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "raw",
        help="Directory containing .mid / .midi files (default: data/raw/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "data" / "index" / "melody_index.json",
        help="Output JSON index path (default: data/index/melody_index.json)",
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=5,
        help="Number of notes per segment (default: 5)",
    )
    return parser.parse_args()


def _progress(current: int, total: int, label: str, width: int = 30) -> None:
    """Print an in-place progress bar to stdout."""
    filled = int(width * current / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {current}/{total}  {label:<40}", end="", flush=True)


def build_index(
    raw_dir: Path,
    out_path: Path,
    segment_length: int,
) -> None:
    midi_files = sorted(
        p for p in raw_dir.rglob("*") if p.suffix.lower() in {".mid", ".midi"}
    )

    if not midi_files:
        print(f"No MIDI files found in {raw_dir}")
        return

    total = len(midi_files)
    print(f"Found {total} MIDI file(s) in {raw_dir}")
    print(f"Segment length: {segment_length} notes\n")

    # Load any previously built index so incremental runs merge cleanly.
    index = load_index(out_path)

    ok_count = 0
    err_count = 0

    for i, midi_path in enumerate(midi_files, start=1):
        rel = midi_path.relative_to(raw_dir)
        _progress(i, total, str(rel))

        try:
            notes = extract_notes_from_midi(midi_path)
        except Exception:
            print(f"\n  [WARN] Skipping {rel} — could not extract notes:")
            traceback.print_exc(limit=1)
            err_count += 1
            continue

        segments = create_segments(notes, segment_length=segment_length)

        if not segments:
            print(f"\n  [WARN] Skipping {rel} — fewer than {segment_length} notes.")
            err_count += 1
            continue

        for seg_idx, segment in enumerate(segments):
            try:
                intervals = encode_intervals(segment)
            except Exception:
                # A single bad segment should not abort the whole file.
                continue

            key = str(intervals)
            entry = {"file": str(rel), "segment_index": seg_idx}

            if key not in index:
                index[key] = []
            index[key].append(entry)

        ok_count += 1

    print()  # newline after progress bar
    print(f"\nProcessed {ok_count}/{total} file(s) successfully ({err_count} skipped).")

    dest = save_index(index, out_path)
    unique_signatures = len(index)
    total_entries = sum(len(v) for v in index.values())
    print(f"Index saved to: {dest}")
    print(f"  {unique_signatures:,} unique interval signatures")
    print(f"  {total_entries:,} total index entries")


if __name__ == "__main__":
    args = _parse_args()
    build_index(
        raw_dir=args.raw_dir,
        out_path=args.out,
        segment_length=args.segment_length,
    )
