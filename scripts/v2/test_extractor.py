"""Quick smoke-test CLI for MelodyExtractor.

Usage
-----
    python scripts/v2/test_extractor.py --midi path/to/file.mid
    python scripts/v2/test_extractor.py --midi path/to/file.mid --n 30 --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pretty_midi

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from melody_matcher.preprocessing.melody_extractor import (  # noqa: E402
    MelodyExtractionError,
    MelodyExtractor,
    _NON_MELODY_PROGRAMS,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect the melody extracted from a MIDI file.")
    p.add_argument("--midi", required=True, type=Path, metavar="FILE")
    p.add_argument("--n", type=int, default=20, metavar="N",
                   help="Number of melody notes to print (default: 20)")
    p.add_argument("--log-level", default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _print_instrument_table(pm: pretty_midi.PrettyMIDI) -> None:
    print(f"\n{'Tracks':=<60}")
    print(f"  {'#':>2}  {'Prog':>4}  {'Drum':>4}  {'Notes':>5}  {'Filtered':>8}  Name")
    print(f"  {'-'*2}  {'-'*4}  {'-'*4}  {'-'*5}  {'-'*8}  {'-'*20}")
    for i, instr in enumerate(pm.instruments):
        filtered = instr.is_drum or instr.program in _NON_MELODY_PROGRAMS
        flag = "yes" if filtered else ""
        print(
            f"  {i:>2}  {instr.program:>4}  "
            f"{'yes' if instr.is_drum else '':>4}  "
            f"{len(instr.notes):>5}  "
            f"{flag:>8}  "
            f"{instr.name!r}"
        )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    midi_path: Path = args.midi.resolve()
    if not midi_path.is_file():
        print(f"Error: file not found: {midi_path}", file=sys.stderr)
        sys.exit(1)

    # --- Load for inspection ---
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as exc:
        print(f"Error: pretty_midi could not load {midi_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    tempo_changes = pm.get_tempo_changes()
    tempos = tempo_changes[1]
    avg_tempo = float(tempos.mean()) if len(tempos) else 0.0

    print(f"\n{'File info':=<60}")
    print(f"  Path       : {midi_path}")
    print(f"  Resolution : {pm.resolution} ppq")
    print(f"  Tempo      : {avg_tempo:.1f} BPM avg  ({len(tempos)} change(s))")
    print(f"  End time   : {pm.get_end_time():.2f} s")
    print(f"  Instruments: {len(pm.instruments)}")

    _print_instrument_table(pm)

    # --- Extract melody ---
    extractor = MelodyExtractor()
    try:
        melody = extractor.extract(midi_path)
    except MelodyExtractionError as exc:
        print(f"\nExtraction failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'Melody':=<60}")
    print(f"  Total notes extracted : {len(melody)}")

    if not melody:
        print("  (no notes — file may contain only drums or filtered instruments)")
        return

    show = melody[: args.n]
    print(f"  Showing first {len(show)} of {len(melody)}:\n")
    print(f"  {'#':>4}  {'Pitch':>5}  {'Name':<4}  {'Start (beat)':>12}  {'Dur (beats)':>11}")
    print(f"  {'-'*4}  {'-'*5}  {'-'*4}  {'-'*12}  {'-'*11}")
    for idx, note in enumerate(show, start=1):
        name = pretty_midi.note_number_to_name(note.pitch)
        print(
            f"  {idx:>4}  {note.pitch:>5}  {name:<4}  "
            f"{note.start_beat:>12.3f}  {note.duration_beats:>11.3f}"
        )


if __name__ == "__main__":
    main()
