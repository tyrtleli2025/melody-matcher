"""MIDI reading utilities for melody extraction."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from music21 import chord, converter, note, stream


def extract_notes_from_midi(file_path: str | Path) -> list[note.Note]:
    """Parse a MIDI file and extract a simple melody note sequence.

    This first-pass implementation handles polyphonic MIDI by using a top-line
    heuristic: for each onset time, it keeps only the highest pitch. This tends
    to approximate the primary melody in many common arrangements.

    Rests are ignored.

    Args:
        file_path: Path to a ``.mid`` or ``.midi`` file.

    Returns:
        A list of ``music21.note.Note`` objects in chronological order.
        Returns an empty list when no notes are found.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a file, or music21 cannot parse it.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"MIDI file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file path, got: {path}")

    try:
        parsed: stream.Score = converter.parse(str(path))
    except Exception as exc:  # pragma: no cover - exact parse errors vary by file
        raise ValueError(f"Failed to parse MIDI file: {path}") from exc

    # Collect candidate notes keyed by absolute onset offset.
    # For chords, keep only the top note from that chord (highest pitch).
    notes_by_offset: dict[float, list[note.Note]] = defaultdict(list)

    for element in parsed.recurse().notes:
        try:
            offset = float(element.getOffsetInHierarchy(parsed))
        except Exception:
            offset = float(element.offset)

        if isinstance(element, note.Note):
            notes_by_offset[offset].append(element)
            continue

        if isinstance(element, chord.Chord) and element.notes:
            highest = max(element.notes, key=lambda n: n.pitch.midi)
            notes_by_offset[offset].append(highest)

    melody: list[note.Note] = []
    for offset in sorted(notes_by_offset):
        highest_at_onset = max(notes_by_offset[offset], key=lambda n: n.pitch.midi)
        melody.append(highest_at_onset)

    return melody


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    demo_path = repo_root / "data" / "raw" / "castle-complete.mid"
    try:
        extracted = extract_notes_from_midi(demo_path)
        notes = [n.nameWithOctave for n in extracted]
        print(f"Loaded {len(extracted)} notes from {demo_path}")
        print("First 10 notes:", notes)
    except Exception as err:
        print(f"Demo failed for {demo_path}: {err}")
