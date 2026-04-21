"""Sliding-window segmentation of note sequences for melody indexing."""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def create_segments(notes: list[T], segment_length: int = 5) -> list[list[T]]:
    """Split a note sequence into overlapping fixed-length windows.

    Each consecutive window is offset by one note, so every possible
    sub-melody of the given length is represented exactly once. This
    makes the output suitable for bulk interval encoding and index
    population.

    Args:
        notes: Ordered sequence of notes (any element type — typically
            ``str`` pitch names or ``music21.note.Note`` objects).
        segment_length: Number of notes per segment. Must be >= 1.

    Returns:
        A list of segments, where each segment is a sub-list of length
        ``segment_length``. Returns an empty list when ``notes`` contains
        fewer elements than ``segment_length``.

        Example::

            create_segments(["A", "B", "C", "D", "E"], segment_length=3)
            # -> [["A", "B", "C"], ["B", "C", "D"], ["C", "D", "E"]]

    Raises:
        TypeError: If ``notes`` is not a list or ``segment_length`` is not an int.
        ValueError: If ``segment_length`` is less than 1.
    """
    if not isinstance(notes, list):
        raise TypeError(
            f"'notes' must be a list, got {type(notes).__name__}."
        )
    if not isinstance(segment_length, int):
        raise TypeError(
            f"'segment_length' must be an int, got {type(segment_length).__name__}."
        )
    if segment_length < 1:
        raise ValueError(
            f"'segment_length' must be >= 1, got {segment_length}."
        )

    if len(notes) < segment_length:
        return []

    return [notes[i : i + segment_length] for i in range(len(notes) - segment_length + 1)]


if __name__ == "__main__":
    demo_notes = ["C4", "D4", "E4", "F4", "G4", "A4", "B4"]
    segment_length = 3

    print("Input notes:", demo_notes)
    print(f"Segment length: {segment_length}")
    print()

    segments = create_segments(demo_notes, segment_length=segment_length)
    print(f"Produced {len(segments)} segment(s):")
    for i, seg in enumerate(segments):
        print(f"  [{i}] {seg}")

    print()
    print("Edge case — list shorter than segment length:")
    short = create_segments(["C4", "D4"], segment_length=5)
    print(f"  create_segments(['C4', 'D4'], segment_length=5) -> {short}")
