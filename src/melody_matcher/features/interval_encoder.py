"""Utilities for converting note sequences into relative semitone intervals."""

from __future__ import annotations

from collections.abc import Sequence

from music21 import interval, note, pitch


def _to_music21_note(value: str | note.Note) -> note.Note:
    """Convert a supported note representation into a ``music21.note.Note``.

    Args:
        value: A pitch string (for example ``"C4"``) or a ``music21.note.Note``.

    Returns:
        A ``music21.note.Note`` instance representing the same pitch.

    Raises:
        TypeError: If ``value`` is not a supported type.
        ValueError: If ``value`` cannot be parsed into a valid musical note.
    """
    if isinstance(value, note.Note):
        return value

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Empty note string is not valid.")
        try:
            parsed_pitch = pitch.Pitch(cleaned)
        except Exception as exc:  # pragma: no cover - exact exception varies in music21
            raise ValueError(f"Invalid note string: {value!r}") from exc
        return note.Note(parsed_pitch)

    raise TypeError(
        "Unsupported note value type. Expected str or music21.note.Note, "
        f"got {type(value).__name__}."
    )


def encode_intervals(notes: Sequence[str | note.Note]) -> list[int]:
    """Encode a melody as semitone jumps between consecutive notes.

    This function converts each consecutive note pair into a chromatic interval
    measured in semitones (ascending intervals are positive, descending are
    negative). The output is transposition-invariant and suitable for melody
    matching.

    Args:
        notes: Ordered sequence of notes, where each item is either:
            - a pitch string such as ``"C4"``, ``"F#3"``, ``"Bb5"``, or
            - a ``music21.note.Note`` instance.

    Returns:
        A list of integer semitone differences between consecutive notes.
        Example: ``["C4", "E4", "G4"] -> [4, 3]``.

        If fewer than 2 notes are provided, returns an empty list.

    Raises:
        TypeError: If ``notes`` is not a sequence, or contains unsupported item types.
        ValueError: If a note string is invalid.
    """
    if not isinstance(notes, Sequence):
        raise TypeError(
            f"'notes' must be a sequence of note values, got {type(notes).__name__}."
        )

    if len(notes) < 2:
        return []

    normalized_notes = [_to_music21_note(n) for n in notes]
    semitone_steps: list[int] = []

    for current_note, next_note in zip(normalized_notes, normalized_notes[1:]):
        chromatic_interval = interval.Interval(current_note, next_note)
        semitone_steps.append(int(chromatic_interval.chromatic.semitones))

    return semitone_steps


if __name__ == "__main__":
    # Simple "Twinkle Twinkle Little Star" opening phrase:
    # C C G G A A G
    demo_notes = ["C4", "C4", "G4", "G4", "A4", "A4", "G4"]
    print("Input notes:", demo_notes)
    print(encode_intervals(demo_notes))