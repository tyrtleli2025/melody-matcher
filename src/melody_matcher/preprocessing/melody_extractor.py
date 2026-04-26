"""Melody extraction from multi-track MIDI files.

Pipeline
--------
1. Load the MIDI file with ``pretty_midi``.
2. Drop drum instruments (``is_drum == True``).
3. Drop non-melody instruments by General MIDI program number (bass,
   synth pads, FX, percussive, sound effects).
4. If every instrument was filtered out, fall back to all non-drum tracks
   so the caller always gets *something* to work with.
5. Apply the **Skyline Algorithm**: merge all remaining tracks and, at
   every instant, keep only the highest-sounding pitch.
6. Convert onset/offset times (seconds) to beat positions.
7. Return a list of :class:`MelodyNote` named-tuples.

General MIDI program exclusions
--------------------------------
+----------+-------------------------+----------------------------------+
| Range    | Category                | Reason excluded                  |
+==========+=========================+==================================+
| 32–39    | Bass                    | Sub-melody register              |
+----------+-------------------------+----------------------------------+
| 88–95    | Synth Pad               | Slow-attack chordal texture      |
+----------+-------------------------+----------------------------------+
| 96–103   | Synth FX                | Non-pitched soundscapes          |
+----------+-------------------------+----------------------------------+
| 112–127  | Percussive + SFX        | Rhythmic, non-pitched            |
+----------+-------------------------+----------------------------------+
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pretty_midi

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GM program-number filter
# ---------------------------------------------------------------------------

# Programs (0-indexed) whose tracks are unlikely to carry the melody line.
_BASS_PROGRAMS: frozenset[int] = frozenset(range(32, 40))
_PAD_PROGRAMS: frozenset[int] = frozenset(range(88, 96))
_FX_PROGRAMS: frozenset[int] = frozenset(range(96, 104))
_PERCUSSIVE_AND_SFX: frozenset[int] = frozenset(range(112, 128))

_NON_MELODY_PROGRAMS: frozenset[int] = (
    _BASS_PROGRAMS | _PAD_PROGRAMS | _FX_PROGRAMS | _PERCUSSIVE_AND_SFX
)

# Minimum number of notes an instrument must have to be considered.
_MIN_INSTRUMENT_NOTES: int = 4


# ---------------------------------------------------------------------------
# Public data type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MelodyNote:
    """One note in the extracted monophonic melody.

    Attributes:
        pitch: MIDI pitch number (0–127; 60 = middle C).
        start_beat: Onset time in beats (0.0 = start of the file).
        duration_beats: Note duration in beats.
    """

    pitch: int
    start_beat: float
    duration_beats: float

    def __str__(self) -> str:
        name = pretty_midi.note_number_to_name(self.pitch)
        return (
            f"{name:<3s}  pitch={self.pitch:3d}  "
            f"start={self.start_beat:.3f}b  dur={self.duration_beats:.3f}b"
        )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MelodyExtractionError(Exception):
    """Raised when a MIDI file cannot be parsed or yields no melody."""


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class MelodyExtractor:
    """Extract a monophonic melody sequence from a multi-track MIDI file.

    Args:
        min_duration_beats: Notes shorter than this (in beats) are discarded
            as likely artifacts.  Default 0.05 ≈ a 64th-note at 120 BPM.
    """

    def __init__(self, min_duration_beats: float = 0.05) -> None:
        self._min_dur = min_duration_beats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, midi_path: Path | str) -> list[MelodyNote]:
        """Extract a monophonic melody from *midi_path*.

        Args:
            midi_path: Path to a ``.mid`` / ``.midi`` file.

        Returns:
            Ordered list of :class:`MelodyNote` objects (may be empty if the
            file contains no usable pitched content).

        Raises:
            MelodyExtractionError: If the file cannot be loaded by
                ``pretty_midi`` (corrupted header, truncated file, etc.).
        """
        path = Path(midi_path)
        try:
            pm = pretty_midi.PrettyMIDI(str(path))
        except Exception as exc:
            raise MelodyExtractionError(
                f"pretty_midi could not load {path}: {exc}"
            ) from exc

        beat_times: np.ndarray = pm.get_beats()
        if len(beat_times) < 2:
            log.warning("%s: fewer than 2 beats detected — returning empty melody.", path.name)
            return []

        instruments = self._filter_instruments(pm.instruments)
        if not instruments:
            log.warning(
                "%s: all instruments filtered out; falling back to all non-drum tracks.",
                path.name,
            )
            instruments = [i for i in pm.instruments if not i.is_drum]

        raw_notes: list[pretty_midi.Note] = [
            note
            for instr in instruments
            for note in instr.notes
        ]

        if not raw_notes:
            log.warning("%s: no notes remain after filtering.", path.name)
            return []

        skyline_notes = self._skyline(raw_notes)
        melody = self._to_beat_sequence(skyline_notes, beat_times)
        log.debug(
            "%s: %d raw notes → %d skyline → %d melody notes.",
            path.name, len(raw_notes), len(skyline_notes), len(melody),
        )
        return melody

    # ------------------------------------------------------------------
    # Instrument filter
    # ------------------------------------------------------------------

    def _filter_instruments(
        self, instruments: Sequence[pretty_midi.Instrument]
    ) -> list[pretty_midi.Instrument]:
        """Return only instruments likely to carry a melody line.

        Drops:
        - Drum instruments (``is_drum == True``).
        - Instruments whose GM program number falls in :data:`_NON_MELODY_PROGRAMS`.
        - Instruments with fewer than :data:`_MIN_INSTRUMENT_NOTES` notes.
        """
        kept: list[pretty_midi.Instrument] = []
        for instr in instruments:
            if instr.is_drum:
                continue
            if instr.program in _NON_MELODY_PROGRAMS:
                log.debug(
                    "Skipping instrument %r (prog=%d) — non-melody program.",
                    instr.name, instr.program,
                )
                continue
            if len(instr.notes) < _MIN_INSTRUMENT_NOTES:
                log.debug(
                    "Skipping instrument %r (prog=%d) — only %d notes.",
                    instr.name, instr.program, len(instr.notes),
                )
                continue
            kept.append(instr)
        return kept

    # ------------------------------------------------------------------
    # Skyline algorithm
    # ------------------------------------------------------------------

    @staticmethod
    def _skyline(notes: list[pretty_midi.Note]) -> list[pretty_midi.Note]:
        """Apply the Skyline Algorithm to produce a monophonic note list.

        At any instant, the note with the highest pitch wins.  When a
        higher-pitched note starts while a lower-pitched note is still
        sounding, the lower note is trimmed to the new note's onset time.
        Notes that start after all current notes have ended are appended
        as-is.

        Args:
            notes: All notes from all melody-eligible instruments, in any order.

        Returns:
            A list of non-overlapping :class:`pretty_midi.Note` objects in
            chronological order.
        """
        # Primary sort: onset time; secondary: pitch descending (higher pitch
        # wins when two notes start at exactly the same instant).
        sorted_notes = sorted(notes, key=lambda n: (n.start, -n.pitch))

        melody: list[pretty_midi.Note] = []
        cur_pitch: int = -1
        cur_start: float = 0.0
        cur_end: float = 0.0
        cur_velocity: int = 64

        for note in sorted_notes:
            if note.start >= cur_end:
                # Clean boundary or gap — emit the running note and start a new one.
                if cur_pitch >= 0:
                    melody.append(
                        pretty_midi.Note(
                            velocity=cur_velocity,
                            pitch=cur_pitch,
                            start=cur_start,
                            end=cur_end,
                        )
                    )
                cur_pitch = note.pitch
                cur_start = note.start
                cur_end = note.end
                cur_velocity = note.velocity

            elif note.pitch > cur_pitch:
                # Higher-pitched note overlaps: trim running note and take over.
                if cur_pitch >= 0 and note.start > cur_start:
                    melody.append(
                        pretty_midi.Note(
                            velocity=cur_velocity,
                            pitch=cur_pitch,
                            start=cur_start,
                            end=note.start,  # trimmed
                        )
                    )
                cur_pitch = note.pitch
                cur_start = note.start
                cur_end = max(cur_end, note.end)
                cur_velocity = note.velocity

            elif note.end > cur_end and note.pitch == cur_pitch:
                # Same pitch, extends the current note — lengthen it.
                cur_end = note.end

            # else: lower-pitched note during a higher active note — discard.

        # Flush the last running note.
        if cur_pitch >= 0:
            melody.append(
                pretty_midi.Note(
                    velocity=cur_velocity,
                    pitch=cur_pitch,
                    start=cur_start,
                    end=cur_end,
                )
            )

        return melody

    # ------------------------------------------------------------------
    # Beat conversion
    # ------------------------------------------------------------------

    def _to_beat_sequence(
        self,
        notes: list[pretty_midi.Note],
        beat_times: np.ndarray,
    ) -> list[MelodyNote]:
        """Convert a list of timed notes to beat-position :class:`MelodyNote` objects.

        Uses linear interpolation between adjacent beat timestamps.  Notes
        outside the range of ``beat_times`` are extrapolated using the last
        known inter-beat interval so that pickup bars and trailing notes are
        handled gracefully.

        Notes shorter than ``min_duration_beats`` are dropped.

        Args:
            notes: Monophonic note list from :meth:`_skyline`.
            beat_times: Array of beat onset times in seconds, as returned by
                :func:`pretty_midi.PrettyMIDI.get_beats`.

        Returns:
            Ordered list of :class:`MelodyNote` with beat-relative positions.
        """
        result: list[MelodyNote] = []
        for note in notes:
            start_b = _seconds_to_beat(note.start, beat_times)
            end_b = _seconds_to_beat(note.end, beat_times)
            dur_b = end_b - start_b
            if dur_b < self._min_dur:
                continue
            result.append(MelodyNote(pitch=note.pitch, start_beat=start_b, duration_beats=dur_b))
        return result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _seconds_to_beat(t: float, beat_times: np.ndarray) -> float:
    """Convert a time in seconds to a fractional beat position.

    Interpolates between adjacent beats; extrapolates using the last
    inter-beat interval for times beyond the final beat.
    """
    n = len(beat_times)
    # Find the beat interval that contains t.
    idx = int(np.searchsorted(beat_times, t, side="right")) - 1

    if idx < 0:
        # Before the first beat — extrapolate backwards.
        dt = beat_times[1] - beat_times[0]
        return (t - beat_times[0]) / dt if dt > 0 else 0.0

    if idx >= n - 1:
        # After the last beat — extrapolate forwards.
        dt = beat_times[-1] - beat_times[-2]
        return (n - 1) + ((t - beat_times[-1]) / dt if dt > 0 else 0.0)

    dt = beat_times[idx + 1] - beat_times[idx]
    frac = (t - beat_times[idx]) / dt if dt > 0 else 0.0
    return idx + frac
