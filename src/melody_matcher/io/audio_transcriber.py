"""Audio-to-symbolic transcription using Spotify Basic Pitch (Phase 4).

Pipeline
--------
1. **Load & pre-process** — :func:`librosa.load` handles any audio format
   (WAV, MP3, OGG, FLAC, …) and resamples to 22 050 Hz mono.  Loudness is
   then normalised to −23 LUFS (EBU R 128) so that Basic Pitch receives
   a consistent amplitude envelope regardless of recording level.

2. **Inference** — The pre-processed audio is written to a temporary WAV file
   and passed to :func:`basic_pitch.inference.predict`, which runs the
   ONNX model (ICASSP 2022) and returns raw polyphonic note events as a list
   of ``(start_s, end_s, pitch_midi, amplitude, pitch_bends)`` tuples.

3. **Post-processing** — Raw events are cleaned into a monophonic
   :class:`~melody_matcher.preprocessing.melody_extractor.MelodyNote` sequence:

   a. *Vocal range filter* — discard notes outside C3–C6 (MIDI 48–84) to
      suppress octave-error hallucinations that Basic Pitch produces on noisy
      recordings.
   b. *Minimum duration filter* — drop notes shorter than ``min_note_ms``
      (default 60 ms); these are typically breath-attack transients.
   c. *Monophonic enforcement* — when notes overlap, keep the one with higher
      amplitude (confidence); trim the loser to the winner's onset so that
      no two notes sound simultaneously.
   d. *Gap merge* — re-join consecutive same-pitch notes separated by a gap
      smaller than ``gap_merge_ms`` (default 50 ms), which models the natural
      break that occurs when a singer takes a breath on a sustained note.
   e. *Second duration filter* — re-apply the minimum duration check after
      merging (trimming can produce new sub-threshold fragments).
   f. *Seconds → beats* — convert event times to beat positions using a
      nominal 120 BPM.  Because :func:`~melody_matcher.preprocessing.normalizer.compute_features`
      divides all IOIs by their per-melody median, any constant tempo
      multiplier cancels out, making the beat unit arbitrary but consistent.

Why a temp file instead of passing audio directly?
    :func:`~basic_pitch.inference.predict` accepts only a file path (it
    calls :func:`librosa.load` internally).  Writing a 22 050 Hz mono WAV
    to ``/tmp`` takes a few milliseconds and avoids forking the Basic Pitch
    internals.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import NamedTuple

import librosa
import numpy as np
import pyloudnorm
import soundfile as sf
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict as _bp_predict

from melody_matcher.preprocessing.melody_extractor import MelodyNote

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAMPLE_RATE: int = 22_050          # Basic Pitch's required sample rate
_TARGET_LUFS: float = -23.0         # EBU R 128 broadcast loudness target
_MAX_GAIN_DB: float = 30.0          # safety cap — never amplify more than 30 dB

# Vocal range: C3 (MIDI 48, ≈ 131 Hz) – C6 (MIDI 84, ≈ 1047 Hz)
_VOCAL_MIDI_LOW: int = 48           # C3
_VOCAL_MIDI_HIGH: int = 84          # C6
_VOCAL_HZ_LOW: float = 130.0        # Basic Pitch frequency gate (Hz)
_VOCAL_HZ_HIGH: float = 1050.0

_NOMINAL_BPM: float = 120.0         # arbitrary reference tempo for s → beats


# ---------------------------------------------------------------------------
# Internal note event type
# ---------------------------------------------------------------------------

class _Event(NamedTuple):
    """Raw note event in seconds."""
    start: float
    end: float
    pitch: int
    amplitude: float


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class TranscriptionError(Exception):
    """Raised when transcription fails or yields too few usable notes."""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AudioTranscriber:
    """Transcribe a hummed/sung audio file to a monophonic
    :class:`~melody_matcher.preprocessing.melody_extractor.MelodyNote`
    sequence suitable for :class:`~melody_matcher.retrieval.search_engine.SearchEngine`.

    Args:
        onset_threshold: Basic Pitch onset activation threshold in [0, 1].
            Higher values → fewer but higher-confidence note onsets.
            Default 0.5.
        frame_threshold: Basic Pitch frame activation threshold in [0, 1].
            Lower values → longer sustained notes captured.  Default 0.3.
        min_note_ms: Minimum note duration in milliseconds.  Notes shorter
            than this are discarded both before and after gap-merging.
            Default 60 ms.
        gap_merge_ms: Maximum silence gap (ms) between two consecutive
            same-pitch notes that should be merged into one.  Models the
            natural break a singer takes on a sustained note.  Default 50 ms.
        min_notes: Minimum number of valid notes required.  A
            :exc:`TranscriptionError` is raised if the result has fewer.
            Default 5.
    """

    def __init__(
        self,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        min_note_ms: float = 60.0,
        gap_merge_ms: float = 50.0,
        min_notes: int = 5,
    ) -> None:
        self._onset_threshold = onset_threshold
        self._frame_threshold = frame_threshold
        self._min_note_s = min_note_ms / 1000.0
        self._gap_merge_s = gap_merge_ms / 1000.0
        self._min_notes = min_notes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: Path | str) -> list[MelodyNote]:
        """Transcribe an audio file to a monophonic :class:`MelodyNote` list.

        Args:
            audio_path: Path to the audio file.  Supported formats: WAV, MP3,
                OGG, FLAC, AIFF, and anything :func:`librosa.load` can read.

        Returns:
            An ordered list of :class:`MelodyNote` objects.

        Raises:
            FileNotFoundError: If *audio_path* does not exist.
            TranscriptionError: If the file is silent, contains no pitched
                content in the vocal range, or yields fewer than ``min_notes``
                valid notes after post-processing.
        """
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        log.info("Loading audio: %s", audio_path)
        audio, sr = self._load_and_preprocess(audio_path)

        if audio.max() == 0.0:
            raise TranscriptionError(
                f"{audio_path.name} appears to be silent (all-zero waveform)."
            )

        raw_events = self._run_basic_pitch(audio)

        if not raw_events:
            raise TranscriptionError(
                "Basic Pitch detected no pitched content in the audio. "
                "Check that the file contains a sung or hummed melody."
            )

        notes = self._postprocess(raw_events)

        if len(notes) < self._min_notes:
            raise TranscriptionError(
                f"Only {len(notes)} valid note(s) remain after post-processing "
                f"(minimum required: {self._min_notes}). "
                "The recording may be too short or too noisy."
            )

        log.info("Transcription complete: %d notes.", len(notes))
        return notes

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _load_and_preprocess(self, audio_path: Path) -> tuple[np.ndarray, int]:
        """Load audio, convert to mono 22 050 Hz, and normalise to −23 LUFS.

        Returns:
            ``(audio, sample_rate)`` where *audio* is a 1-D float32 array.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, _ = librosa.load(str(audio_path), sr=_SAMPLE_RATE, mono=True)

        log.debug("Loaded %.2f s of audio (%d samples).", len(audio) / _SAMPLE_RATE, len(audio))

        audio = _normalize_lufs(audio, _SAMPLE_RATE)
        return audio, _SAMPLE_RATE

    # ------------------------------------------------------------------
    # Basic Pitch inference
    # ------------------------------------------------------------------

    def _run_basic_pitch(self, audio: np.ndarray) -> list[_Event]:
        """Write audio to a temp file, run Basic Pitch, and return raw events.

        Basic Pitch only accepts a file path (it runs :func:`librosa.load`
        internally), so we write the pre-processed array to a temp WAV,
        call :func:`~basic_pitch.inference.predict`, then delete the temp file.

        Args:
            audio: 1-D float32 mono array at 22 050 Hz.

        Returns:
            List of :class:`_Event` named-tuples in chronological order.
        """
        tmp_path: str | None = None
        try:
            # Write pre-processed audio to a named temp file.
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(tmp_path, audio, _SAMPLE_RATE, subtype="PCM_16")

            log.debug("Running Basic Pitch on temp file %s …", tmp_path)
            with _silence_stdout():
                _model_output, _midi_data, note_events = _bp_predict(
                    audio_path=tmp_path,
                    model_or_model_path=ICASSP_2022_MODEL_PATH,
                    onset_threshold=self._onset_threshold,
                    frame_threshold=self._frame_threshold,
                    minimum_note_length=self._min_note_s * 1000,  # ms
                    minimum_frequency=_VOCAL_HZ_LOW,
                    maximum_frequency=_VOCAL_HZ_HIGH,
                    multiple_pitch_bends=False,
                    melodia_trick=True,
                )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # note_events: list of (start_s, end_s, pitch_midi, amplitude, pitch_bends)
        return [
            _Event(start=float(s), end=float(e), pitch=int(p), amplitude=float(a))
            for s, e, p, a, _bends in note_events
        ]

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _postprocess(self, events: list[_Event]) -> list[MelodyNote]:
        """Clean raw polyphonic events into a monophonic :class:`MelodyNote` list.

        Stages (in order):
        1. Vocal range filter (C3–C6, MIDI 48–84).
        2. Minimum duration filter.
        3. Monophonic enforcement (highest amplitude wins at overlaps).
        4. Same-pitch gap merge (< ``gap_merge_ms``).
        5. Second minimum duration filter (catches fragments created by trimming).
        6. Sort by onset and convert seconds → beats.

        Args:
            events: Raw note events from Basic Pitch, in any order.

        Returns:
            Chronologically ordered :class:`MelodyNote` list.
        """
        # 1. Vocal range
        events = [e for e in events if _VOCAL_MIDI_LOW <= e.pitch <= _VOCAL_MIDI_HIGH]
        log.debug("After vocal range filter: %d events.", len(events))

        # 2. First duration filter
        events = [e for e in events if (e.end - e.start) >= self._min_note_s]
        log.debug("After first duration filter: %d events.", len(events))

        if not events:
            return []

        # 3. Monophonic enforcement
        events = _enforce_monophonic(events)
        log.debug("After monophonic enforcement: %d events.", len(events))

        # 4. Gap merge
        events = _merge_same_pitch(events, self._gap_merge_s)
        log.debug("After gap merge: %d events.", len(events))

        # 5. Second duration filter (trimming can expose sub-threshold fragments)
        events = [e for e in events if (e.end - e.start) >= self._min_note_s]
        log.debug("After second duration filter: %d events.", len(events))

        # 6. Sort and convert to MelodyNote
        events.sort(key=lambda e: e.start)
        beats_per_second = _NOMINAL_BPM / 60.0
        return [
            MelodyNote(
                pitch=e.pitch,
                start_beat=round(e.start * beats_per_second, 6),
                duration_beats=round((e.end - e.start) * beats_per_second, 6),
            )
            for e in events
        ]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _normalize_lufs(audio: np.ndarray, sr: int) -> np.ndarray:
    """Normalise *audio* to :data:`_TARGET_LUFS` using ITU-R BS.1770-4.

    Args:
        audio: 1-D float32 mono array.
        sr: Sample rate in Hz.

    Returns:
        Gain-adjusted audio array.  The input is returned unchanged if it is
        silent or too short for loudness measurement.
    """
    if len(audio) < sr:
        log.debug("Audio shorter than 1 s — skipping LUFS normalisation.")
        return audio

    meter = pyloudnorm.Meter(sr)
    # pyloudnorm expects shape (n_samples, n_channels); reshape for mono.
    lufs = meter.integrated_loudness(audio[:, np.newaxis])

    if not np.isfinite(lufs):
        log.debug("Integrated loudness is %s (silent?) — skipping normalisation.", lufs)
        return audio

    gain_db = min(_TARGET_LUFS - lufs, _MAX_GAIN_DB)
    log.debug("LUFS: %.1f dB → applying %.1f dB gain.", lufs, gain_db)
    return audio * float(10 ** (gain_db / 20.0))


def _enforce_monophonic(events: list[_Event]) -> list[_Event]:
    """Keep only the highest-amplitude note at any overlapping instant.

    When a higher-amplitude note starts while a lower-amplitude note is still
    sounding, the lower note is trimmed to the new note's onset.  Notes that
    result in a zero-duration window are removed.

    Args:
        events: Note events, in any order.

    Returns:
        Non-overlapping note events, sorted by start time.
    """
    # Primary: start time.  Tie-break: amplitude descending so the highest-
    # confidence note is processed first when two notes begin simultaneously.
    ordered = sorted(events, key=lambda e: (e.start, -e.amplitude))

    result: list[_Event] = []
    cur_end: float = 0.0
    cur_amp: float = 0.0

    for ev in ordered:
        if ev.start >= cur_end:
            # Clean gap — accept as-is.
            result.append(ev)
            cur_end = ev.end
            cur_amp = ev.amplitude
        elif ev.amplitude > cur_amp:
            # Higher-confidence note starts mid-previous note: trim the loser.
            if result and ev.start > result[-1].start:
                prev = result[-1]
                result[-1] = _Event(prev.start, ev.start, prev.pitch, prev.amplitude)
            elif result:
                result.pop()   # starts at exactly the same time — discard previous
            result.append(ev)
            cur_end = ev.end
            cur_amp = ev.amplitude
        # else: lower-confidence overlapping note — discard.

    return result


def _merge_same_pitch(events: list[_Event], gap_s: float) -> list[_Event]:
    """Re-join consecutive same-pitch notes separated by a gap < *gap_s*.

    Singers often interrupt a sustained note briefly while breathing; this
    step stitches such pairs back into a single note.

    Args:
        events: Monophonic note events sorted by start time.
        gap_s: Maximum gap (seconds) to bridge.

    Returns:
        Merged event list (still sorted by start time).
    """
    if not events:
        return []

    merged: list[_Event] = [events[0]]
    for ev in events[1:]:
        prev = merged[-1]
        gap = ev.start - prev.end
        if ev.pitch == prev.pitch and 0.0 <= gap < gap_s:
            # Extend the previous note to the end of the current one.
            merged[-1] = _Event(
                start=prev.start,
                end=ev.end,
                pitch=prev.pitch,
                amplitude=max(prev.amplitude, ev.amplitude),
            )
        else:
            merged.append(ev)

    return merged


@contextlib.contextmanager
def _silence_stdout() -> None:
    """Suppress stdout for the duration of the context (Basic Pitch prints to it)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield
