"""CLI smoke-test for AudioTranscriber.

Usage
-----
    python scripts/v2/test_transcriber.py --audio path/to/recording.wav
    python scripts/v2/test_transcriber.py --audio recording.wav --output my_debug.mid
    python scripts/v2/test_transcriber.py --audio recording.wav --log-level DEBUG

The script:
  1. Runs AudioTranscriber on the given audio file.
  2. Prints a table of the extracted MelodyNote objects.
  3. Writes a ``debug_output.mid`` MIDI file (or --output path) so you can
     listen to what the model heard.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pretty_midi

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from melody_matcher.io.audio_transcriber import AudioTranscriber, TranscriptionError  # noqa: E402
from melody_matcher.preprocessing.melody_extractor import MelodyNote               # noqa: E402
from melody_matcher.preprocessing.normalizer import compute_features               # noqa: E402

_NOMINAL_BPM = 120.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transcribe an audio file to MelodyNotes and write a debug MIDI.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--audio", required=True, type=Path, metavar="FILE",
                   help="Input audio file (WAV, MP3, OGG, FLAC, …)")
    p.add_argument("--output", type=Path, default=None, metavar="FILE",
                   help="Debug MIDI output path (default: debug_output.mid in cwd)")
    p.add_argument("--onset-threshold", type=float, default=0.5, metavar="F",
                   help="Basic Pitch onset threshold 0–1 (default 0.5)")
    p.add_argument("--frame-threshold", type=float, default=0.3, metavar="F",
                   help="Basic Pitch frame threshold 0–1 (default 0.3)")
    p.add_argument("--min-note-ms", type=float, default=60.0, metavar="MS",
                   help="Drop notes shorter than this many ms (default 60)")
    p.add_argument("--gap-merge-ms", type=float, default=50.0, metavar="MS",
                   help="Merge same-pitch notes with gap < this many ms (default 50)")
    p.add_argument("--log-level", default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _print_notes(notes: list[MelodyNote], audio_path: Path) -> None:
    """Render a clean note table to stdout."""
    intervals, parsons = compute_features(notes)

    print(f"\n{'Transcription result':=<65}")
    print(f"  Audio       : {audio_path}")
    print(f"  Notes       : {len(notes)}")
    print(f"  Parsons     : {parsons[:50]}{'…' if len(parsons) > 50 else ''}")
    print()

    col_n   = 4
    col_nm  = 4
    col_pit = 5
    col_stb = 12
    col_dur = 11
    col_ioi = 11
    col_dlt = 9

    header = (
        f"  {'#':>{col_n}}  {'Note':<{col_nm}}  {'MIDI':>{col_pit}}  "
        f"{'Start (beat)':>{col_stb}}  {'Dur (beats)':>{col_dur}}  "
        f"{'IOI (norm)':>{col_ioi}}  {'Δsemi':>{col_dlt}}"
    )
    sep = "  " + "-" * (col_n + col_nm + col_pit + col_stb + col_dur + col_ioi + col_dlt + 12)
    print(header)
    print(sep)

    for i, note in enumerate(notes, start=1):
        name = pretty_midi.note_number_to_name(note.pitch)
        ioi_str = f"{intervals[i-1][1]:>{col_ioi}.3f}" if i <= len(intervals) else f"{'—':>{col_ioi}}"
        dlt_str = f"{int(intervals[i-1][0]):+{col_dlt}d}" if i <= len(intervals) else f"{'—':>{col_dlt}}"
        print(
            f"  {i:>{col_n}}  {name:<{col_nm}}  {note.pitch:>{col_pit}}  "
            f"{note.start_beat:>{col_stb}.3f}  {note.duration_beats:>{col_dur}.3f}  "
            f"{ioi_str}  {dlt_str}"
        )
    print()


def _save_debug_midi(notes: list[MelodyNote], output_path: Path) -> None:
    """Write the transcribed notes to a MIDI file for playback verification.

    Converts beat positions back to seconds using the same nominal 120 BPM
    used during transcription, so the MIDI tempo and rhythmic proportions
    are preserved exactly.
    """
    seconds_per_beat = 60.0 / _NOMINAL_BPM   # 0.5 s / beat at 120 BPM

    pm = pretty_midi.PrettyMIDI(initial_tempo=_NOMINAL_BPM)
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Flute"),
        name="Transcribed Melody",
    )

    for note in notes:
        start_s = note.start_beat * seconds_per_beat
        end_s = (note.start_beat + note.duration_beats) * seconds_per_beat
        end_s = max(end_s, start_s + 0.01)   # guard against zero-duration
        instrument.notes.append(
            pretty_midi.Note(velocity=80, pitch=note.pitch, start=start_s, end=end_s)
        )

    pm.instruments.append(instrument)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(output_path))
    print(f"  Debug MIDI saved → {output_path}")
    print(f"  (playback at {_NOMINAL_BPM:.0f} BPM — tempo is nominal, only intervals matter)")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(levelname)-8s  %(name)s: %(message)s",
    )

    if not args.audio.is_file():
        print(f"Error: audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(2)

    output_path = args.output or Path("debug_output.mid")

    print(f"\nTranscribing {args.audio.name} …", flush=True)

    transcriber = AudioTranscriber(
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        min_note_ms=args.min_note_ms,
        gap_merge_ms=args.gap_merge_ms,
    )

    try:
        notes = transcriber.transcribe(args.audio)
    except TranscriptionError as exc:
        print(f"\nTranscription failed: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_notes(notes, args.audio)
    _save_debug_midi(notes, output_path)


if __name__ == "__main__":
    main()
