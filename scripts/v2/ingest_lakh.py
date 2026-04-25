"""Phase 1.1 — Ingest the LMD-matched MIDI corpus and write a manifest.

Usage
-----
    python scripts/v2/ingest_lakh.py [options]

    Options
    -------
    --midi-root     Root of the extracted lmd_matched/ tree
                    (default: data/v2/raw_midi/lmd_matched/)
    --manifest-out  Output path for the JSON manifest
                    (default: data/v2/processed/lmd_manifest.json)
    --log-level     Logging verbosity (default: INFO)

Exit codes
----------
0   Success — manifest written.
1   Internal error — traceback logged.
2   Data missing — lmd_matched not found or contains no MIDI files.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path bootstrap — importable without `pip install -e .`
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from melody_matcher.data.lakh_parser import LakhMatchedParser  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_MIDI_ROOT = _REPO_ROOT / "data" / "v2" / "raw_midi" / "lmd_matched"
_DEFAULT_MANIFEST_OUT = _REPO_ROOT / "data" / "v2" / "processed" / "lmd_manifest.json"
_MIDI_EXTENSIONS: frozenset[str] = frozenset({".mid", ".midi"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def _has_midi_files(root: Path) -> bool:
    """Return True as soon as any .mid/.midi file is found under *root*."""
    return any(
        p.suffix.lower() in _MIDI_EXTENSIONS
        for p in root.rglob("*")
        if p.is_file()
    )


def _print_download_instructions(midi_root: Path) -> None:
    lines = [
        f"LMD-matched dataset not found at {midi_root}.",
        "Download it from: https://colinraffel.com/projects/lmd/",
        "File: lmd_matched.tar.gz (~1.6 GB)",
        "Extract with: tar -xzf lmd_matched.tar.gz -C data/v2/raw_midi/",
    ]
    for line in lines:
        print(line, file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest LMD-matched MIDI corpus and write a JSON manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--midi-root",
        type=Path,
        default=_DEFAULT_MIDI_ROOT,
        metavar="DIR",
        help=f"Root of lmd_matched/ tree (default: {_DEFAULT_MIDI_ROOT})",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=_DEFAULT_MANIFEST_OUT,
        metavar="FILE",
        help=f"Output JSON manifest path (default: {_DEFAULT_MANIFEST_OUT})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------


def _run(args: argparse.Namespace) -> None:
    log = logging.getLogger(__name__)

    midi_root: Path = args.midi_root.resolve()
    manifest_out: Path = args.manifest_out.resolve()

    # --- Guard: data must be present ---
    if not midi_root.is_dir() or not _has_midi_files(midi_root):
        _print_download_instructions(args.midi_root)
        sys.exit(2)

    parser = LakhMatchedParser(midi_root)

    # --- Pass 1: count valid files for a deterministic tqdm total ---
    log.info("Pass 1/2 — counting valid MIDI files under %s …", midi_root)
    valid_count = sum(1 for _ in parser.iter_files())
    log.info("Pass 1/2 — %d valid MIDI files found.", valid_count)

    # --- Pass 2: collect entries ---
    log.info("Pass 2/2 — building manifest …")
    grouped: dict[str, list[str]] = {}

    with tqdm(total=valid_count, desc="Scanning", unit="file", leave=True) as pbar:
        for entry in parser.iter_files():
            msd_id: str = entry["msd_id"]
            rel_path: str = entry["midi_path"].relative_to(midi_root).as_posix()
            grouped.setdefault(msd_id, []).append(rel_path)
            pbar.update(1)

    skipped = parser.skipped_count

    # --- Build manifest ---
    entries: dict[str, list[str]] = {
        msd_id: sorted(paths)
        for msd_id, paths in sorted(grouped.items())
    }
    total_msd = len(entries)
    total_midi = sum(len(v) for v in entries.values())

    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "midi_root": str(midi_root),
        "stats": {
            "total_msd_ids": total_msd,
            "total_midi_files": total_midi,
            "mean_midis_per_msd": (total_midi / total_msd) if total_msd else 0.0,
            "skipped_files": skipped,
        },
        "entries": entries,
    }

    # --- Atomic write ---
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_out.with_name(manifest_out.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp_path, manifest_out)

    log.info(
        "Manifest written → %s  (%d MSD IDs, %d MIDI files, %d skipped)",
        manifest_out,
        total_msd,
        total_midi,
        skipped,
    )


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)
    log = logging.getLogger(__name__)

    try:
        _run(args)
    except SystemExit:
        raise
    except Exception:
        log.error("Internal error:\n%s", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
