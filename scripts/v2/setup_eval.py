"""
Phase 0 — Evaluation harness setup.

Run this script once to download and extract the MIR-QBSH dataset into the
frozen eval directory.  The dataset must NEVER be used for training.

Usage
-----
    source venv/bin/activate
    python scripts/v2/setup_eval.py [--dry-run]

Expected directory layout after extraction
-------------------------------------------
    data/v2/eval/mir_qbsh/
        QBSHdata/
            <song_id>/          # folder name == ground-truth song ID
                <query_id>.wav  # one or more hummed queries per song
        PitchFile/              # optional: pre-extracted pitch vectors (.pv)
            <song_id>/
                <query_id>.pv

The MIR-QBSH loader (src/melody_matcher/eval/dataset_loader.py) derives the
ground-truth song ID from the parent folder name, so the layout above must be
preserved when extracting.

How to obtain the dataset
--------------------------
MIR-QBSH is distributed through the MIREX community and the original authors
at National Chengchi University (NCCU), Taiwan.

Option 1 — Direct download (preferred if still hosted):
    Homepage : http://mirlab.org/dataSet/public/
    Direct   : http://mirlab.org/dataSet/public/MIR-QBSH.rar
    Mirror   : https://github.com/mir-dataset-loaders/mir-dataset-loaders
               (search for "mir_qbsh" loader)

Option 2 — Via mirdata (pip install mirdata):
    python -c "
    import mirdata
    dataset = mirdata.initialize('mir_qbsh')
    dataset.download()
    "
    Then move the extracted folder into data/v2/eval/mir_qbsh/.

Option 3 — Manual extraction:
    1. Download the .rar or .zip archive from one of the sources above.
    2. Extract it so that data/v2/eval/mir_qbsh/QBSHdata/ exists.
    3. Run this script with --dry-run to verify the layout is correct.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MIR_QBSH_DIR = REPO_ROOT / "data" / "v2" / "eval" / "mir_qbsh"
HAND_RECORDED_DIR = REPO_ROOT / "data" / "v2" / "eval" / "hand_recorded"

# Top-level sub-directories the MIR-QBSH archive creates after extraction.
EXPECTED_SUBDIRS = ["QBSHdata"]


def check_layout(root: Path) -> dict[str, bool]:
    """Return a dict of expected sub-path → exists for the given root."""
    return {subdir: (root / subdir).is_dir() for subdir in EXPECTED_SUBDIRS}


def count_wav_files(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.wav"))


def verify(dry_run: bool) -> int:
    """
    Verify that the MIR-QBSH dataset has been extracted correctly.

    Returns 0 on success, 1 on failure.
    """
    log.info("Checking eval directory layout …")

    if not MIR_QBSH_DIR.exists():
        log.error("Directory not found: %s", MIR_QBSH_DIR)
        log.error("Run `mkdir -p %s` and extract the dataset there.", MIR_QBSH_DIR)
        return 1

    layout = check_layout(MIR_QBSH_DIR)
    missing = [name for name, present in layout.items() if not present]

    if missing:
        log.warning("MIR-QBSH directory exists but expected sub-directories are missing:")
        for name in missing:
            log.warning("  missing: %s/%s", MIR_QBSH_DIR, name)
        log.warning("The dataset may not have been extracted yet.  See module docstring for instructions.")
        if dry_run:
            log.info("[dry-run] Skipping further checks.")
            return 0
        return 1

    wav_count = count_wav_files(MIR_QBSH_DIR)
    log.info("Found %d .wav file(s) under %s", wav_count, MIR_QBSH_DIR)

    if wav_count == 0:
        log.warning("No .wav files found.  Did extraction complete successfully?")
        return 1

    # MIR-QBSH should contain roughly 4000 queries across ~48 songs.
    if wav_count < 100:
        log.warning(
            "Only %d .wav files found; MIR-QBSH should have ~4000. "
            "The archive may be incomplete.",
            wav_count,
        )

    log.info("Layout check passed.  MIR-QBSH appears correctly extracted.")
    return 0


def verify_hand_recorded() -> None:
    if not HAND_RECORDED_DIR.exists():
        log.warning("hand_recorded directory missing: %s", HAND_RECORDED_DIR)
        return
    wav_count = count_wav_files(HAND_RECORDED_DIR)
    if wav_count == 0:
        log.info(
            "hand_recorded directory is empty.  "
            "Record 100–200 phone-mic queries and place them here as: "
            "data/v2/eval/hand_recorded/<song_id>/<query_id>.wav"
        )
    else:
        log.info("hand_recorded: %d .wav file(s) found.", wav_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Phase 0 eval directory layout.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print checks without failing on missing files.",
    )
    args = parser.parse_args()

    exit_code = verify(dry_run=args.dry_run)
    verify_hand_recorded()

    if exit_code == 0:
        log.info("Setup check complete.  Run `python scripts/v2/eval.py` to evaluate.")
    else:
        log.error(
            "Setup incomplete.  See the docstring at the top of this file for "
            "download instructions, then re-run."
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
