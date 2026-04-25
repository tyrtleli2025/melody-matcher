"""Phase 1.2/1.3 — Resolve MSD IDs to Last.fm popularity scores.

This script reads ``lmd_manifest.json``, looks up the artist + title for each
MSD ID from a local metadata source, queries the Last.fm API for play counts,
and writes ``popularity_scores.json`` with scores normalized to [0.0, 1.0].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Quick-start
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Get a free Last.fm API key
   → https://www.last.fm/api/account/create  (takes ~30 seconds)

2. Create ``.env`` in the project root::

       LASTFM_API_KEY=your_32_char_key_here

3. Download MSD track metadata (needed to map MSD ID → artist + title).

   Option A — ``unique_tracks.txt`` (~74 MB compressed, recommended):
       curl -O http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt
       mv unique_tracks.txt data/v2/raw_midi/unique_tracks.txt
       # Columns are delimited by the literal string <SEP>, not tabs or commas.

   Option B — ``track_metadata.db`` SQLite (~1.9 GB, part of the full MSD):
       # Download from http://millionsongdataset.com/pages/getting-dataset/
       mv track_metadata.db data/v2/raw_midi/track_metadata.db

4. Run::

       python scripts/v2/fetch_popularity.py

   The script checkpoints every 100 records to
   ``data/v2/processed/popularity_checkpoint.db`` so you can safely Ctrl-C
   and resume.  At ~3 req/s, 31 k songs takes roughly 3 hours.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Normalization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Raw play counts span many orders of magnitude.  The script applies::

    log_score = log(1 + playcount)

then min-max normalises to [0.0, 1.0] over the corpus.  Songs not found on
Last.fm receive the corpus-median score as a neutral prior, so they are not
artificially ranked first or last.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Exit codes: 0 success · 1 internal error · 2 missing inputs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from melody_matcher.data.lastfm_client import LastFmClient  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_MANIFEST = _REPO_ROOT / "data" / "v2" / "processed" / "lmd_manifest.json"
_DEFAULT_CHECKPOINT = _REPO_ROOT / "data" / "v2" / "processed" / "popularity_checkpoint.db"
_DEFAULT_OUT = _REPO_ROOT / "data" / "v2" / "processed" / "popularity_scores.json"
_DEFAULT_UNIQUE_TRACKS = _REPO_ROOT / "data" / "v2" / "raw_midi" / "unique_tracks.txt"
_DEFAULT_METADATA_DB = _REPO_ROOT / "data" / "v2" / "raw_midi" / "track_metadata.db"

_CHECKPOINT_INTERVAL = 100  # flush to SQLite every N records


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# MSD metadata loaders
# ---------------------------------------------------------------------------

def _load_metadata_from_unique_tracks(path: Path) -> dict[str, tuple[str, str]]:
    """Parse ``unique_tracks.txt`` → {msd_id: (artist, title)}.

    File format: ``<track_id><SEP><song_id><SEP><artist_name><SEP><title>``
    The literal string ``<SEP>`` is used as the column delimiter.
    """
    meta: dict[str, tuple[str, str]] = {}
    log = logging.getLogger(__name__)
    log.info("Loading MSD metadata from %s …", path)
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("<SEP>")
            if len(parts) < 4:
                continue
            track_id, _, artist, title = parts[0], parts[1], parts[2], parts[3]
            meta[track_id] = (artist.strip(), title.strip())
    log.info("Loaded %d MSD track records.", len(meta))
    return meta


def _load_metadata_from_sqlite(path: Path) -> dict[str, tuple[str, str]]:
    """Query ``track_metadata.db`` → {msd_id: (artist, title)}."""
    log = logging.getLogger(__name__)
    log.info("Loading MSD metadata from SQLite %s …", path)
    meta: dict[str, tuple[str, str]] = {}
    with sqlite3.connect(path) as con:
        for row in con.execute("SELECT track_id, artist_name, title FROM songs"):
            msd_id, artist, title = row
            if msd_id and artist and title:
                meta[str(msd_id)] = (str(artist).strip(), str(title).strip())
    log.info("Loaded %d MSD track records.", len(meta))
    return meta


def _load_metadata_from_csv(path: Path) -> dict[str, tuple[str, str]]:
    """Load a custom CSV with columns ``msd_id,artist,title`` (header row optional)."""
    import csv
    log = logging.getLogger(__name__)
    log.info("Loading MSD metadata from CSV %s …", path)
    meta: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 3:
                continue
            msd_id, artist, title = row[0].strip(), row[1].strip(), row[2].strip()
            if msd_id.startswith("TR") and artist:
                meta[msd_id] = (artist, title)
    log.info("Loaded %d MSD track records from CSV.", len(meta))
    return meta


def _auto_load_metadata(
    unique_tracks: Path,
    metadata_db: Path,
    metadata_csv: Path | None,
) -> dict[str, tuple[str, str]]:
    """Load metadata from whichever source is present, with priority order."""
    if metadata_csv and metadata_csv.is_file():
        return _load_metadata_from_csv(metadata_csv)
    if unique_tracks.is_file():
        return _load_metadata_from_unique_tracks(unique_tracks)
    if metadata_db.is_file():
        return _load_metadata_from_sqlite(metadata_db)
    return {}


# ---------------------------------------------------------------------------
# Checkpoint (SQLite)
# ---------------------------------------------------------------------------

def _open_checkpoint(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, isolation_level=None)  # autocommit
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            msd_id    TEXT PRIMARY KEY,
            artist    TEXT,
            title     TEXT,
            playcount INTEGER,
            listeners INTEGER,
            status    TEXT,  -- 'found' | 'not_found' | 'no_metadata'
            fetched_at REAL
        )
        """
    )
    return con


def _already_done(con: sqlite3.Connection) -> set[str]:
    rows = con.execute("SELECT msd_id FROM results").fetchall()
    return {r[0] for r in rows}


def _save_batch(con: sqlite3.Connection, batch: list[tuple]) -> None:
    con.executemany(
        """
        INSERT OR REPLACE INTO results
            (msd_id, artist, title, playcount, listeners, status, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        batch,
    )


def _load_checkpoint(con: sqlite3.Connection) -> dict[str, int]:
    """Return {msd_id: playcount} for all 'found' records."""
    rows = con.execute(
        "SELECT msd_id, playcount FROM results WHERE status = 'found'"
    ).fetchall()
    return {r[0]: int(r[1]) for r in rows}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_scores(
    playcounts: dict[str, int],
    all_msd_ids: list[str],
) -> dict[str, float]:
    """Log-normalize play counts to [0.0, 1.0]; assign median to missing IDs."""
    if not playcounts:
        return {msd_id: 0.5 for msd_id in all_msd_ids}

    log_scores = {msd_id: math.log1p(pc) for msd_id, pc in playcounts.items()}
    lo = min(log_scores.values())
    hi = max(log_scores.values())
    span = hi - lo

    if span == 0:
        normalized = {msd_id: 0.5 for msd_id in log_scores}
    else:
        normalized = {
            msd_id: (ls - lo) / span for msd_id, ls in log_scores.items()
        }

    median = statistics.median(normalized.values())

    return {
        msd_id: normalized.get(msd_id, median)
        for msd_id in all_msd_ids
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch Last.fm popularity scores for LMD-matched MSD IDs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST, metavar="FILE")
    p.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT, metavar="FILE",
                   help="SQLite checkpoint file (default: %(default)s)")
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT, metavar="FILE",
                   help="Output JSON path (default: %(default)s)")
    p.add_argument("--unique-tracks", type=Path, default=_DEFAULT_UNIQUE_TRACKS, metavar="FILE",
                   help="MSD unique_tracks.txt (default: %(default)s)")
    p.add_argument("--metadata-db", type=Path, default=_DEFAULT_METADATA_DB, metavar="FILE",
                   help="MSD track_metadata.db SQLite (default: %(default)s)")
    p.add_argument("--metadata-csv", type=Path, default=None, metavar="FILE",
                   help="Custom CSV with columns msd_id,artist,title")
    p.add_argument("--calls-per-second", type=float, default=3.0, metavar="N",
                   help="Last.fm request rate (default: 3.0)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run(args: argparse.Namespace) -> None:
    log = logging.getLogger(__name__)

    # --- Load manifest ---
    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        print("Run scripts/v2/ingest_lakh.py first.", file=sys.stderr)
        sys.exit(2)

    with args.manifest.open(encoding="utf-8") as fh:
        manifest: dict = json.load(fh)

    all_msd_ids: list[str] = sorted(manifest["entries"].keys())
    log.info("Manifest loaded: %d MSD IDs.", len(all_msd_ids))

    # --- Load MSD metadata (artist + title) ---
    metadata = _auto_load_metadata(
        args.unique_tracks, args.metadata_db, args.metadata_csv
    )
    if not metadata:
        print(
            "\nNo MSD metadata found. Download unique_tracks.txt:\n"
            "  curl -O http://millionsongdataset.com/sites/default/files/"
            "AdditionalFiles/unique_tracks.txt\n"
            "  mv unique_tracks.txt data/v2/raw_midi/unique_tracks.txt\n"
            "Then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(2)

    # --- Open checkpoint ---
    checkpoint_con = _open_checkpoint(args.checkpoint)
    done = _already_done(checkpoint_con)
    remaining = [msd_id for msd_id in all_msd_ids if msd_id not in done]
    log.info(
        "Checkpoint: %d done, %d remaining.", len(done), len(remaining)
    )

    # --- Fetch from Last.fm ---
    batch: list[tuple] = []

    with LastFmClient(calls_per_second=args.calls_per_second) as client:
        with tqdm(
            total=len(remaining),
            desc="Fetching Last.fm",
            unit="song",
            dynamic_ncols=True,
        ) as pbar:
            for msd_id in remaining:
                artist_title = metadata.get(msd_id)

                if artist_title is None:
                    batch.append(
                        (msd_id, None, None, 0, 0, "no_metadata", _now())
                    )
                    pbar.update(1)
                else:
                    artist, title = artist_title
                    result = client.get_track_info(artist, title)

                    if result is not None:
                        batch.append((
                            msd_id, artist, title,
                            result["playcount"], result["listeners"],
                            "found", _now(),
                        ))
                    else:
                        batch.append(
                            (msd_id, artist, title, 0, 0, "not_found", _now())
                        )
                    pbar.update(1)

                if len(batch) >= _CHECKPOINT_INTERVAL:
                    _save_batch(checkpoint_con, batch)
                    batch.clear()
                    log.debug("Checkpoint flushed.")

            # flush tail
            if batch:
                _save_batch(checkpoint_con, batch)

    # --- Build final scores ---
    playcounts = _load_checkpoint(checkpoint_con)
    checkpoint_con.close()

    found = len(playcounts)
    not_found = len(all_msd_ids) - found
    log.info("Last.fm results: %d found, %d not found / no metadata.", found, not_found)

    scores = _normalize_scores(playcounts, all_msd_ids)

    # --- Write output ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stats": {
            "total_msd_ids": len(all_msd_ids),
            "found_on_lastfm": found,
            "not_found": not_found,
            "median_score": statistics.median(scores.values()),
        },
        "scores": scores,
    }
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, args.out)

    log.info(
        "Popularity scores written → %s  (%d IDs, %d found on Last.fm, %d not found)",
        args.out, len(all_msd_ids), found, not_found,
    )


def _now() -> float:
    import time
    return time.time()


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
