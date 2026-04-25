"""Tests for melody_matcher.data.lakh_parser.LakhMatchedParser."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from melody_matcher.data.lakh_parser import LakhMatchedParser

# Two syntactically valid MSD IDs used across tests.
_ID_A = "TRAAAAV128F421A322"   # TR + 16 uppercase alphanumerics
_ID_B = "TRAABXG128F429B4A9"   # same


def _midi(path: Path, content: bytes = b"MThd\x00\x00\x00\x06") -> Path:
    """Create a minimal file at *path*, making parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_total_count(self, tmp_path: Path) -> None:
        root = tmp_path / "lmd_matched"
        _midi(root / "A" / "A" / "A" / _ID_A / "abc123.mid")
        _midi(root / "A" / "A" / "B" / _ID_B / "def456.mid")
        _midi(root / "A" / "A" / "B" / _ID_B / "ghi789.mid")

        entries = list(LakhMatchedParser(root).iter_files())

        assert len(entries) == 3

    def test_msd_ids_extracted(self, tmp_path: Path) -> None:
        root = tmp_path / "lmd_matched"
        _midi(root / "A" / "A" / "A" / _ID_A / "abc123.mid")
        _midi(root / "A" / "A" / "B" / _ID_B / "def456.mid")

        ids = {e["msd_id"] for e in LakhMatchedParser(root).iter_files()}

        assert ids == {_ID_A, _ID_B}

    def test_group_by_msd_id_lengths(self, tmp_path: Path) -> None:
        root = tmp_path / "lmd_matched"
        _midi(root / "A" / "A" / "A" / _ID_A / "abc123.mid")
        _midi(root / "A" / "A" / "B" / _ID_B / "def456.mid")
        _midi(root / "A" / "A" / "B" / _ID_B / "ghi789.mid")

        grouped = LakhMatchedParser(root).group_by_msd_id()

        assert set(grouped.keys()) == {_ID_A, _ID_B}
        assert len(grouped[_ID_A]) == 1
        assert len(grouped[_ID_B]) == 2

    def test_entry_shape(self, tmp_path: Path) -> None:
        root = tmp_path / "lmd_matched"
        _midi(root / "A" / "A" / "A" / _ID_A / "abc123.mid")

        entry = next(LakhMatchedParser(root).iter_files())

        assert set(entry.keys()) == {"msd_id", "midi_path", "md5"}
        assert entry["msd_id"] == _ID_A
        assert entry["md5"] == "abc123"
        assert isinstance(entry["midi_path"], Path)
        assert entry["midi_path"].is_absolute()

    def test_group_by_msd_id_returns_paths(self, tmp_path: Path) -> None:
        root = tmp_path / "lmd_matched"
        p = _midi(root / "A" / "A" / "A" / _ID_A / "abc123.mid")

        grouped = LakhMatchedParser(root).group_by_msd_id()

        assert grouped[_ID_A][0] == p.resolve()


# ---------------------------------------------------------------------------
# Case-insensitive extensions
# ---------------------------------------------------------------------------

class TestExtensions:
    def test_uppercase_mid(self, tmp_path: Path) -> None:
        root = tmp_path / "lmd_matched"
        _midi(root / "A" / "A" / "A" / _ID_A / "track.MID")

        entries = list(LakhMatchedParser(root).iter_files())

        assert len(entries) == 1

    def test_lowercase_midi(self, tmp_path: Path) -> None:
        root = tmp_path / "lmd_matched"
        _midi(root / "A" / "A" / "A" / _ID_A / "track.midi")

        entries = list(LakhMatchedParser(root).iter_files())

        assert len(entries) == 1

    def test_mixed_extensions_all_found(self, tmp_path: Path) -> None:
        root = tmp_path / "lmd_matched"
        _midi(root / "A" / "A" / "A" / _ID_A / "a.mid")
        _midi(root / "A" / "A" / "A" / _ID_A / "b.MID")
        _midi(root / "A" / "A" / "A" / _ID_A / "c.midi")

        entries = list(LakhMatchedParser(root).iter_files())

        assert len(entries) == 3

    def test_non_midi_extension_ignored(self, tmp_path: Path) -> None:
        root = tmp_path / "lmd_matched"
        _midi(root / "A" / "A" / "A" / _ID_A / "song.mp3")
        _midi(root / "A" / "A" / "A" / _ID_A / "song.txt")

        entries = list(LakhMatchedParser(root).iter_files())

        assert len(entries) == 0


# ---------------------------------------------------------------------------
# Invalid parent folder name
# ---------------------------------------------------------------------------

class TestInvalidParent:
    def test_non_msd_parent_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        root = tmp_path / "lmd_matched"
        _midi(root / "FOOBAR" / "track.mid")

        with caplog.at_level(logging.WARNING, logger="melody_matcher.data.lakh_parser"):
            entries = list(LakhMatchedParser(root).iter_files())

        assert len(entries) == 0
        assert any("FOOBAR" in r.message for r in caplog.records)

    def test_warning_is_at_warning_level(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        root = tmp_path / "lmd_matched"
        _midi(root / "invalid_parent" / "track.mid")

        with caplog.at_level(logging.WARNING, logger="melody_matcher.data.lakh_parser"):
            list(LakhMatchedParser(root).iter_files())

        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_skipped_count_incremented(self, tmp_path: Path) -> None:
        root = tmp_path / "lmd_matched"
        _midi(root / "BADPARENT" / "track.mid")

        parser = LakhMatchedParser(root)
        list(parser.iter_files())

        assert parser.skipped_count == 1

    def test_valid_alongside_invalid(self, tmp_path: Path) -> None:
        """Valid and invalid siblings — only valid entries are yielded."""
        root = tmp_path / "lmd_matched"
        _midi(root / "A" / "A" / "A" / _ID_A / "good.mid")
        _midi(root / "BADFOLDER" / "bad.mid")

        entries = list(LakhMatchedParser(root).iter_files())

        assert len(entries) == 1
        assert entries[0]["msd_id"] == _ID_A


# ---------------------------------------------------------------------------
# Hidden files and macOS archive artifacts
# ---------------------------------------------------------------------------

class TestSkippedArtifacts:
    def test_hidden_mid_skipped(self, tmp_path: Path) -> None:
        root = tmp_path / "lmd_matched"
        _midi(root / "A" / "A" / "A" / _ID_A / ".hidden.mid")

        entries = list(LakhMatchedParser(root).iter_files())

        assert len(entries) == 0

    def test_macosx_dir_skipped(self, tmp_path: Path) -> None:
        root = tmp_path / "lmd_matched"
        _midi(root / "__MACOSX" / _ID_A / "song.mid")

        entries = list(LakhMatchedParser(root).iter_files())

        assert len(entries) == 0

    def test_macosx_nested_skipped(self, tmp_path: Path) -> None:
        """__MACOSX anywhere in the path should cause a skip."""
        root = tmp_path / "lmd_matched"
        _midi(root / "A" / "__MACOSX" / _ID_A / "deep.mid")

        entries = list(LakhMatchedParser(root).iter_files())

        assert len(entries) == 0

    def test_ds_store_skipped(self, tmp_path: Path) -> None:
        root = tmp_path / "lmd_matched"
        (root / "A" / "A" / "A" / _ID_A).mkdir(parents=True)
        (root / "A" / "A" / "A" / _ID_A / ".DS_Store").write_bytes(b"")

        entries = list(LakhMatchedParser(root).iter_files())

        assert len(entries) == 0

    def test_hidden_plus_valid_coexist(self, tmp_path: Path) -> None:
        root = tmp_path / "lmd_matched"
        _midi(root / "A" / "A" / "A" / _ID_A / ".hidden.mid")
        _midi(root / "A" / "A" / "A" / _ID_A / "visible.mid")

        entries = list(LakhMatchedParser(root).iter_files())

        assert len(entries) == 1
        assert entries[0]["md5"] == "visible"


# ---------------------------------------------------------------------------
# Non-existent root
# ---------------------------------------------------------------------------

class TestInvalidRoot:
    def test_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError, match="does_not_exist"):
            LakhMatchedParser(tmp_path / "does_not_exist")

    def test_file_as_root_raises(self, tmp_path: Path) -> None:
        a_file = tmp_path / "not_a_dir.mid"
        a_file.write_bytes(b"")

        with pytest.raises(NotADirectoryError):
            LakhMatchedParser(a_file)

    def test_error_message_includes_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "absent_dir"
        with pytest.raises(NotADirectoryError, match=str(missing)):
            LakhMatchedParser(missing)


# ---------------------------------------------------------------------------
# Symlink escape
# ---------------------------------------------------------------------------

class TestSymlinkEscape:
    def test_symlink_outside_root_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        root = tmp_path / "lmd_matched"
        msd_dir = root / "A" / "A" / "A" / _ID_A
        msd_dir.mkdir(parents=True)

        outside = tmp_path / "outside.mid"
        outside.write_bytes(b"MThd")

        symlink = msd_dir / "escape.mid"
        symlink.symlink_to(outside)

        with caplog.at_level(logging.WARNING, logger="melody_matcher.data.lakh_parser"):
            entries = list(LakhMatchedParser(root).iter_files())

        assert len(entries) == 0
        assert any("escapes root" in r.message for r in caplog.records)

    def test_symlink_skipped_count(self, tmp_path: Path) -> None:
        root = tmp_path / "lmd_matched"
        msd_dir = root / "A" / "A" / "A" / _ID_A
        msd_dir.mkdir(parents=True)

        outside = tmp_path / "outside.mid"
        outside.write_bytes(b"MThd")
        (msd_dir / "escape.mid").symlink_to(outside)

        parser = LakhMatchedParser(root)
        list(parser.iter_files())

        assert parser.skipped_count == 1

    def test_symlink_inside_root_allowed(self, tmp_path: Path) -> None:
        """A symlink whose target is still inside root is valid."""
        root = tmp_path / "lmd_matched"
        msd_dir = root / "A" / "A" / "A" / _ID_A
        msd_dir.mkdir(parents=True)

        real_file = msd_dir / "real.mid"
        real_file.write_bytes(b"MThd")

        link = msd_dir / "link.mid"
        link.symlink_to(real_file)

        entries = list(LakhMatchedParser(root).iter_files())
        # Both the real file and the symlink (pointing inside root) are valid.
        assert len(entries) == 2
