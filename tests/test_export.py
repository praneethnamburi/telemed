"""Tests for ``immersionlab.telemed.export`` input-shape normalisation.

The COM-based extraction path itself is not unit-testable without a
running Echo Wave II instance, but the source-normalisation that
underlies ``telemed.export(source)`` (file / folder / list / mix) is
pure Python and worth pinning so the dispatch doesn't silently
regress.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _make_synthetic_tvd_files(root: Path, names: list[str]) -> list[Path]:
    """Create empty placeholder .tvd files so glob walking sees them."""
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for n in names:
        p = root / n
        p.write_bytes(b"")  # empty -- never opened by these tests
        paths.append(p)
    return paths


def test_normalize_single_file(tmp_path):
    from immersionlab.telemed._extract import _normalize_sources

    files = _make_synthetic_tvd_files(tmp_path, ["a.tvd"])
    out = _normalize_sources(files[0], recursive=True, pattern="*.tvd")
    assert out == files


def test_normalize_single_folder_recursive(tmp_path):
    from immersionlab.telemed._extract import _normalize_sources

    _make_synthetic_tvd_files(tmp_path, ["a.tvd"])
    _make_synthetic_tvd_files(tmp_path / "sub", ["b.tvd"])
    out = _normalize_sources(tmp_path, recursive=True, pattern="*.tvd")
    names = sorted(p.name for p in out)
    assert names == ["a.tvd", "b.tvd"]


def test_normalize_single_folder_non_recursive(tmp_path):
    from immersionlab.telemed._extract import _normalize_sources

    _make_synthetic_tvd_files(tmp_path, ["a.tvd"])
    _make_synthetic_tvd_files(tmp_path / "sub", ["b.tvd"])
    out = _normalize_sources(tmp_path, recursive=False, pattern="*.tvd")
    names = [p.name for p in out]
    assert names == ["a.tvd"]


def test_normalize_list_of_files(tmp_path):
    from immersionlab.telemed._extract import _normalize_sources

    files = _make_synthetic_tvd_files(tmp_path, ["a.tvd", "b.tvd", "c.tvd"])
    out = _normalize_sources(files, recursive=True, pattern="*.tvd")
    assert sorted(p.name for p in out) == ["a.tvd", "b.tvd", "c.tvd"]


def test_normalize_list_of_folders(tmp_path):
    from immersionlab.telemed._extract import _normalize_sources

    fa = tmp_path / "fa"
    fb = tmp_path / "fb"
    _make_synthetic_tvd_files(fa, ["a1.tvd", "a2.tvd"])
    _make_synthetic_tvd_files(fb, ["b1.tvd"])
    out = _normalize_sources([fa, fb], recursive=True, pattern="*.tvd")
    assert sorted(p.name for p in out) == ["a1.tvd", "a2.tvd", "b1.tvd"]


def test_normalize_mixed_files_and_folders(tmp_path):
    from immersionlab.telemed._extract import _normalize_sources

    folder = tmp_path / "data"
    standalone = tmp_path / "loose.tvd"
    _make_synthetic_tvd_files(folder, ["x.tvd"])
    standalone.write_bytes(b"")
    out = _normalize_sources([folder, standalone], recursive=True, pattern="*.tvd")
    assert sorted(p.name for p in out) == ["loose.tvd", "x.tvd"]


def test_normalize_dedupes_overlapping_roots(tmp_path):
    """Passing the same folder twice -- or nested parents -- should
    yield each file once."""
    from immersionlab.telemed._extract import _normalize_sources

    sub = tmp_path / "sub"
    _make_synthetic_tvd_files(sub, ["a.tvd"])
    # parent + child + the file itself, plus the file again
    out = _normalize_sources(
        [tmp_path, sub, sub / "a.tvd", sub / "a.tvd"],
        recursive=True,
        pattern="*.tvd",
    )
    assert len(out) == 1
    assert out[0].name == "a.tvd"


def test_normalize_skips_nonexistent_entries(tmp_path):
    """Non-existent paths should be silently skipped (not raise) --
    they just won't show up in the results dict."""
    from immersionlab.telemed._extract import _normalize_sources

    real = _make_synthetic_tvd_files(tmp_path, ["real.tvd"])[0]
    out = _normalize_sources(
        [real, tmp_path / "ghost.tvd", tmp_path / "no_such_folder"],
        recursive=True,
        pattern="*.tvd",
    )
    assert out == [real]


def test_normalize_pattern_kwarg(tmp_path):
    from immersionlab.telemed._extract import _normalize_sources

    _make_synthetic_tvd_files(tmp_path, ["a.tvd", "b.dat", "c.tvd"])
    out = _normalize_sources(tmp_path, recursive=True, pattern="*.dat")
    assert [p.name for p in out] == ["b.dat"]


def test_is_network_path_heuristic():
    from immersionlab.telemed._extract import _is_network_path

    assert _is_network_path(Path(r"\\server\share\file.tvd"))
    assert _is_network_path(Path("//server/share/file.tvd"))
    assert _is_network_path(Path("M:/data/file.tvd"))
    assert _is_network_path(Path("s:/data/file.tvd"))
    assert not _is_network_path(Path("C:/data/file.tvd"))
    assert not _is_network_path(Path("c:/data/file.tvd"))


def test_export_with_no_matches_returns_empty(tmp_path):
    """``export()`` on an empty folder is a no-op (no COM connection
    is attempted, so this test runs without EchoWave)."""
    from immersionlab import telemed

    empty = tmp_path / "empty_dir"
    empty.mkdir()
    out = telemed.export(empty)
    assert out == {}


# ---------- Background-prefetch staging primitives ----------


def test_stage_one_no_copy_passthrough(tmp_path):
    """``use_copy=False`` returns a _StagedFile whose local paths are
    just the originals (no file copy made, no stage_dir to clean up)."""
    from immersionlab.telemed._extract import _stage_one

    src = tmp_path / "x.tvd"
    src.write_bytes(b"abc")
    dst = tmp_path / "x.tvd.h5"
    staged = _stage_one(src, dst, use_copy=False, temp_root=tmp_path)
    assert staged.src_tvd == src
    assert staged.dst_h5 == dst
    assert staged.local_tvd == src
    assert staged.local_h5 == dst
    assert staged.stage_dir is None


def test_stage_one_with_copy_makes_local(tmp_path):
    """``use_copy=True`` copies the source into a fresh temp dir and
    points local_tvd / local_h5 there."""
    from immersionlab.telemed._extract import _stage_one

    src = tmp_path / "src.tvd"
    src.write_bytes(b"hello")
    dst = tmp_path / "out" / "src.tvd.h5"
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    staged = _stage_one(src, dst, use_copy=True, temp_root=temp_root)
    assert staged.stage_dir is not None
    assert staged.stage_dir.is_dir()
    assert staged.stage_dir.parent == temp_root
    assert staged.local_tvd.is_file()
    assert staged.local_tvd.read_bytes() == b"hello"
    assert staged.local_h5 == staged.local_tvd.with_suffix(".tvd.h5")


def test_unstage_one_upload_true_copies_and_cleans(tmp_path):
    """``upload=True`` copies the local .h5 to dst_h5 and removes the
    staging directory."""
    from immersionlab.telemed._extract import _StagedFile, _unstage_one

    stage = tmp_path / "stage"
    stage.mkdir()
    local_tvd = stage / "x.tvd"
    local_h5 = stage / "x.tvd.h5"
    local_tvd.write_bytes(b"")
    local_h5.write_bytes(b"result-bytes")
    dst = tmp_path / "dst" / "x.tvd.h5"
    dst.parent.mkdir()
    staged = _StagedFile(
        src_tvd=Path("M:/fake/x.tvd"), dst_h5=dst,
        local_tvd=local_tvd, local_h5=local_h5, stage_dir=stage,
    )
    _unstage_one(staged, upload=True)
    assert dst.read_bytes() == b"result-bytes"
    assert not stage.exists()


def test_unstage_one_upload_false_skips_copy(tmp_path):
    """``upload=False`` cleans up local temp but does NOT copy back --
    used when extract failed and there's nothing valid to upload."""
    from immersionlab.telemed._extract import _StagedFile, _unstage_one

    stage = tmp_path / "stage"
    stage.mkdir()
    local_h5 = stage / "x.tvd.h5"
    local_h5.write_bytes(b"partial")
    dst = tmp_path / "dst" / "x.tvd.h5"
    dst.parent.mkdir()
    staged = _StagedFile(
        src_tvd=Path("M:/fake/x.tvd"), dst_h5=dst,
        local_tvd=stage / "x.tvd", local_h5=local_h5, stage_dir=stage,
    )
    _unstage_one(staged, upload=False)
    assert not dst.exists()
    assert not stage.exists()


def test_unstage_one_no_copy_is_noop(tmp_path):
    """``stage_dir=None`` means no local copy was made -- unstage
    should be a no-op (don't try to delete the source!)."""
    from immersionlab.telemed._extract import _StagedFile, _unstage_one

    src = tmp_path / "x.tvd"
    src.write_bytes(b"original")
    dst = tmp_path / "x.tvd.h5"
    dst.write_bytes(b"sidecar")
    staged = _StagedFile(
        src_tvd=src, dst_h5=dst,
        local_tvd=src, local_h5=dst, stage_dir=None,
    )
    _unstage_one(staged, upload=True)
    # Both files survive unchanged.
    assert src.read_bytes() == b"original"
    assert dst.read_bytes() == b"sidecar"
