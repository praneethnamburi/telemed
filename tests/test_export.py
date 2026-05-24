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
