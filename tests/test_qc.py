"""Tests for the completeness-QC + LUT-inversion-guard surfaces.

Covers:
* ``telemed.read_tvd_n_frames`` -- parsing the recorded frame count out
  of a synthetic "UIFF" (.tvd) header (64-bit chunk sizes).
* the LUT-inversion detector (``_samples_look_lut_inverted``) on
  dark- vs bright-background frames.
* ``telemed.verify_complete`` -- complete / truncated / no-reference
  classification against the stored ``tvd_declared_n_frames`` attr and
  the sibling ``.tvd`` header.
* ``telemed.backfill_tvd_n_frames`` -- writing the attr into an existing
  sidecar.

All synthetic; no COM, EchoWave, or real fixtures required.
"""

from __future__ import annotations

import struct
from pathlib import Path

import h5py
import numpy as np
import pytest

from telemed._extract import (
    TelemedRoi,
    _samples_look_lut_inverted,
    read_tvd_frame_ticks,
    read_tvd_n_frames,
    read_tvd_time_ms,
)

# ---------- Synthetic .tvd header builder ----------


def _uiff_chunk(cid: bytes, payload: bytes) -> bytes:
    """One UIFF chunk: 4-byte id + 8-byte LE size + payload."""
    return cid + struct.pack("<Q", len(payload)) + payload


def _make_synthetic_tvd(path: Path, recorded_n_frames: int, *, magic: bytes = b"UIFF") -> Path:
    """Write a minimal "UIFF" container whose first ``strh`` chunk carries
    ``recorded_n_frames`` at the documented payload offset (0x14).

    Layout mirrors the real header enough for the walker: a top-level
    form chunk (``UIFF`` + 4-byte form type) containing a ``strh`` chunk.
    """
    strh_payload = bytearray(60)
    struct.pack_into("<I", strh_payload, 0x14, recorded_n_frames)
    strh = _uiff_chunk(b"strh", bytes(strh_payload))
    body = b"UDI " + strh  # 4-byte form type, then children
    container = magic + struct.pack("<Q", len(body)) + body
    path.write_bytes(container)
    return path


def test_read_tvd_n_frames_parses_recorded_count(tmp_path):
    f = _make_synthetic_tvd(tmp_path / "rec.tvd", recorded_n_frames=62955)
    assert read_tvd_n_frames(f) == 62955


def test_read_tvd_n_frames_rejects_non_tvd(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_bytes(b"RIFF\x00\x00\x00\x00AVI not a tvd at all")
    assert read_tvd_n_frames(bad) is None


def test_read_tvd_n_frames_missing_file(tmp_path):
    assert read_tvd_n_frames(tmp_path / "absent.tvd") is None


def test_read_tvd_n_frames_no_strh_returns_none(tmp_path):
    # Valid magic, but no strh chunk in the header window.
    body = b"UDI " + _uiff_chunk(b"junk", b"\x00" * 16)
    (tmp_path / "headless.tvd").write_bytes(b"UIFF" + struct.pack("<Q", len(body)) + body)
    assert read_tvd_n_frames(tmp_path / "headless.tvd") is None


# ---- per-frame timing (COM-free) ----


def _frame_chunk(end_tick: int, *, cid: bytes = b"00bb", payload: int = 16) -> bytes:
    """One frame chunk: 64-byte header (uint64 end-tick at offset 24) + dummy pixel payload."""
    header = bytearray(64)
    struct.pack_into("<Q", header, 24, end_tick)
    return _uiff_chunk(cid, bytes(header) + b"\x00" * payload)


def _make_synthetic_timed_tvd(path: Path, ticks, *, cid: bytes = b"00bb") -> Path:
    body = b"UDI " + b"".join(_frame_chunk(t, cid=cid) for t in ticks)
    path.write_bytes(b"UIFF" + struct.pack("<Q", len(body)) + body)
    return path


def test_read_tvd_time_ms_bit_exact():
    import numpy as np

    # end ticks (100 ns) with the real 14/16 ms oscillation; time_ms = (tick - tick0)/10000.
    t0 = 2_566_459_391
    ticks = [t0, t0 + 149_523, t0 + 308_580, t0 + 448_604, t0 + 608_772]
    f = _make_synthetic_timed_tvd(Path(__file__).parent / "_tmp_timed.tvd", ticks)
    try:
        got = read_tvd_time_ms(f)
        assert read_tvd_frame_ticks(f) == ticks
        expected = (np.asarray(ticks, dtype=np.int64) - ticks[0]) / 10000.0
        assert np.array_equal(got, expected)            # bit-exact, not approximate
        assert got[0] == 0.0
    finally:
        f.unlink()


def test_read_tvd_time_ms_uint64_no_wrap(tmp_path):
    # A long recording pushes the tick past 2**32 -- the 8-byte read must not wrap.
    import numpy as np

    big = 5_000_000_000          # > 2**32
    ticks = [big, big + 149_000, big + 298_000]
    f = _make_synthetic_timed_tvd(tmp_path / "long.tvd", ticks)
    got = read_tvd_time_ms(f)
    assert np.array_equal(got, np.array([0.0, 14.9, 29.8]))


def test_read_tvd_time_ms_non_tvd_returns_none(tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_bytes(b"RIFF\x00\x00\x00\x00not a tvd")
    assert read_tvd_time_ms(bad) is None
    assert read_tvd_frame_ticks(bad) is None


# ---------- LUT-inversion detector ----------


def _roi(w=40, h=40):
    # 1-based inclusive coords covering a w x h panel at the top-left.
    return TelemedRoi(
        img_id=1,
        x1=1,
        x2=w,
        y1=1,
        y2=h,
        width=w,
        height=h,
        physical_dx_cm_per_px=0.01,
        physical_dy_cm_per_px=0.01,
    )


def test_lut_detector_flags_bright_background():
    bright = [np.full((40, 40), 240, np.uint8) for _ in range(3)]
    assert _samples_look_lut_inverted(bright, _roi()) is True


def test_lut_detector_passes_dark_background():
    # Sparse bright echoes on a near-black field -- normal B-mode.
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(3):
        fr = np.zeros((40, 40), np.uint8)
        fr[rng.integers(0, 40, 30), rng.integers(0, 40, 30)] = 255
        frames.append(fr)
    assert _samples_look_lut_inverted(frames, _roi()) is False


def test_lut_detector_empty_samples_is_false():
    assert _samples_look_lut_inverted([], _roi()) is False


# ---------- verify_complete ----------


def _make_h5(path: Path, n_frames: int, *, declared=None, frames_fill=None) -> Path:
    """Minimal sidecar with the attrs verify_complete reads.

    ``frames_fill``: if given, write a ``/frames/gray`` group filled
    with this constant uint8 value (for looks_lut_inverted tests).
    """
    with h5py.File(path, "w") as h5:
        h5.attrs["n_frames"] = n_frames
        h5.attrs["full_frame_width"] = 96
        h5.attrs["full_frame_height"] = 64
        h5.attrs["n_b_images"] = 1
        h5.attrs["roi1_x1"] = 10
        h5.attrs["roi1_x2"] = 50
        h5.attrs["roi1_y1"] = 5
        h5.attrs["roi1_y2"] = 45
        h5.attrs["roi1_width"] = 41
        h5.attrs["roi1_height"] = 41
        h5.attrs["physical_dx1_cm_per_px"] = 0.01
        h5.attrs["physical_dy1_cm_per_px"] = 0.01
        h5.attrs["source_tvd_path"] = "C:/synthetic/test.tvd"
        h5.attrs["extracted_at_iso"] = "2026-05-27T00:00:00"
        h5.attrs["schema_version"] = "v1"
        if declared is not None:
            h5.attrs["tvd_declared_n_frames"] = declared
        tg = h5.create_group("timing")
        tg.create_dataset("frame_idx_1n", data=np.arange(1, n_frames + 1, dtype=np.int32))
        tg.create_dataset("time_ms", data=np.arange(n_frames, dtype=np.float64) * 15.0)
        tg.create_dataset("ifi_ms", data=np.full(n_frames, 15.0, dtype=np.float64))
        if frames_fill is not None:
            arr = np.full((n_frames, 64, 96), frames_fill, dtype=np.uint8)
            h5.create_group("frames").create_dataset("gray", data=arr, chunks=(1, 64, 96))
    return path


def test_verify_complete_marks_complete_within_tolerance(tmp_path):
    from telemed import verify_complete

    # declared = extracted + 2 (the benign header overcount).
    _make_h5(tmp_path / "ok.tvd.h5", n_frames=1000, declared=1002)
    res = verify_complete(tmp_path, progress=False)
    info = res[str(tmp_path / "ok.tvd.h5")]
    assert info["status"] == "complete"
    assert info["extracted"] == 1000
    assert info["declared"] == 1002


def test_verify_complete_flags_truncation(tmp_path):
    from telemed import verify_complete

    # Extracted far below the .tvd-declared count -> memory truncation.
    _make_h5(tmp_path / "short.tvd.h5", n_frames=12000, declared=90000)
    res = verify_complete(tmp_path, progress=False)
    info = res[str(tmp_path / "short.tvd.h5")]
    assert info["status"] == "truncated"
    assert info["issues"] and "78000 missing" in info["issues"][0]


def test_verify_complete_unknown_without_reference(tmp_path):
    from telemed import verify_complete

    # No stored attr and no sibling .tvd -> can't tell.
    _make_h5(tmp_path / "lonely.tvd.h5", n_frames=500, declared=None)
    res = verify_complete(tmp_path, progress=False)
    assert res[str(tmp_path / "lonely.tvd.h5")]["status"] == "unknown"


def test_verify_complete_falls_back_to_sibling_tvd(tmp_path):
    from telemed import verify_complete

    # No stored attr, but the sibling .tvd header is parseable.
    _make_h5(tmp_path / "rec.tvd.h5", n_frames=2000, declared=None)
    _make_synthetic_tvd(tmp_path / "rec.tvd", recorded_n_frames=2002)
    info = verify_complete(tmp_path, progress=False)[str(tmp_path / "rec.tvd.h5")]
    assert info["status"] == "complete"
    assert info["declared"] == 2002


# ---------- backfill ----------


def test_backfill_adds_attr_from_sibling_tvd(tmp_path):
    from telemed import backfill_tvd_n_frames

    h5 = _make_h5(tmp_path / "rec.tvd.h5", n_frames=2000, declared=None)
    _make_synthetic_tvd(tmp_path / "rec.tvd", recorded_n_frames=2002)
    res = backfill_tvd_n_frames(tmp_path, progress=False)
    assert res[str(h5)] == "added (2002)"
    with h5py.File(h5, "r") as f:
        assert int(f.attrs["tvd_declared_n_frames"]) == 2002


def test_backfill_skips_when_no_sibling_tvd(tmp_path):
    from telemed import backfill_tvd_n_frames

    h5 = _make_h5(tmp_path / "orphan.tvd.h5", n_frames=2000, declared=None)
    res = backfill_tvd_n_frames(tmp_path, progress=False)
    assert res[str(h5)] == "skipped: no sibling .tvd"


def test_backfill_reports_update_when_attr_exists(tmp_path):
    from telemed import backfill_tvd_n_frames

    h5 = _make_h5(tmp_path / "rec.tvd.h5", n_frames=2000, declared=1)
    _make_synthetic_tvd(tmp_path / "rec.tvd", recorded_n_frames=2002)
    res = backfill_tvd_n_frames(tmp_path, progress=False)
    assert res[str(h5)] == "updated (2002)"


# ---------- Log surface ----------


def test_log_exposes_declared_n_frames(tmp_path):
    from telemed import Log

    _make_h5(tmp_path / "rec.tvd.h5", n_frames=2000, declared=2002)
    assert Log(tmp_path / "rec.tvd.h5").tvd_declared_n_frames == 2002


def test_log_declared_n_frames_none_when_absent(tmp_path):
    from telemed import Log

    _make_h5(tmp_path / "rec.tvd.h5", n_frames=2000, declared=None)
    assert Log(tmp_path / "rec.tvd.h5").tvd_declared_n_frames is None


# ---------- looks_lut_inverted ----------


def test_looks_lut_inverted_flags_bright_sidecar(tmp_path):
    from telemed import looks_lut_inverted

    _make_h5(tmp_path / "inv.tvd.h5", n_frames=5, frames_fill=240)
    assert looks_lut_inverted(tmp_path / "inv.tvd.h5") is True


def test_looks_lut_inverted_passes_dark_sidecar(tmp_path):
    from telemed import looks_lut_inverted

    _make_h5(tmp_path / "ok.tvd.h5", n_frames=5, frames_fill=8)
    assert looks_lut_inverted(tmp_path / "ok.tvd.h5") is False


def test_looks_lut_inverted_false_without_frames(tmp_path):
    from telemed import looks_lut_inverted

    # No /frames/gray (frames=False extract) -> nothing to judge.
    _make_h5(tmp_path / "nf.tvd.h5", n_frames=5, frames_fill=None)
    assert looks_lut_inverted(tmp_path / "nf.tvd.h5") is False


# ---- end-tick caching (sidecar) ----


def test_read_tvd_frame_ticks_cache_roundtrip(tmp_path):
    """cache=True writes a sibling <stem>.tvd.ticks.npy; a present sidecar serves the ticks even
    after the .tvd itself is gone (the read avoids re-walking the container)."""
    from telemed._extract import _ticks_sidecar_path

    ticks = [100, 100 + 149_000, 100 + 298_000, 100 + 449_000]
    f = _make_synthetic_timed_tvd(tmp_path / "c.tvd", ticks)
    assert read_tvd_frame_ticks(f, cache=True) == ticks
    sc = _ticks_sidecar_path(f)
    assert sc.is_file() and sc.name == "c.tvd.ticks.npy"
    f.unlink()                                            # drop the .tvd; sidecar must still serve
    assert read_tvd_frame_ticks(f, cache=True) == ticks


def test_read_tvd_time_ms_uses_cached_ticks(tmp_path):
    import numpy as np

    ticks = [5_000_000_000, 5_000_000_000 + 149_000, 5_000_000_000 + 298_000]
    f = _make_synthetic_timed_tvd(tmp_path / "t.tvd", ticks)
    read_tvd_time_ms(f, cache=True)                       # populate sidecar
    f.unlink()
    got = read_tvd_time_ms(f, cache=True)                 # served from sidecar only
    assert np.array_equal(got, np.array([0.0, 14.9, 29.8]))


def test_cache_false_writes_no_sidecar(tmp_path):
    from telemed._extract import _ticks_sidecar_path

    f = _make_synthetic_timed_tvd(tmp_path / "n.tvd", [1, 1 + 149_000])
    read_tvd_frame_ticks(f)                               # cache defaults to False
    assert not _ticks_sidecar_path(f).exists()
