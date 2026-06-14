"""Tests for the fully COM-free path (telemed._comfree).

A synthetic ``.tvd`` builder exercises the container streaming + strf geometry + h5/mp4 writing
end-to-end (no real device file needed); separate units cover the .NET-BinaryFormatter reader and
the ffmpeg argv builder.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from telemed import _comfree


# ---------- synthetic .tvd builder ----------


def _lpstr(s: str) -> bytes:
    b = s.encode("utf-8")
    assert len(b) < 128
    return bytes([len(b)]) + b


def _leaf(cid: bytes, payload: bytes) -> bytes:
    out = cid + struct.pack("<Q", len(payload)) + payload
    return out + (b"\x00" if len(payload) & 1 else b"")


def _container(cid: bytes, formtype: bytes, children: list) -> bytes:
    body = formtype + b"".join(children)
    out = cid + struct.pack("<Q", len(body)) + body
    return out + (b"\x00" if len(body) & 1 else b"")


def _strh(stream_type: bytes, declared: int) -> bytes:
    b = bytearray(60)
    b[0:len(stream_type)] = stream_type
    struct.pack_into("<I", b, 0x14, declared)
    return bytes(b)


def _strf(samples: int, active: int, total: int) -> bytes:
    # body after the 'vids' fourcc: read_geometry finds the [8, samples, total] triad and the
    # recurring active value in [samples/2, samples).
    g = struct.pack("<6I", active, active, 8, samples, total, 0)
    return b"vids" + g


def _frame(cid: bytes, total: int, end_tick: int, fill: int) -> bytes:
    sub = struct.pack("<12I", 64, 0, 281, total, 0, 0,
                      end_tick & 0xFFFFFFFF, end_tick >> 32, 0, 0, 0, total)
    return _leaf(cid, sub + bytes([fill]) * total)


def make_synth_tvd(path: Path, *, lines=96, samples=512, active=384, n_frames=5, n_probes=2) -> None:
    total = lines * samples
    hdr_children: list = []
    for p in range(n_probes):
        st = b"2DusB   " if p == 0 else b"2DusB" + str(p + 1).encode() + b"  "
        hdr_children += [_leaf(b"strh", _strh(st, n_frames)), _leaf(b"strf", _strf(samples, active, total))]
    hdrl = _container(b"LIST", b"hdrl", hdr_children)
    frames: list = []
    for k in range(n_frames):
        for p in range(n_probes):
            cid = b"00bb" if p == 0 else b"01bb"
            frames.append(_frame(cid, total, end_tick=10000 * (k + 1) + p, fill=(k * 7 + p) % 200 + 1))
    movi = _container(b"LIST", b"movi", frames)
    path.write_bytes(_container(b"UIFF", b"UDI ", [hdrl, movi]))


# ---------- geometry ----------


def test_read_geometry_synth(tmp_path):
    tvd = tmp_path / "syn.tvd"
    make_synth_tvd(tvd, lines=96, samples=512, active=384, n_frames=3, n_probes=2)
    g = _comfree.read_geometry(tvd)
    assert g is not None
    assert (g.lines, g.samples, g.active, g.datasize) == (96, 512, 384, 96 * 512)


def test_read_geometry_rejects_non_tvd(tmp_path):
    p = tmp_path / "x.tvd"
    p.write_bytes(b"NOTUIFF" + b"\x00" * 100)
    assert _comfree.read_geometry(p) is None


# ---------- end-to-end extract (synthetic) ----------


def _have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not on PATH")
def test_extract_one_dual_probe(tmp_path):
    tvd = tmp_path / "rec.tvd"
    make_synth_tvd(tvd, lines=96, samples=512, active=384, n_frames=6, n_probes=2)
    out = tmp_path / "out"
    r = _comfree.extract_one(tvd, out, progress=False)

    # two probe videos, native lines x active, full declared frame count each
    vids = sorted(Path(v).name for v in r["videos"])
    assert vids == ["rec_b1.mp4", "rec_b2.mp4"]
    assert r["counts"] == {"_b1": 6, "_b2": 6}
    assert r["geometry"] == (96, 384)
    for v in r["videos"]:
        assert Path(v).exists() and Path(v).stat().st_size > 0

    # h5: metadata + timing, NO frames
    with h5py.File(r["h5"], "r") as h:
        assert h.attrs["backend"] == "comfree"
        assert h.attrs["schema_version"] == "comfree-v1"
        assert h.attrs["n_b_images"] == 2
        assert h.attrs["n_frames"] == 6
        assert h.attrs["probe1_video"] == "rec_b1.mp4"
        assert h.attrs["probe2_stream"] == "01bb"
        assert h.attrs["probe1_lines"] == 96 and h.attrs["probe1_active"] == 384
        assert "frames" not in h            # image data lives in the mp4s, not the h5
        assert h["timing/time_ms"].shape == (6,)
        assert float(h["timing/time_ms"][0]) == 0.0


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not on PATH")
def test_extract_one_single_probe(tmp_path):
    tvd = tmp_path / "mono.tvd"
    make_synth_tvd(tvd, lines=96, samples=512, active=384, n_frames=4, n_probes=1)
    out = tmp_path / "out"
    r = _comfree.extract_one(tvd, out, progress=False)
    assert [Path(v).name for v in r["videos"]] == ["mono.mp4"]    # single probe -> <stem>.mp4
    assert r["counts"] == {"single": 4}


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not on PATH")
def test_extract_comfree_skip_existing(tmp_path):
    tvd = tmp_path / "rec.tvd"
    make_synth_tvd(tvd, n_frames=3)
    out = tmp_path / "out"
    r1 = _comfree.extract_comfree(tvd, out, progress=False)
    assert r1[str(tvd)] == "built"
    r2 = _comfree.extract_comfree(tvd, out, progress=False)   # h5 exists -> skip
    assert r2[str(tvd)] == "hit"


# ---------- .NET BinaryFormatter reader ----------


def _bf_class(name: str, int_members: list, bool_members: list) -> bytes:
    """A minimal ClassWithMembersAndTypes record with Int32 + Boolean primitive members."""
    members = [(n, v, 8) for n, v in int_members] + [(n, v, 1) for n, v in bool_members]
    body = b"\x05" + struct.pack("<i", 1)            # rt=5, ObjectId=1
    body += _lpstr(name)
    body += struct.pack("<i", len(members))
    for n, _, _ in members:
        body += _lpstr(n)
    body += bytes([0] * len(members))               # BinaryTypeEnum: all Primitive
    body += bytes([m[2] for m in members])          # PrimitiveTypeEnum per member
    body += struct.pack("<i", 0)                    # LibraryId
    for _, v, pe in members:
        body += struct.pack("<i", v) if pe == 8 else bytes([1 if v else 0])
    return body


def test_parse_named_class_primitives():
    blob = b"\x00" * 16 + _bf_class(
        "EchoWave.UsgHWSettings",
        int_members=[("<x>b_gain", 87), ("<x>b_depth", 50), ("<x>b_compound_frames_number", 3)],
        bool_members=[("<x>b_speckle_filtration_enabled_real", True)],
    ) + b"\x00" * 8
    out = _comfree._parse_named_class(blob, b"EchoWave.UsgHWSettings")
    assert out["b_gain"] == 87
    assert out["b_depth"] == 50
    assert out["b_compound_frames_number"] == 3
    assert out["b_speckle_filtration_enabled_real"] is True


def test_parse_named_class_missing_returns_empty():
    assert _comfree._parse_named_class(b"no such class here", b"EchoWave.Nope") == {}


# ---------- ffmpeg argv builder ----------


def test_ffmpeg_cmd_lossless_native_with_sar():
    cmd = _comfree._ffmpeg_cmd(Path("o.mp4"), w=116, h=884, fps=73.9459, lossless=True, crf=24,
                               preset="ultrafast", sar=9.64, hflip=False, overwrite=True)
    assert cmd[0] == "ffmpeg"
    assert "-s" in cmd and cmd[cmd.index("-s") + 1] == "116x884"
    assert "libx265" in cmd
    assert "lossless=1" in cmd
    assert "-crf" not in cmd
    vf = cmd[cmd.index("-vf") + 1]
    assert vf == "setsar=9.64000"


def test_ffmpeg_cmd_crf_and_hflip():
    cmd = _comfree._ffmpeg_cmd(Path("o.mp4"), w=8, h=12, fps=30.0, lossless=False, crf=18,
                               preset="veryfast", sar=None, hflip=True, overwrite=False)
    assert "-crf" in cmd and cmd[cmd.index("-crf") + 1] == "18"
    assert "lossless=1" not in cmd
    assert cmd[cmd.index("-vf") + 1] == "hflip"     # hflip applied, no setsar when sar is None
    assert "-n" in cmd                              # overwrite=False -> -n
