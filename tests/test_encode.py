"""Tests for ``immersionlab.telemed._encode`` and the
``telemed.export(kind=...)`` dispatcher.

Most tests monkeypatch ``_encode_frames`` so we never actually shell
ffmpeg; the cmd shape is pinned byte-for-byte and the dispatch /
multi-panel / orientation / progress paths are exercised against
synthetic ``.tvd.h5`` fixtures built inline. One opt-in test pipes
through real ffmpeg to catch Popen/stdin plumbing regressions
(skipped if ffmpeg is not on PATH).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

# Force Agg in case anything down the import chain touches matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")


# ---------- Synthetic-sidecar helpers (v4-shaped) ----------


def _make_synthetic_h5(
    path: Path, *,
    n_frames: int = 5,
    h: int = 64, w: int = 96,
    include_frames: bool = True,
    rois: dict | None = None,
    params: dict | None = None,
) -> Path:
    """Write a minimal v4-shape ``.tvd.h5`` for the encode tests.

    Default: single-probe img_id=1 ROI of (10..50, 5..45) at 0.01 cm/px.
    Pass ``rois={1: {...}, 2: {...}}`` for multi-probe fixtures; pass
    ``params={'b_is_scan_direction_changed': True}`` etc. to exercise
    orientation normalisation.
    """
    if rois is None:
        rois = {1: dict(x1=10, x2=50, y1=5, y2=45, width=41, height=41,
                        dx=0.01, dy=0.01)}
    times = np.cumsum([0.0] + [14.5 + 0.5 * (i % 3) for i in range(n_frames - 1)])
    ifi = np.zeros(n_frames)
    ifi[1:] = np.diff(times)
    with h5py.File(path, "w") as h5:
        h5.attrs["n_frames"] = n_frames
        h5.attrs["full_frame_width"] = w
        h5.attrs["full_frame_height"] = h
        h5.attrs["n_b_images"] = len(rois)
        for img_id, r in rois.items():
            h5.attrs[f"roi{img_id}_x1"] = r["x1"]
            h5.attrs[f"roi{img_id}_x2"] = r["x2"]
            h5.attrs[f"roi{img_id}_y1"] = r["y1"]
            h5.attrs[f"roi{img_id}_y2"] = r["y2"]
            h5.attrs[f"roi{img_id}_width"] = r["width"]
            h5.attrs[f"roi{img_id}_height"] = r["height"]
            h5.attrs[f"physical_dx{img_id}_cm_per_px"] = r["dx"]
            h5.attrs[f"physical_dy{img_id}_cm_per_px"] = r["dy"]
        h5.attrs["source_tvd_path"] = "C:/synthetic/test.tvd"
        h5.attrs["extracted_at_iso"] = "2026-05-24T00:00:00"
        h5.attrs["schema_version"] = 4
        if params:
            for k, v in params.items():
                h5.attrs[f"param_{k}"] = v
        tg = h5.create_group("timing")
        tg.create_dataset("frame_idx_1n", data=np.arange(1, n_frames + 1, dtype=np.int32))
        tg.create_dataset("time_ms", data=times)
        tg.create_dataset("ifi_ms", data=ifi)
        if include_frames:
            arr = np.zeros((n_frames, h, w), dtype=np.uint8)
            for i in range(n_frames):
                arr[i] = (np.linspace(0, 255, w, dtype=np.float32)[None, :]
                          .repeat(h, axis=0) * (1.0 - i / max(n_frames - 1, 1))).astype(np.uint8)
            h5.create_group("frames").create_dataset(
                "gray", data=arr, chunks=(1, h, w),
            )
    return path


def _patch_encode_frames(monkeypatch):
    """Replace ``_encode_frames`` with a fake that captures per-call
    ``(cmd, n_frames_written, frame_shape)`` and touches the output mp4
    path so downstream existence checks see it."""
    from immersionlab.telemed import _encode as _enc

    captures: list[dict] = []

    def fake(cmd, frames_iter):
        cap = {"cmd": cmd, "n_frames": 0, "shape": None}
        for fr in frames_iter:
            cap["n_frames"] += 1
            cap["shape"] = fr.shape
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"")
        captures.append(cap)

    monkeypatch.setattr(_enc, "_encode_frames", fake)
    return captures


# ---------- ffmpeg cmd builder pins ----------


class TestBuildFfmpegCmd:
    """Pin the default cmd shape so encode flags can't silently drift.

    Default = lossless h265 mono + preset slow + ``-fps_mode cfr``,
    ``-hide_banner -loglevel error`` mandatory (else libx265's stderr
    deadlocks the Popen PIPE).
    """

    def test_lossless_default_byte_identical(self):
        from immersionlab.telemed._encode import _build_ffmpeg_cmd

        cmd = _build_ffmpeg_cmd(
            "out.mp4",
            width=41, height=41, fps=68.5,
            codec="h265_mono", lossless=True,
            crf=24, preset="slow", vf_chain=None, overwrite=False,
        )
        assert cmd == [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-n",
            "-f", "rawvideo",
            "-pix_fmt", "gray",
            "-s", "41x41",
            "-r", "68.500000",
            "-i", "-",
            "-c:v", "libx265",
            "-pix_fmt", "gray",
            "-x265-params", "lossless=1",
            "-preset", "slow",
            "-fps_mode", "cfr",
            "-an",
            "out.mp4",
        ]

    def test_lossy_branch_uses_crf(self):
        """``lossless=False`` flips ``-x265-params lossless=1`` for ``-crf N``."""
        from immersionlab.telemed._encode import _build_ffmpeg_cmd

        cmd = _build_ffmpeg_cmd(
            "out.mp4", width=10, height=10, fps=30.0,
            codec="h265_mono", lossless=False,
            crf=22, preset="slow", vf_chain=None, overwrite=False,
        )
        assert "-crf" in cmd and cmd[cmd.index("-crf") + 1] == "22"
        assert "-x265-params" not in cmd
        assert "lossless=1" not in cmd

    def test_overwrite_flips_yn_flag(self):
        from immersionlab.telemed._encode import _build_ffmpeg_cmd

        cmd = _build_ffmpeg_cmd(
            "out.mp4", width=10, height=10, fps=30.0,
            codec="h265_mono", lossless=True,
            crf=24, preset="slow", vf_chain=None, overwrite=True,
        )
        assert "-y" in cmd and "-n" not in cmd

    def test_vf_chain_inserts_filters(self):
        """``vf_chain`` items become a single ``-vf a,b,c`` flag."""
        from immersionlab.telemed._encode import _build_ffmpeg_cmd

        cmd = _build_ffmpeg_cmd(
            "out.mp4", width=10, height=10, fps=30.0,
            codec="h265_mono", lossless=True,
            crf=24, preset="slow", vf_chain=["hflip", "transpose=1"],
            overwrite=False,
        )
        assert "-vf" in cmd
        assert cmd[cmd.index("-vf") + 1] == "hflip,transpose=1"

    def test_empty_vf_chain_omits_flag(self):
        from immersionlab.telemed._encode import _build_ffmpeg_cmd

        cmd = _build_ffmpeg_cmd(
            "out.mp4", width=10, height=10, fps=30.0,
            codec="h265_mono", lossless=True,
            crf=24, preset="slow", vf_chain=[], overwrite=False,
        )
        assert "-vf" not in cmd

    def test_unknown_codec_raises(self):
        from immersionlab.telemed._encode import _build_ffmpeg_cmd

        with pytest.raises(ValueError, match="not supported"):
            _build_ffmpeg_cmd(
                "out.mp4", width=10, height=10, fps=30.0,
                codec="vp9", lossless=True,
                crf=24, preset="slow", vf_chain=None, overwrite=False,
            )


# ---------- Orientation normalisation ----------


def test_orientation_vf_default_empty():
    """No scan-direction-changed, no rotation -> empty filter chain."""
    from immersionlab.telemed._encode import _orientation_vf

    assert _orientation_vf({}) == []
    assert _orientation_vf({"b_is_scan_direction_changed": False, "b_rotate": 0}) == []


def test_orientation_vf_hflip_when_scan_direction_changed():
    """``b_is_scan_direction_changed=True`` -> ``hflip``."""
    from immersionlab.telemed._encode import _orientation_vf

    assert _orientation_vf({"b_is_scan_direction_changed": True}) == ["hflip"]


def test_orientation_vf_warns_on_nonzero_rotate():
    """Non-zero ``b_rotate`` is undocumented enum -> warn, don't apply."""
    from immersionlab.telemed._encode import _orientation_vf

    with pytest.warns(UserWarning, match="b_rotate=2"):
        out = _orientation_vf({"b_rotate": 2})
    # Pixels pass through untouched -- caller can investigate.
    assert "transpose" not in out and "rotate" not in out


# ---------- Output naming ----------


def test_plan_targets_single_probe_naming(tmp_path):
    """Single img_id -> ``<stem>.mp4`` (no suffix)."""
    from immersionlab.telemed._encode import _plan_targets

    h5 = tmp_path / "rec.tvd.h5"
    h5.write_bytes(b"")
    targets = _plan_targets(h5, [1])
    assert len(targets) == 1
    assert targets[0].out_path.name == "rec.mp4"
    assert targets[0].img_id == 1


def test_plan_targets_dual_probe_naming(tmp_path):
    """Multiple img_ids -> ``<stem>_b{N}.mp4`` per panel."""
    from immersionlab.telemed._encode import _plan_targets

    h5 = tmp_path / "rec.tvd.h5"
    h5.write_bytes(b"")
    targets = _plan_targets(h5, [1, 2])
    names = sorted(t.out_path.name for t in targets)
    assert names == ["rec_b1.mp4", "rec_b2.mp4"]


def test_plan_targets_strips_composite_suffix(tmp_path):
    """``<stem>.tvd.h5`` -> stem is ``<stem>``, not ``<stem>.tvd``."""
    from immersionlab.telemed._encode import _plan_targets

    h5 = tmp_path / "pia02_s018_003.tvd.h5"
    h5.write_bytes(b"")
    targets = _plan_targets(h5, [1])
    assert targets[0].out_path.name == "pia02_s018_003.mp4"


def test_plan_targets_honours_out_dir(tmp_path):
    from immersionlab.telemed._encode import _plan_targets

    h5 = tmp_path / "rec.tvd.h5"
    h5.write_bytes(b"")
    other = tmp_path / "elsewhere"
    targets = _plan_targets(h5, [1], out_dir=other)
    assert targets[0].out_path.parent == other


# ---------- _resolve_h5_sources ----------


def test_resolve_h5_sources_accepts_h5_file(tmp_path):
    from immersionlab.telemed._encode import _resolve_h5_sources

    h5 = tmp_path / "rec.tvd.h5"
    h5.write_bytes(b"")
    out = _resolve_h5_sources(h5, recursive=True, pattern="*.tvd.h5")
    assert out == [h5]


def test_resolve_h5_sources_accepts_tvd_resolves_sidecar(tmp_path):
    """Passing a .tvd file -> use its sibling .tvd.h5 if present."""
    from immersionlab.telemed._encode import _resolve_h5_sources

    tvd = tmp_path / "rec.tvd"
    h5 = tmp_path / "rec.tvd.h5"
    tvd.write_bytes(b"")
    h5.write_bytes(b"")
    out = _resolve_h5_sources(tvd, recursive=True, pattern="*.tvd.h5")
    assert out == [h5]


def test_resolve_h5_sources_tvd_without_sidecar_skipped(tmp_path):
    """.tvd with no sibling .tvd.h5 -> silently skipped."""
    from immersionlab.telemed._encode import _resolve_h5_sources

    tvd = tmp_path / "rec.tvd"
    tvd.write_bytes(b"")
    out = _resolve_h5_sources(tvd, recursive=True, pattern="*.tvd.h5")
    assert out == []


def test_resolve_h5_sources_walks_folder_recursive(tmp_path):
    from immersionlab.telemed._encode import _resolve_h5_sources

    (tmp_path / "a.tvd.h5").write_bytes(b"")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.tvd.h5").write_bytes(b"")
    out = _resolve_h5_sources(tmp_path, recursive=True, pattern="*.tvd.h5")
    assert sorted(p.name for p in out) == ["a.tvd.h5", "b.tvd.h5"]


def test_resolve_h5_sources_dedupes_overlapping(tmp_path):
    from immersionlab.telemed._encode import _resolve_h5_sources

    h5 = tmp_path / "x.tvd.h5"
    h5.write_bytes(b"")
    out = _resolve_h5_sources(
        [tmp_path, h5, h5], recursive=True, pattern="*.tvd.h5",
    )
    assert len(out) == 1


def test_resolve_h5_sources_excludes_tvd_h5old_renames(tmp_path):
    """User-renamed ``<stem>.tvd.h5OLD`` files (made when re-extracting
    under a schema bump) must NOT match the default ``*.tvd.h5`` glob.

    Regression guard surfaced 2026-05-24: C:/data/temp2 had v3 sidecars
    renamed to ``.tvd.h5OLD`` to make room for a v4 re-extract; we don't
    want the encode side picking those up alongside the fresh v4 ones.
    """
    from immersionlab.telemed._encode import _resolve_h5_sources

    (tmp_path / "a.tvd.h5").write_bytes(b"")
    (tmp_path / "b.tvd.h5OLD").write_bytes(b"")
    (tmp_path / "c.tvd").write_bytes(b"")  # standalone .tvd (no sidecar)
    out = _resolve_h5_sources(tmp_path, recursive=True, pattern="*.tvd.h5")
    assert [p.name for p in out] == ["a.tvd.h5"]


# ---------- Single-recording export_video ----------


class TestExportVideoSingleProbe:
    def test_basic_writes_one_mp4(self, tmp_path, monkeypatch):
        from immersionlab.telemed import export_video

        cap = _patch_encode_frames(monkeypatch)
        h5 = _make_synthetic_h5(tmp_path / "rec.tvd.h5", n_frames=5)
        results = export_video(h5)
        assert results == {str(tmp_path / "rec.mp4"): "built"}
        assert len(cap) == 1
        # ROI 10..50 x 5..45 -> 41x41 cropped frame.
        assert cap[0]["shape"] == (41, 41)
        assert cap[0]["n_frames"] == 5

    def test_lossless_is_default(self, tmp_path, monkeypatch):
        """No CRF in default cmd; ``-x265-params lossless=1`` present."""
        from immersionlab.telemed import export_video

        cap = _patch_encode_frames(monkeypatch)
        h5 = _make_synthetic_h5(tmp_path / "rec.tvd.h5", n_frames=3)
        export_video(h5)
        cmd = cap[0]["cmd"]
        assert "-x265-params" in cmd
        assert "lossless=1" in cmd
        assert "-crf" not in cmd

    def test_lossy_branch_uses_crf(self, tmp_path, monkeypatch):
        from immersionlab.telemed import export_video

        cap = _patch_encode_frames(monkeypatch)
        h5 = _make_synthetic_h5(tmp_path / "rec.tvd.h5", n_frames=3)
        export_video(h5, lossless=False, crf=22)
        cmd = cap[0]["cmd"]
        assert "-crf" in cmd and cmd[cmd.index("-crf") + 1] == "22"
        assert "-x265-params" not in cmd

    def test_out_dir_routes_output(self, tmp_path, monkeypatch):
        from immersionlab.telemed import export_video

        cap = _patch_encode_frames(monkeypatch)
        h5 = _make_synthetic_h5(tmp_path / "rec.tvd.h5", n_frames=3)
        other = tmp_path / "out"
        results = export_video(h5, out_dir=other)
        assert str(other / "rec.mp4") in results
        assert (other / "rec.mp4").exists()

    def test_skip_existing(self, tmp_path, monkeypatch):
        """Pre-existing mp4 -> ``hit`` status, no encode call made."""
        from immersionlab.telemed import export_video

        cap = _patch_encode_frames(monkeypatch)
        h5 = _make_synthetic_h5(tmp_path / "rec.tvd.h5", n_frames=3)
        (tmp_path / "rec.mp4").write_bytes(b"pre")
        results = export_video(h5)
        assert results[str(tmp_path / "rec.mp4")] == "hit"
        assert cap == []

    def test_no_frames_yields_error_status(self, tmp_path, monkeypatch):
        """Sidecar without /frames/gray -> error in results, no crash."""
        from immersionlab.telemed import export_video

        _patch_encode_frames(monkeypatch)
        h5 = _make_synthetic_h5(
            tmp_path / "rec.tvd.h5", n_frames=3, include_frames=False,
        )
        results = export_video(h5)
        status = results[str(tmp_path / "rec.mp4")]
        assert status.startswith("error:")
        assert "frames=False" in status or "/frames/gray" in status


# ---------- Multi-probe split ----------


class TestExportVideoDualProbe:
    """pia02-shape sidecars (n_b_images=2) split into per-panel mp4s."""

    def test_writes_two_mp4s_with_b1_b2_suffix(self, tmp_path, monkeypatch):
        from immersionlab.telemed import export_video

        cap = _patch_encode_frames(monkeypatch)
        rois = {
            1: dict(x1=10, x2=50, y1=5, y2=45, width=41, height=41,
                    dx=0.012, dy=0.013),
            2: dict(x1=60, x2=100, y1=5, y2=45, width=41, height=41,
                    dx=0.011, dy=0.014),
        }
        h5 = _make_synthetic_h5(tmp_path / "rec.tvd.h5", n_frames=4, rois=rois)
        results = export_video(h5)
        out_names = sorted(Path(p).name for p in results)
        assert out_names == ["rec_b1.mp4", "rec_b2.mp4"]
        assert all(v == "built" for v in results.values())
        # Two encode calls, one per panel.
        assert len(cap) == 2

    def test_skip_existing_per_panel(self, tmp_path, monkeypatch):
        """Per-output skip: b1 exists, b2 doesn't -> b1 hit, b2 built."""
        from immersionlab.telemed import export_video

        cap = _patch_encode_frames(monkeypatch)
        rois = {
            1: dict(x1=10, x2=50, y1=5, y2=45, width=41, height=41,
                    dx=0.01, dy=0.01),
            2: dict(x1=60, x2=100, y1=5, y2=45, width=41, height=41,
                    dx=0.01, dy=0.01),
        }
        h5 = _make_synthetic_h5(tmp_path / "rec.tvd.h5", n_frames=3, rois=rois)
        (tmp_path / "rec_b1.mp4").write_bytes(b"pre")
        results = export_video(h5)
        assert results[str(tmp_path / "rec_b1.mp4")] == "hit"
        assert results[str(tmp_path / "rec_b2.mp4")] == "built"
        # Only one encode call (the b2 build).
        assert len(cap) == 1


# ---------- Orientation normalisation end-to-end ----------


class TestExportVideoOrientation:
    def test_hflip_in_cmd_when_scan_direction_changed(self, tmp_path, monkeypatch):
        from immersionlab.telemed import export_video

        cap = _patch_encode_frames(monkeypatch)
        h5 = _make_synthetic_h5(
            tmp_path / "rec.tvd.h5", n_frames=3,
            params={"b_is_scan_direction_changed": True},
        )
        export_video(h5)
        cmd = cap[0]["cmd"]
        assert "-vf" in cmd and "hflip" in cmd[cmd.index("-vf") + 1]

    def test_no_vf_when_canonical(self, tmp_path, monkeypatch):
        from immersionlab.telemed import export_video

        cap = _patch_encode_frames(monkeypatch)
        h5 = _make_synthetic_h5(
            tmp_path / "rec.tvd.h5", n_frames=3,
            params={"b_is_scan_direction_changed": False, "b_rotate": 0},
        )
        export_video(h5)
        assert "-vf" not in cap[0]["cmd"]

    def test_normalize_orientation_false_disables(self, tmp_path, monkeypatch):
        from immersionlab.telemed import export_video

        cap = _patch_encode_frames(monkeypatch)
        h5 = _make_synthetic_h5(
            tmp_path / "rec.tvd.h5", n_frames=3,
            params={"b_is_scan_direction_changed": True},
        )
        export_video(h5, normalize_orientation=False)
        assert "-vf" not in cap[0]["cmd"]


# ---------- Folder + iterable dispatch ----------


def test_export_video_walks_folder(tmp_path, monkeypatch):
    from immersionlab.telemed import export_video

    cap = _patch_encode_frames(monkeypatch)
    _make_synthetic_h5(tmp_path / "a.tvd.h5", n_frames=2)
    _make_synthetic_h5(tmp_path / "b.tvd.h5", n_frames=2)
    results = export_video(tmp_path)
    assert sorted(Path(p).name for p in results) == ["a.mp4", "b.mp4"]
    assert len(cap) == 2


def test_export_video_progress_shows_tqdm_per_panel(tmp_path, monkeypatch):
    """``progress=True`` (default) should construct + tick a tqdm bar
    during each panel's encode. We monkeypatch ``tqdm.auto.tqdm`` to a
    capturing fake so we don't actually render a bar in tests.
    """
    from immersionlab.telemed import export_video

    _patch_encode_frames(monkeypatch)

    instances: list = []

    class _FakeTqdm:
        def __init__(self, **kw):
            self.total = kw.get("total")
            self.desc = kw.get("desc")
            self.updates = 0
            self.closed = False
            instances.append(self)

        def update(self, n):
            self.updates += n

        def close(self):
            self.closed = True

    import tqdm as _tqdm_pkg
    import tqdm.auto as _tqdm_auto
    monkeypatch.setattr(_tqdm_pkg, "tqdm", _FakeTqdm, raising=False)
    monkeypatch.setattr(_tqdm_auto, "tqdm", _FakeTqdm, raising=False)

    h5 = _make_synthetic_h5(tmp_path / "rec.tvd.h5", n_frames=5)
    export_video(h5, progress=True)

    assert len(instances) == 1
    bar = instances[0]
    assert bar.total == 5
    assert bar.desc == "rec"
    assert bar.updates == 5
    assert bar.closed is True


def test_export_video_progress_false_no_tqdm(tmp_path, monkeypatch):
    """``progress=False`` must not construct a tqdm bar."""
    from immersionlab.telemed import export_video

    _patch_encode_frames(monkeypatch)

    constructions: list = []

    class _Sentinel:
        def __init__(self, **kw):
            constructions.append(kw)

        def update(self, n): pass

        def close(self): pass

    import tqdm.auto as _tqdm_auto
    monkeypatch.setattr(_tqdm_auto, "tqdm", _Sentinel, raising=False)

    h5 = _make_synthetic_h5(tmp_path / "rec.tvd.h5", n_frames=3)
    export_video(h5, progress=False)
    assert constructions == []


def test_export_video_progress_callback(tmp_path, monkeypatch):
    from immersionlab.telemed import export_video

    _patch_encode_frames(monkeypatch)
    _make_synthetic_h5(tmp_path / "a.tvd.h5", n_frames=2)
    calls = []

    def cb(idx, total, path, status):
        calls.append((idx, total, path.name, status))

    export_video(tmp_path, progress=False, progress_callback=cb)
    assert calls == [(0, 1, "a.mp4", "built")]


# ---------- Log.to_video convenience ----------


def test_log_to_video_delegates_to_export_video(tmp_path, monkeypatch):
    """``Log.to_video()`` should call ``export_video`` with the sidecar
    path and forward kwargs; default writes ``<stem>.mp4`` next to
    the sidecar."""
    from immersionlab.telemed import Log

    cap = _patch_encode_frames(monkeypatch)
    h5 = _make_synthetic_h5(tmp_path / "rec.tvd.h5", n_frames=3)
    results = Log(h5).to_video()
    assert str(tmp_path / "rec.mp4") in results
    assert results[str(tmp_path / "rec.mp4")] == "built"
    assert len(cap) == 1


def test_log_to_video_forwards_kwargs(tmp_path, monkeypatch):
    from immersionlab.telemed import Log

    cap = _patch_encode_frames(monkeypatch)
    h5 = _make_synthetic_h5(tmp_path / "rec.tvd.h5", n_frames=3)
    Log(h5).to_video(lossless=False, crf=18)
    cmd = cap[0]["cmd"]
    assert "-crf" in cmd and cmd[cmd.index("-crf") + 1] == "18"


# ---------- Dispatcher ----------


class TestPipelineEntryPoints:
    """The three named entry points: ``export_h5``, ``export_video``,
    ``process``. ``kind=`` dispatcher was retired 2026-05-24 in favour
    of explicit names."""

    def test_export_h5_no_match_returns_empty(self, tmp_path):
        """``export_h5`` short-circuits on an empty folder (no EchoWave
        connection attempted -- runs under any Python)."""
        from immersionlab import telemed

        empty = tmp_path / "empty"
        empty.mkdir()
        assert telemed.export_h5(empty) == {}

    def test_export_video_no_match_returns_empty(self, tmp_path):
        from immersionlab import telemed

        empty = tmp_path / "empty"
        empty.mkdir()
        assert telemed.export_video(empty) == {}

    def test_process_chains_h5_and_video(self, tmp_path, monkeypatch):
        """``process()`` calls both stages and returns
        ``{"h5": {...}, "video": {...}}``. Use empty folder to
        sidestep COM."""
        from immersionlab import telemed

        empty = tmp_path / "empty"
        empty.mkdir()
        out = telemed.process(empty)
        assert set(out) == {"h5", "video"}
        assert out["h5"] == {}
        assert out["video"] == {}

    def test_process_routes_video_only_kwarg(self, tmp_path, monkeypatch):
        """Video-only kwargs (``lossless``, ``crf``, ...) must flow to
        the video stage only -- not raise as 'unexpected kwarg' against
        export_h5."""
        from immersionlab import telemed

        cap = _patch_encode_frames(monkeypatch)
        h5 = _make_synthetic_h5(tmp_path / "rec.tvd.h5", n_frames=3)
        # h5 stage finds no .tvd (pattern default), so it short-circuits;
        # video stage picks up the existing .tvd.h5 fixture and encodes.
        out = telemed.process(tmp_path, lossless=False, crf=22)
        assert out["h5"] == {}
        # video stage built one mp4 from the synthetic sidecar.
        assert any(v == "built" for v in out["video"].values())
        assert "-crf" in cap[0]["cmd"]
        assert cap[0]["cmd"][cap[0]["cmd"].index("-crf") + 1] == "22"

    def test_process_unknown_kwarg_raises(self, tmp_path):
        """Bogus kwarg -> TypeError naming the accepted h5/video sets."""
        from immersionlab import telemed

        with pytest.raises(TypeError, match="unknown kwargs"):
            telemed.process(tmp_path, banana=True)

    def test_kind_dispatcher_removed(self):
        """Regression guard: ``telemed.export`` and ``kind=`` are gone.
        Anyone still calling ``telemed.export(source)`` should hit a
        clean AttributeError -- not silent passthrough to the wrong
        function."""
        from immersionlab import telemed

        assert not hasattr(telemed, "export"), (
            "telemed.export was retired 2026-05-24; use export_h5 / "
            "export_video / process"
        )


# ---------- Real-ffmpeg roundtrip (skipped if absent) ----------


@pytest.mark.skipif(
    not shutil.which("ffmpeg"),
    reason="ffmpeg not on PATH; skipping real-pipe integration test",
)
def test_export_video_real_ffmpeg_roundtrip(tmp_path):
    """Pipe synthetic frames through real ffmpeg, verify the mp4 lands
    + has the right frame count. Catches Popen / stdin-bytes plumbing
    regressions that the monkeypatched tests can't see."""
    import cv2 as cv
    from immersionlab.telemed import export_video

    # Use the full-frame ROI so cropping doesn't shrink to ~odd dims
    # that libx265 dislikes; ultrafast preset keeps the test snappy.
    h5 = _make_synthetic_h5(
        tmp_path / "real.tvd.h5", n_frames=5, h=64, w=96,
        rois={1: dict(x1=1, x2=96, y1=1, y2=64, width=96, height=64,
                      dx=0.01, dy=0.01)},
    )
    results = export_video(h5, preset="ultrafast")
    out_mp4 = tmp_path / "real.mp4"
    assert results[str(out_mp4)] == "built"
    assert out_mp4.is_file() and out_mp4.stat().st_size > 0

    cap = cv.VideoCapture(str(out_mp4))
    try:
        n = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    assert n == 5
