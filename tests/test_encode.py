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
    frames_data=None,
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
            if frames_data is not None:
                arr = np.asarray(frames_data, dtype=np.uint8)
                assert arr.shape == (n_frames, h, w), (
                    f"frames_data shape {arr.shape} != ({n_frames}, {h}, {w})"
                )
            else:
                arr = np.zeros((n_frames, h, w), dtype=np.uint8)
                for i in range(n_frames):
                    arr[i] = (np.linspace(0, 255, w, dtype=np.float32)[None, :]
                              .repeat(h, axis=0) * (1.0 - i / max(n_frames - 1, 1))).astype(np.uint8)
            h5.create_group("frames").create_dataset(
                "gray", data=arr, chunks=(1, h, w),
            )
    return path


def _telemed_shaped_frames(n_frames=5, full_h=64, full_w=96,
                           roi=(10, 50, 5, 45), margin_w=5,
                           gray=56, seed=0):
    """Build frames with Telemed-shaped UI margins (gray side bands +
    saturated bottom-tick row) inside the panel ROI. The detector
    should find an inner box strictly smaller than the panel on
    both axes.
    """
    rng = np.random.default_rng(seed)
    x1, x2, y1, y2 = roi
    arr = np.zeros((n_frames, full_h, full_w), dtype=np.uint8)
    for i in range(n_frames):
        arr[i, y1-1:y2, x1-1:x2] = gray
        ix1 = x1 - 1 + margin_w
        ix2 = x2 - margin_w
        iy1 = y1 - 1
        iy2 = y2 - 1
        arr[i, iy1:iy2, ix1:ix2] = rng.integers(
            8, 41, size=(iy2 - iy1, ix2 - ix1), dtype=np.uint8,
        )
        arr[i, y2 - 1, x1 - 1:x2] = 255
    return arr


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
        """``process()`` triages + dispatches; on an empty folder
        every per-stage result dict is empty.

        The return shape is ``{"h5", "video", "toc"}`` since 2026-05-24
        (TOC sidecars built inline by the dispatcher)."""
        from immersionlab import telemed

        empty = tmp_path / "empty"
        empty.mkdir()
        out = telemed.process(empty)
        assert set(out) == {"h5", "video", "toc"}
        assert out["h5"] == {}
        assert out["video"] == {}
        assert out["toc"] == {}

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


# ---------- Inner-image autocrop detector (encode-time) ----------


def _synthetic_panel(W=200, H=100, margin_w=30, gray=56,
                     inner_low=8, inner_high=40, tick_row=True,
                     seed=0):
    """Build a panel mimicking the Telemed UI layout for unit-testing
    ``_detect_image_roi``. Layout: uniform UI gray on the side
    margins, randomised inner-image brightness in the middle, and an
    optional saturated tick row at the bottom.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    panel = np.full((H, W), gray, dtype=np.uint8)
    inner = rng.integers(
        inner_low, inner_high + 1,
        size=(H, W - 2 * margin_w), dtype=np.uint8,
    )
    panel[:, margin_w:W - margin_w] = inner
    if tick_row:
        panel[-1, :] = 255
    return panel


class TestDetectImageRoi:
    """Unit tests for the encode-time inner-image detector."""

    def test_strips_margins_and_tick_row(self):
        from immersionlab.telemed._encode import _detect_image_roi

        panel = _synthetic_panel(W=400, H=120, margin_w=80, tick_row=True)
        inner = _detect_image_roi(panel)
        assert inner is not None
        x_s, x_e, y_s, y_e = inner
        assert abs(x_s - 80) <= 2
        assert abs(x_e - 320) <= 2
        assert y_s == 0
        assert y_e <= 119  # tick row excluded
        assert y_e >= 110

    def test_thin_margins(self):
        """Dual-probe-shape panel (margins ~5 px each side) -- this
        was the case that broke an earlier Otsu-on-col_std prototype.
        """
        from immersionlab.telemed._encode import _detect_image_roi

        panel = _synthetic_panel(W=200, H=100, margin_w=5, tick_row=True)
        inner = _detect_image_roi(panel)
        assert inner is not None
        x_s, x_e, y_s, y_e = inner
        assert abs(x_s - 5) <= 2
        assert abs(x_e - 195) <= 2
        assert y_s == 0
        assert y_e <= 99

    def test_no_tick_row_keeps_full_height(self):
        from immersionlab.telemed._encode import _detect_image_roi

        panel = _synthetic_panel(W=200, H=100, margin_w=20, tick_row=False)
        inner = _detect_image_roi(panel)
        assert inner is not None
        _, _, y_s, y_e = inner
        assert y_s == 0
        assert y_e == 100

    def test_uniform_input_returns_none(self):
        """Fully-black panel -> no detectable box -> None; the encoder
        falls back to the panel ROI."""
        import numpy as np

        from immersionlab.telemed._encode import _detect_image_roi

        panel = np.zeros((100, 200), dtype=np.uint8)
        assert _detect_image_roi(panel) is None


class TestExportVideoAutocrop:
    """``crop="image"`` (the default) runs the detector against
    sampled frames and crops to the inner box; falls back to the
    panel ROI with a warning when the detector returns None."""

    def test_detector_picks_inner_box_on_shaped_frames(self, tmp_path, monkeypatch):
        """Telemed-shaped synthetic frames (gray margins + tick row
        inside the panel ROI) -> detector finds an inner box smaller
        than the panel; the ffmpeg cmd's -s WxH matches the inner
        dims."""
        from immersionlab.telemed import export_video

        cap = _patch_encode_frames(monkeypatch)
        rois = {1: dict(x1=10, x2=50, y1=5, y2=45, width=41, height=41,
                        dx=0.01, dy=0.01)}
        frames = _telemed_shaped_frames(
            n_frames=5, full_h=64, full_w=96, roi=(10, 50, 5, 45), margin_w=5,
        )
        h5 = _make_synthetic_h5(
            tmp_path / "rec.tvd.h5", n_frames=5, rois=rois,
            frames_data=frames,
        )
        export_video(h5)
        out_h, out_w = cap[0]["shape"]
        # Strict inequality: inner is smaller than the 41x41 panel on
        # both axes.
        assert out_h < 41
        assert out_w < 41
        cmd = cap[0]["cmd"]
        assert "-s" in cmd
        assert cmd[cmd.index("-s") + 1] == f"{out_w}x{out_h}"

    def test_panel_used_when_detector_returns_none(self, tmp_path, monkeypatch):
        """Flat synthetic frames (no margin step) -> detector returns
        None -> fallback to panel + UserWarning so the regression is
        visible."""
        import pytest as _pt
        from immersionlab.telemed import export_video

        cap = _patch_encode_frames(monkeypatch)
        rois = {1: dict(x1=10, x2=50, y1=5, y2=45, width=41, height=41,
                        dx=0.01, dy=0.01)}
        h5 = _make_synthetic_h5(tmp_path / "rec.tvd.h5", n_frames=3, rois=rois)
        with _pt.warns(UserWarning, match="couldn't identify inner ultrasound image"):
            export_video(h5)
        assert cap[0]["shape"] == (41, 41)

    def test_crop_panel_skips_detector_entirely(self, tmp_path, monkeypatch):
        """``crop="panel"`` doesn't run the detector + doesn't warn,
        even on Telemed-shaped frames where the detector would
        succeed."""
        import warnings

        from immersionlab.telemed import export_video

        cap = _patch_encode_frames(monkeypatch)
        rois = {1: dict(x1=10, x2=50, y1=5, y2=45, width=41, height=41,
                        dx=0.01, dy=0.01)}
        frames = _telemed_shaped_frames(
            n_frames=5, full_h=64, full_w=96, roi=(10, 50, 5, 45), margin_w=5,
        )
        h5 = _make_synthetic_h5(
            tmp_path / "rec.tvd.h5", n_frames=5, rois=rois,
            frames_data=frames,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test
            export_video(h5, crop="panel")
        assert cap[0]["shape"] == (41, 41)

    def test_invalid_crop_kwarg_surfaced_as_error_status(self, tmp_path, monkeypatch):
        """``crop="bogus"`` -> ValueError surfaced as a per-target
        error status (the loop catches per-target failures)."""
        from immersionlab.telemed import export_video

        _patch_encode_frames(monkeypatch)
        h5 = _make_synthetic_h5(tmp_path / "rec.tvd.h5", n_frames=3)
        results = export_video(h5, crop="bogus")
        status = next(iter(results.values()))
        assert status.startswith("error:")
        assert "crop=" in status


# ---------- TOC sidecar build ----------


class TestBuildToc:
    """``build_toc=True`` (default) builds a dnav .dnav-toc sidecar
    after each successful encode (and rebuilds a missing one on the
    skip-existing path). All tests monkeypatch the dnav probe so they
    don't need a real mp4 to demux."""

    def _patch_toc(self, monkeypatch):
        """Replace ``_ensure_toc_sidecar`` with a tracker that records
        which paths were probed and returns ``"built"`` for each."""
        from immersionlab.telemed import _encode

        seen: list = []

        def _fake(mp4_path):
            seen.append(mp4_path)
            return "built"

        monkeypatch.setattr(_encode, "_ensure_toc_sidecar", _fake)
        return seen

    def test_default_builds_toc_after_successful_encode(self, tmp_path, monkeypatch):
        from immersionlab.telemed import export_video

        _patch_encode_frames(monkeypatch)
        seen = self._patch_toc(monkeypatch)
        h5 = _make_synthetic_h5(tmp_path / "scan.tvd.h5", n_frames=3)
        results = export_video(h5)
        mp4 = tmp_path / "scan.mp4"
        assert results[str(mp4)] == "built"
        assert seen == [mp4]

    def test_build_toc_false_skips(self, tmp_path, monkeypatch):
        from immersionlab.telemed import export_video

        _patch_encode_frames(monkeypatch)
        seen = self._patch_toc(monkeypatch)
        h5 = _make_synthetic_h5(tmp_path / "scan.tvd.h5", n_frames=3)
        export_video(h5, build_toc=False)
        assert seen == []

    def test_skip_existing_rebuilds_missing_toc(self, tmp_path, monkeypatch):
        """If the mp4 already exists but its sidecar doesn't, the
        skip-existing path still ensures the TOC."""
        from immersionlab.telemed import export_video

        _patch_encode_frames(monkeypatch)
        seen = self._patch_toc(monkeypatch)
        h5 = _make_synthetic_h5(tmp_path / "scan.tvd.h5", n_frames=3)
        mp4 = tmp_path / "scan.mp4"
        mp4.write_bytes(b"")
        results = export_video(h5)
        assert results[str(mp4)] == "hit"
        assert seen == [mp4]

    def test_skip_existing_with_build_toc_false_does_nothing(self, tmp_path, monkeypatch):
        from immersionlab.telemed import export_video

        _patch_encode_frames(monkeypatch)
        seen = self._patch_toc(monkeypatch)
        h5 = _make_synthetic_h5(tmp_path / "scan.tvd.h5", n_frames=3)
        (tmp_path / "scan.mp4").write_bytes(b"")
        export_video(h5, build_toc=False)
        assert seen == []

    def test_encode_failure_skips_toc(self, tmp_path, monkeypatch):
        """If the encode raises, TOC is not attempted for the failed mp4."""
        from immersionlab.telemed import export_video

        seen = self._patch_toc(monkeypatch)
        h5 = _make_synthetic_h5(tmp_path / "scan.tvd.h5", n_frames=3)
        results = export_video(h5, crop="bogus")  # forces ValueError per panel
        status = next(iter(results.values()))
        assert status.startswith("error:")
        assert seen == []

    def test_dnav_missing_warns_once_returns_skipped(self, tmp_path, monkeypatch):
        """When dnav isn't importable, ``_ensure_toc_sidecar`` returns
        a ``"skipped: no dnav"`` status and emits exactly one warning
        across many calls (per-module flag)."""
        from immersionlab.telemed import _encode

        monkeypatch.setattr(_encode, "_HAS_DNAV", False)
        monkeypatch.setattr(_encode, "_DNAV_WARNED", False)
        with pytest.warns(UserWarning, match="datanavigator is not importable"):
            s1 = _encode._ensure_toc_sidecar(tmp_path / "a.mp4")
        # Second call: no new warning (flag latched).
        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            s2 = _encode._ensure_toc_sidecar(tmp_path / "b.mp4")
        assert s1 == s2 == "skipped: no dnav"
        assert not any("datanavigator" in str(w.message) for w in caught)

    def test_ensure_toc_returns_hit_when_sidecar_exists(self, tmp_path):
        """Real dnav round-trip: precompute against an existing mp4
        twice; first call returns built (or error if not a real mp4
        -- we just need the path to be a file), second returns hit."""
        from immersionlab.telemed import _encode

        if not _encode._HAS_DNAV:
            pytest.skip("datanavigator not importable in this env")
        # Real dnav needs a decodable mp4; use the real-ffmpeg roundtrip
        # fixture if ffmpeg is available, otherwise skip.
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not on PATH; need a real mp4 for dnav probe")
        from immersionlab.telemed import export_video

        h5 = _make_synthetic_h5(
            tmp_path / "real.tvd.h5", n_frames=5, h=64, w=96,
            rois={1: dict(x1=1, x2=96, y1=1, y2=64, width=96, height=64,
                          dx=0.01, dy=0.01)},
        )
        export_video(h5, preset="ultrafast", build_toc=False)
        mp4 = tmp_path / "real.mp4"
        s1 = _encode._ensure_toc_sidecar(mp4)
        s2 = _encode._ensure_toc_sidecar(mp4)
        assert s1 in ("built", "built (uncached)")
        assert s2 == "hit"
