"""Tests for ``immersionlab.telemed._dispatch``.

The dispatcher is the entry point for ``telemed.process()`` -- it
triages sources into "needs extract" vs "has h5", runs the appropriate
pipeline(s), and (in Scenario 4) runs both concurrently.

The COM-bound extract and the libx265 encode are both stubbed; what's
under test is the orchestration shape (triage, dispatch, postprocess
fan-out, result-dict aggregation).
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")


# ---------- Helpers ----------


def _touch_tvd(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _make_h5(path: Path, *, n_frames: int = 3, h: int = 32, w: int = 48) -> Path:
    """Minimal v1a5-shape .tvd.h5 sidecar (enough for export_video to encode)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.cumsum([0.0] + [14.5 for _ in range(n_frames - 1)])
    ifi = np.zeros(n_frames)
    ifi[1:] = np.diff(times)
    with h5py.File(path, "w") as f5:
        f5.attrs["n_frames"] = n_frames
        f5.attrs["full_frame_width"] = w
        f5.attrs["full_frame_height"] = h
        f5.attrs["n_b_images"] = 1
        f5.attrs["roi1_x1"] = 1
        f5.attrs["roi1_x2"] = w
        f5.attrs["roi1_y1"] = 1
        f5.attrs["roi1_y2"] = h
        f5.attrs["roi1_width"] = w
        f5.attrs["roi1_height"] = h
        f5.attrs["physical_dx1_cm_per_px"] = 0.01
        f5.attrs["physical_dy1_cm_per_px"] = 0.01
        f5.attrs["source_tvd_path"] = "C:/synthetic/x.tvd"
        f5.attrs["extracted_at_iso"] = "2026-05-24T00:00:00"
        f5.attrs["schema_version"] = "v1a5"
        tg = f5.create_group("timing")
        tg.create_dataset("frame_idx_1n", data=np.arange(1, n_frames + 1, dtype=np.int32))
        tg.create_dataset("time_ms", data=times)
        tg.create_dataset("ifi_ms", data=ifi)
        arr = np.zeros((n_frames, h, w), dtype=np.uint8)
        f5.create_group("frames").create_dataset(
            "gray", data=arr, chunks=(1, h, w),
        )
    return path


def _patch_extract_and_connect(monkeypatch):
    """Stub COM-bound functions so dispatch tests don't need EchoWave."""
    from immersionlab.telemed import _extract

    def _fake_extract(tvd_path, out_path=None, **kwargs):
        out = Path(out_path) if out_path is not None else Path(str(tvd_path) + ".h5")
        # Build a real h5 so export_video downstream can actually open it.
        _make_h5(out, n_frames=3)
        return out

    monkeypatch.setattr(_extract, "_extract_one", _fake_extract)
    monkeypatch.setattr(_extract, "connect", lambda: None)


def _patch_encode(monkeypatch):
    """Stub the ffmpeg pipe so we don't shell out; mp4 file is touched."""
    from immersionlab.telemed import _encode

    def _fake(cmd, frames_iter):
        # The output path is always the last argv arg.
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\x00")  # placeholder mp4 bytes
        # Drain the generator so /frames/gray reads actually happen
        # (matches the real ffmpeg behaviour of pulling every frame).
        for _ in frames_iter:
            pass

    monkeypatch.setattr(_encode, "_encode_frames", _fake)


def _patch_toc(monkeypatch):
    """Replace ``_ensure_toc_sidecar`` with a tracker that returns built."""
    from immersionlab.telemed import _encode

    seen: list = []

    def _fake(mp4_path):
        seen.append(Path(mp4_path))
        return "built"

    monkeypatch.setattr(_encode, "_ensure_toc_sidecar", _fake)
    return seen


# ---------- Triage ----------


class TestTriage:
    def test_empty_folder(self, tmp_path):
        from immersionlab.telemed._dispatch import _triage

        a, b = _triage(tmp_path)
        assert a == [] and b == []

    def test_only_tvds_with_no_h5_all_set_a(self, tmp_path):
        from immersionlab.telemed._dispatch import _triage

        _touch_tvd(tmp_path / "a.tvd")
        _touch_tvd(tmp_path / "b.tvd")
        a, b = _triage(tmp_path)
        assert sorted(p.name for p in a) == ["a.tvd", "b.tvd"]
        assert b == []

    def test_tvd_with_sibling_h5_goes_to_set_b_only(self, tmp_path):
        from immersionlab.telemed._dispatch import _triage

        _touch_tvd(tmp_path / "a.tvd")
        _make_h5(tmp_path / "a.tvd.h5")
        a, b = _triage(tmp_path)
        assert a == []
        assert [p.name for p in b] == ["a.tvd.h5"]

    def test_mixed_split(self, tmp_path):
        from immersionlab.telemed._dispatch import _triage

        _touch_tvd(tmp_path / "needs_extract.tvd")
        _touch_tvd(tmp_path / "already_done.tvd")
        _make_h5(tmp_path / "already_done.tvd.h5")
        a, b = _triage(tmp_path)
        assert [p.name for p in a] == ["needs_extract.tvd"]
        assert [p.name for p in b] == ["already_done.tvd.h5"]

    def test_orphaned_h5_with_no_tvd_still_set_b(self, tmp_path):
        """User deleted the .tvd after extracting; the .tvd.h5 still
        needs encode+TOC."""
        from immersionlab.telemed._dispatch import _triage

        _make_h5(tmp_path / "orphan.tvd.h5")
        a, b = _triage(tmp_path)
        assert a == []
        assert [p.name for p in b] == ["orphan.tvd.h5"]


# ---------- Scenario 1: one .tvd, nothing processed ----------


class TestScenario1:
    """Single .tvd, no .h5; full pipeline runs end-to-end on one file."""

    def test_single_file_end_to_end(self, tmp_path, monkeypatch):
        from immersionlab import telemed

        _patch_extract_and_connect(monkeypatch)
        _patch_encode(monkeypatch)
        toc_seen = _patch_toc(monkeypatch)

        tvd = _touch_tvd(tmp_path / "scan.tvd")
        out = telemed.process(tmp_path, copy_to_local=False)

        assert out["h5"] == {str(tvd): "built"}
        # Single panel -> bare <stem>.mp4 next to the .tvd.
        mp4 = tmp_path / "scan.mp4"
        assert out["video"] == {str(mp4): "built"}
        assert out["toc"] == {str(mp4): "built"}
        assert toc_seen == [mp4]
        # And the sidecar h5 actually lands on disk (fake-extract writes it).
        assert (tmp_path / "scan.tvd.h5").exists()


# ---------- Scenario 2: many .tvd, none processed ----------


class TestScenario2:
    """Multiple .tvd, no .h5; Pipeline A runs with the postprocess hook
    encoding + TOCing + uploading each file's output."""

    def test_two_files_both_get_full_treatment(self, tmp_path, monkeypatch):
        from immersionlab import telemed

        _patch_extract_and_connect(monkeypatch)
        _patch_encode(monkeypatch)
        toc_seen = _patch_toc(monkeypatch)

        for name in ("a.tvd", "b.tvd"):
            _touch_tvd(tmp_path / name)
        out = telemed.process(tmp_path, copy_to_local=False)

        assert set(out["h5"]) == {
            str(tmp_path / "a.tvd"),
            str(tmp_path / "b.tvd"),
        }
        assert all(v == "built" for v in out["h5"].values())
        mp4s = {tmp_path / "a.mp4", tmp_path / "b.mp4"}
        assert set(Path(p) for p in out["video"]) == mp4s
        assert all(v == "built" for v in out["video"].values())
        assert set(Path(p) for p in out["toc"]) == mp4s
        assert set(toc_seen) == mp4s


# ---------- Scenario 3: many .tvd, all have .h5 ----------


class TestScenario3:
    """Multiple .tvd with .h5 already present; Pipeline B (encode-only)
    handles everything; extract is never touched."""

    def test_all_h5s_go_through_pipeline_b(self, tmp_path, monkeypatch):
        from immersionlab import telemed

        _patch_encode(monkeypatch)
        toc_seen = _patch_toc(monkeypatch)

        # Stub _extract_one too as a defensive check -- it should NEVER
        # be invoked in this scenario (no .tvd lacks its .h5).
        from immersionlab.telemed import _extract
        called: list = []
        monkeypatch.setattr(
            _extract, "_extract_one",
            lambda *a, **kw: called.append(a) or None,
        )
        monkeypatch.setattr(_extract, "connect", lambda: None)

        for name in ("a", "b"):
            _touch_tvd(tmp_path / f"{name}.tvd")
            _make_h5(tmp_path / f"{name}.tvd.h5")

        out = telemed.process(tmp_path)

        assert called == [], "extract should not run when every file has .h5"
        # h5 dict is empty -- export_h5 was never invoked (set_a empty).
        assert out["h5"] == {}
        mp4s = {tmp_path / "a.mp4", tmp_path / "b.mp4"}
        assert set(Path(p) for p in out["video"]) == mp4s
        assert all(v == "built" for v in out["video"].values())
        # Pipeline B reflects TOC status from the sidecar presence;
        # _ensure_toc_sidecar was called inside export_video.
        assert set(toc_seen) == mp4s


# ---------- Scenario 4: mixed; both pipelines run concurrently ----------


class TestScenario4:
    """Some .tvd need extract, others already have .h5. Both pipelines
    run concurrently on a 2-thread executor (different bottlenecks)."""

    def test_concurrent_dispatch_processes_both_sets(self, tmp_path, monkeypatch):
        from immersionlab import telemed

        _patch_extract_and_connect(monkeypatch)
        _patch_encode(monkeypatch)
        _patch_toc(monkeypatch)

        _touch_tvd(tmp_path / "needs.tvd")
        _touch_tvd(tmp_path / "has.tvd")
        _make_h5(tmp_path / "has.tvd.h5")

        out = telemed.process(tmp_path, copy_to_local=False)

        # Pipeline A handled the needs-extract file.
        assert out["h5"] == {str(tmp_path / "needs.tvd"): "built"}
        # Pipeline B handled the has-h5 file's encode; Pipeline A's
        # postprocess handled needs.tvd's mp4. Both land in video/toc.
        mp4s = {tmp_path / "needs.mp4", tmp_path / "has.mp4"}
        assert set(Path(p) for p in out["video"]) == mp4s
        assert all(v == "built" for v in out["video"].values())
        assert set(Path(p) for p in out["toc"]) == mp4s


# ---------- Postprocess closure -- staged-vs-local upload behaviour ----------


class TestPostprocessUpload:
    """``_make_postprocess`` differentiates the staged case (upload
    h5+mp4 from local temp to network) vs the local case (encode lands
    directly at destination)."""

    def test_local_source_no_upload_no_cleanup(self, tmp_path, monkeypatch):
        """When ``staged.stage_dir is None`` (source already local),
        the postprocess doesn't touch the local temp tree because
        there isn't one -- it just records final paths + builds TOC."""
        from immersionlab.telemed._dispatch import _make_postprocess
        from immersionlab.telemed._extract import _StagedFile

        _patch_encode(monkeypatch)
        toc_seen = _patch_toc(monkeypatch)

        src = _touch_tvd(tmp_path / "rec.tvd")
        h5 = _make_h5(tmp_path / "rec.tvd.h5")
        staged = _StagedFile(
            src_tvd=src, dst_h5=h5,
            local_tvd=src, local_h5=h5, stage_dir=None,
        )
        video_results: dict = {}
        toc_results: dict = {}
        pp = _make_postprocess(
            video_kwargs={},
            video_results=video_results,
            toc_results=toc_results,
        )
        pp(staged, True)

        mp4 = tmp_path / "rec.mp4"
        assert video_results == {str(mp4): "built"}
        assert toc_results == {str(mp4): "built"}
        assert toc_seen == [mp4]
        # No temp dir touched (none existed).
        assert h5.exists()

    def test_staged_source_uploads_and_cleans_up(self, tmp_path, monkeypatch):
        """When ``staged.stage_dir`` is set, the postprocess copies
        h5 + mp4 from local temp to the destination dir, then removes
        the temp tree. TOC is built against the destination mp4."""
        from immersionlab.telemed._dispatch import _make_postprocess
        from immersionlab.telemed._extract import _StagedFile

        _patch_encode(monkeypatch)
        toc_seen = _patch_toc(monkeypatch)

        dst_dir = tmp_path / "network"
        dst_dir.mkdir()
        src = _touch_tvd(dst_dir / "rec.tvd")
        dst_h5 = dst_dir / "rec.tvd.h5"

        stage_dir = tmp_path / "local_temp"
        stage_dir.mkdir()
        local_tvd = stage_dir / "rec.tvd"
        local_tvd.write_bytes(b"")
        local_h5 = _make_h5(stage_dir / "rec.tvd.h5")

        staged = _StagedFile(
            src_tvd=src, dst_h5=dst_h5,
            local_tvd=local_tvd, local_h5=local_h5, stage_dir=stage_dir,
        )
        video_results: dict = {}
        toc_results: dict = {}
        pp = _make_postprocess(
            video_kwargs={},
            video_results=video_results,
            toc_results=toc_results,
        )
        pp(staged, True)

        # Destination has h5 + mp4; local temp is gone.
        assert dst_h5.exists()
        dst_mp4 = dst_dir / "rec.mp4"
        assert dst_mp4.exists()
        assert not stage_dir.exists()
        assert video_results == {str(dst_mp4): "built"}
        assert toc_results == {str(dst_mp4): "built"}
        # TOC was built against the destination mp4, not the local one.
        assert toc_seen == [dst_mp4]

    def test_failure_path_cleans_up_without_upload(self, tmp_path, monkeypatch):
        """``success=False`` -> no encode, no upload, just cleanup."""
        from immersionlab.telemed._dispatch import _make_postprocess
        from immersionlab.telemed._extract import _StagedFile

        _patch_encode(monkeypatch)
        toc_seen = _patch_toc(monkeypatch)

        dst_dir = tmp_path / "network"
        dst_dir.mkdir()
        src = _touch_tvd(dst_dir / "rec.tvd")
        dst_h5 = dst_dir / "rec.tvd.h5"

        stage_dir = tmp_path / "local_temp"
        stage_dir.mkdir()
        local_tvd = stage_dir / "rec.tvd"
        local_tvd.write_bytes(b"")
        # Note: NO local_h5 because the extract failed.
        local_h5 = stage_dir / "rec.tvd.h5"

        staged = _StagedFile(
            src_tvd=src, dst_h5=dst_h5,
            local_tvd=local_tvd, local_h5=local_h5, stage_dir=stage_dir,
        )
        video_results: dict = {}
        toc_results: dict = {}
        pp = _make_postprocess(
            video_kwargs={},
            video_results=video_results,
            toc_results=toc_results,
        )
        pp(staged, False)

        assert not dst_h5.exists()
        assert not (dst_dir / "rec.mp4").exists()
        assert not stage_dir.exists()
        assert video_results == {}
        assert toc_results == {}
        assert toc_seen == []


# ---------- Misc ----------


def test_unknown_kwarg_raises(tmp_path):
    from immersionlab import telemed

    with pytest.raises(TypeError, match="unknown kwargs"):
        telemed.process(tmp_path, banana=True)


def test_process_returns_three_keys_always(tmp_path):
    """The return contract is ``{h5, video, toc}`` even on an empty source."""
    from immersionlab import telemed

    out = telemed.process(tmp_path)
    assert set(out) == {"h5", "video", "toc"}


def test_empty_source_prints_diagnostic(tmp_path, capsys):
    """Silent no-op on an empty folder is the #1 confusion source --
    process() must announce that triage saw nothing. Path-form
    agnostic: a posix-style Path repr also satisfies the contract."""
    from immersionlab import telemed

    telemed.process(tmp_path)
    out = capsys.readouterr().out
    assert "no .tvd or .tvd.h5 files found" in out
    # The path is rendered via repr() so the exact slash style is
    # platform-dependent; just check the leaf name is in there.
    assert tmp_path.name in out


def test_missing_source_path_raises_with_helpful_message(tmp_path):
    """A non-existent source raises early instead of returning an
    empty-result dict -- the most common cause is the Windows
    elevated-vs-unelevated mapped-drive gotcha, which is impossible
    to debug from a silent no-op."""
    from immersionlab import telemed

    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="source path"):
        telemed.process(missing)


def test_missing_entry_in_list_raises(tmp_path):
    """One bad entry in a list aborts triage -- safer than silently
    skipping (matches the "fail fast on misconfig" intent)."""
    from immersionlab import telemed

    good = tmp_path / "good"
    good.mkdir()
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        telemed.process([good, missing])


def test_non_empty_source_prints_triage_summary(tmp_path, monkeypatch, capsys):
    """When triage finds work, process() announces the per-set counts
    so the user can verify the dispatcher saw what they expected."""
    from immersionlab import telemed

    _patch_extract_and_connect(monkeypatch)
    _patch_encode(monkeypatch)
    _patch_toc(monkeypatch)

    _touch_tvd(tmp_path / "needs.tvd")
    _touch_tvd(tmp_path / "done.tvd")
    _make_h5(tmp_path / "done.tvd.h5")

    telemed.process(tmp_path, copy_to_local=False)
    out = capsys.readouterr().out
    assert "1 .tvd needing extract" in out
    assert "1 .tvd.h5 needing encode" in out


# ---------- Phase-level progress messages ----------


class TestPhaseLoggers:
    """Background-worker phase messages: connect / stage / encode /
    upload / cleanup / toc. All gated on ``progress=True``."""

    def test_connect_message_emitted(self, tmp_path, monkeypatch, capsys):
        from immersionlab import telemed

        _patch_extract_and_connect(monkeypatch)
        _patch_encode(monkeypatch)
        _patch_toc(monkeypatch)
        _touch_tvd(tmp_path / "rec.tvd")

        telemed.process(tmp_path, copy_to_local=False)
        out = capsys.readouterr().out
        assert "connecting to EchoWave" in out
        assert "connected to EchoWave" in out

    def test_encode_and_toc_phases_logged_staged_path(
        self, tmp_path, monkeypatch, capsys,
    ):
        """Force ``copy_to_local=True`` to exercise stage/upload/cleanup
        phases too. (When copy_to_local=False, the stage/upload/cleanup
        phases are no-ops.)"""
        from immersionlab import telemed

        _patch_extract_and_connect(monkeypatch)
        _patch_encode(monkeypatch)
        _patch_toc(monkeypatch)
        # Real shutil.copy2 needs the .tvd to actually exist as a file.
        src = _touch_tvd(tmp_path / "rec.tvd")
        src.write_bytes(b"fake tvd")

        telemed.process(tmp_path, copy_to_local=True)
        out = capsys.readouterr().out
        # Each phase emits its tag.
        for tag in ("[stage]", "[encode]", "[upload]", "[cleanup]", "[toc]"):
            assert tag in out, f"missing {tag} in output:\n{out}"

    def test_progress_false_suppresses_phase_messages(
        self, tmp_path, monkeypatch, capsys,
    ):
        from immersionlab import telemed

        _patch_extract_and_connect(monkeypatch)
        _patch_encode(monkeypatch)
        _patch_toc(monkeypatch)
        src = _touch_tvd(tmp_path / "rec.tvd")
        src.write_bytes(b"fake tvd")

        telemed.process(tmp_path, copy_to_local=True, progress=False)
        out = capsys.readouterr().out
        for tag in ("[stage]", "[encode]", "[upload]", "[cleanup]",
                    "[toc]", "connecting to EchoWave"):
            assert tag not in out, f"unexpected {tag!r} under progress=False:\n{out}"

    def test_log_helper_is_threadsafe_under_concurrent_calls(self):
        """``_log`` serialises bg-worker output via a shared lock --
        rapid concurrent calls must produce N intact lines, never
        interleaved character soup."""
        import io
        import sys
        import threading

        from immersionlab.telemed._extract import _log

        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            def _spam(tag):
                for i in range(50):
                    _log(f"message {i}", tag=tag, progress=True)

            ts = [threading.Thread(target=_spam, args=(t,)) for t in ("a", "b", "c")]
            for t in ts: t.start()
            for t in ts: t.join()
        finally:
            sys.stdout = old_stdout

        # 3 threads * 50 messages = 150 lines, all well-formed.
        lines = buf.getvalue().splitlines()
        assert len(lines) == 150
        for line in lines:
            assert line.startswith("["), f"malformed line: {line!r}"
            assert "] [" in line, f"missing tag: {line!r}"
