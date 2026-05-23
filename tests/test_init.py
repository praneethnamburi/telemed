"""Unit tests for ``immersionlab.telemed``.

Patches ``_run_ffmpeg`` so the tests never actually shell ffmpeg; we
only verify the constructed cmd lists, filename construction, and the
skip-if-exists short-circuit.

The default-encoder cmd is pinned byte-for-byte: a past
cv2/decord frame-extraction inconsistency on training-data videos
pointed tentatively at NVENC, so the conservative default for
downstream DLC / annotation workflows is ``libx264`` with no explicit
``-crf`` (libx264 internal default = 23). Any future "modernization"
that changes those flags has to update this test alongside the
implementation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from immersionlab import telemed


class TestBuildCropCmd:
    """Pin the cmd-list builder so the byte stream can't silently regress."""

    def test_default_left_byte_identical(self):
        cmd = telemed._build_crop_cmd(
            "in.mp4", "out_L.mp4", "left",
            encoder=None, crf=None, preset="slow",
        )
        assert cmd == [
            "ffmpeg", "-i", "in.mp4",
            "-vf", "crop=706:558:777:42",
            "-c:v", "libx264", "-preset", "slow",
            "-c:a", "copy",
            "out_L.mp4",
        ]

    def test_default_right_byte_identical(self):
        cmd = telemed._build_crop_cmd(
            "in.mp4", "out_R.mp4", "right",
            encoder=None, crf=None, preset="slow",
        )
        assert cmd == [
            "ffmpeg", "-i", "in.mp4",
            "-vf", "crop=706:558:72:42",
            "-c:v", "libx264", "-preset", "slow",
            "-c:a", "copy",
            "out_R.mp4",
        ]

    def test_explicit_libx264_adds_crf(self):
        cmd = telemed._build_crop_cmd(
            "in.mp4", "out.mp4", "left",
            encoder="libx264", crf=None, preset="slow",
        )
        assert "-crf" in cmd and "28" in cmd
        assert cmd[cmd.index("-c:v") + 1] == "libx264"

    def test_explicit_nvenc(self):
        cmd = telemed._build_crop_cmd(
            "in.mp4", "out.mp4", "left",
            encoder="h264_nvenc", crf=None, preset="slow",
        )
        assert cmd[cmd.index("-c:v") + 1] == "h264_nvenc"
        assert "-rc:v" in cmd and "vbr" in cmd
        assert "-cq:v" in cmd and "28" in cmd

    def test_explicit_encoder_honors_crf(self):
        cmd = telemed._build_crop_cmd(
            "in.mp4", "out.mp4", "left",
            encoder="libx264", crf=20, preset="fast",
        )
        assert cmd[cmd.index("-crf") + 1] == "20"
        assert cmd[cmd.index("-preset") + 1] == "fast"

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
            telemed._build_crop_cmd(
                "in.mp4", "out.mp4", "middle",
                encoder=None, crf=None, preset="slow",
            )


class TestBuildCropCmdMono:
    """Pin the mono (h265 4:0:0) cmd shape so a one-pass crop+mono is
    byte-identical to what dustrack.batch.convert_to_mono would produce
    as a follow-up step at the default CRF=22 knob position."""

    def test_mono_default_left(self):
        cmd = telemed._build_crop_cmd(
            "in.mp4", "out_L.mp4", "left",
            encoder=None, crf=None, preset="slow", mono=True,
        )
        assert cmd == [
            "ffmpeg", "-i", "in.mp4",
            "-vf", "crop=706:558:777:42",
            "-c:v", "libx265",
            "-pix_fmt", "gray",
            "-crf", "22",
            "-preset", "slow",
            "-fps_mode", "passthrough",
            "-an",
            "out_L.mp4",
        ]

    def test_mono_default_right(self):
        cmd = telemed._build_crop_cmd(
            "in.mp4", "out_R.mp4", "right",
            encoder=None, crf=None, preset="slow", mono=True,
        )
        assert "crop=706:558:72:42" in cmd
        assert cmd[cmd.index("-c:v") + 1] == "libx265"
        assert cmd[cmd.index("-pix_fmt") + 1] == "gray"

    def test_mono_honors_explicit_crf(self):
        cmd = telemed._build_crop_cmd(
            "in.mp4", "out.mp4", "left",
            encoder=None, crf=18, preset="medium", mono=True,
        )
        assert cmd[cmd.index("-crf") + 1] == "18"
        assert cmd[cmd.index("-preset") + 1] == "medium"

    def test_mono_drops_audio(self):
        """Mono branch passes ``-an`` and omits ``-c:a copy`` -- ultrasound
        clips don't carry meaningful audio, and PyTables-of-mono pipelines
        downstream don't want a chroma-stripped video to surprise them
        with an audio stream."""
        cmd = telemed._build_crop_cmd(
            "in.mp4", "out.mp4", "left",
            encoder=None, crf=None, preset="slow", mono=True,
        )
        assert "-an" in cmd
        assert "-c:a" not in cmd

    def test_mono_with_libx265_encoder_kwarg_ok(self):
        """Redundant but explicit -- libx265 matches what mono forces, so
        the call still succeeds."""
        cmd = telemed._build_crop_cmd(
            "in.mp4", "out.mp4", "left",
            encoder="libx265", crf=None, preset="slow", mono=True,
        )
        assert cmd[cmd.index("-c:v") + 1] == "libx265"

    def test_mono_with_incompatible_encoder_raises(self):
        """libx264 can't produce true 4:0:0; refuse rather than silently
        emit a yuvj420p-with-constant-chroma fallback."""
        with pytest.raises(ValueError, match="mono=True requires libx265"):
            telemed._build_crop_cmd(
                "in.mp4", "out.mp4", "left",
                encoder="libx264", crf=None, preset="slow", mono=True,
            )

    def test_mono_with_nvenc_raises(self):
        with pytest.raises(ValueError, match="mono=True requires libx265"):
            telemed._build_crop_cmd(
                "in.mp4", "out.mp4", "left",
                encoder="h264_nvenc", crf=None, preset="slow", mono=True,
            )


class TestCropVideo:
    def test_skips_if_dst_exists(self, tmp_path, monkeypatch):
        captured = []
        monkeypatch.setattr(
            telemed, "_run_ffmpeg",
            lambda cmd, **kw: captured.append(cmd),
        )
        dst = tmp_path / "out.mp4"
        dst.write_bytes(b"")  # exists
        telemed.crop_video("in.mp4", dst, "left")
        assert captured == [], "should not invoke ffmpeg when dst exists"

    def test_invokes_ffmpeg_when_dst_missing(self, tmp_path, monkeypatch):
        captured = []
        monkeypatch.setattr(
            telemed, "_run_ffmpeg",
            lambda cmd, **kw: captured.append(cmd),
        )
        dst = tmp_path / "out.mp4"
        telemed.crop_video("in.mp4", dst, "right")
        assert len(captured) == 1
        cmd = captured[0]
        assert cmd[0] == "ffmpeg"
        assert "crop=706:558:72:42" in cmd
        assert str(dst) in cmd

    def test_invalid_side_raises(self, tmp_path):
        with pytest.raises(ValueError):
            telemed.crop_video("in.mp4", tmp_path / "out.mp4", "middle")


class TestCropFolder:
    def test_filename_construction(self, tmp_path, monkeypatch):
        """Output names = ``<first_token_of_stem><suffix>.mp4``."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        dest_dir = tmp_path / "dest"

        # Two synthetic telemed mp4s with the real-world stem shape:
        # "pia02_sNNN_TTT <description> <YYYYMMDD HHMMSS>.mp4".
        (data_dir / "pia02_s009_003 fav piece 20250512 093247.mp4").write_bytes(b"")
        (data_dir / "pia02_s014_001 emgmax warmup 20250519 141257.mp4").write_bytes(b"")

        captured = []

        def fake_crop_video(src, dst, side, **kw):
            captured.append((Path(src).name, Path(dst).name, side))

        monkeypatch.setattr(telemed, "crop_video", fake_crop_video)
        # Skip the FileManager dance — patch it to a shim that returns
        # exactly the files we created above.
        monkeypatch.setattr(
            telemed, "pyfilemanager",
            _FakeFileManagerModule([
                str(data_dir / "pia02_s009_003 fav piece 20250512 093247.mp4"),
                str(data_dir / "pia02_s014_001 emgmax warmup 20250519 141257.mp4"),
            ]),
        )

        telemed.crop_folder(
            data_dir, dest_dir,
            left_suffix="_LFA2", right_suffix="_RFA2",
        )

        # Two files × two sides = four calls
        assert len(captured) == 4
        names_out = {c[1] for c in captured}
        assert names_out == {
            "pia02_s009_003_LFA2.mp4",
            "pia02_s009_003_RFA2.mp4",
            "pia02_s014_001_LFA2.mp4",
            "pia02_s014_001_RFA2.mp4",
        }
        sides = [c[2] for c in captured]
        assert sides == ["left", "right", "left", "right"]
        assert dest_dir.exists()

    def test_mono_propagates_to_crop_video(self, tmp_path, monkeypatch):
        """``crop_folder(mono=True)`` must pass ``mono=True`` through to
        each per-side ``crop_video`` call."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        dest_dir = tmp_path / "dest"
        (data_dir / "pia02_s009_003 fav 20250512 093247.mp4").write_bytes(b"")

        captured = []

        def fake_crop_video(src, dst, side, **kw):
            captured.append(kw.get("mono"))

        monkeypatch.setattr(telemed, "crop_video", fake_crop_video)
        monkeypatch.setattr(
            telemed, "pyfilemanager",
            _FakeFileManagerModule([
                str(data_dir / "pia02_s009_003 fav 20250512 093247.mp4"),
            ]),
        )

        telemed.crop_folder(
            data_dir, dest_dir,
            left_suffix="_L", right_suffix="_R", mono=True,
        )

        assert captured == [True, True]

    def test_custom_stem_split(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        dest_dir = tmp_path / "dest"
        (data_dir / "subj_001-trial_A.mp4").write_bytes(b"")

        captured = []

        def fake_crop_video(src, dst, side, **kw):
            captured.append(Path(dst).name)

        monkeypatch.setattr(telemed, "crop_video", fake_crop_video)
        monkeypatch.setattr(
            telemed, "pyfilemanager",
            _FakeFileManagerModule([str(data_dir / "subj_001-trial_A.mp4")]),
        )

        telemed.crop_folder(
            data_dir, dest_dir,
            left_suffix="_LARM", right_suffix="_RLEG",
            stem_split="-",
        )

        assert set(captured) == {"subj_001_LARM.mp4", "subj_001_RLEG.mp4"}


class _FakeFileManager:
    def __init__(self, files):
        self._files = list(files)

    def add(self, *args, **kwargs):
        return self

    @property
    def all_files(self):
        return list(self._files)


class _FakeFileManagerModule:
    """Stand-in for the ``pyfilemanager`` module so tests don't need the
    real one to walk the filesystem with its ``include`` / ``exclude``
    semantics — the filename-construction logic in ``crop_folder`` is
    what we're pinning here, not pyfilemanager's matching.
    """
    def __init__(self, files):
        self._files = files

    def FileManager(self, *args, **kwargs):  # noqa: N802 - mirror pyfilemanager API
        return _FakeFileManager(self._files)
