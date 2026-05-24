"""Telemed ultrasound video helpers (legacy EchoWave-mp4 workflow).

.. deprecated:: 0.1.0
   The ``crop_video`` / ``crop_folder`` helpers in this module operate
   on the legacy EchoWave-mp4-export workflow (side-by-side mp4 from
   the Telemed device's built-in export, cropped per-side at hardcoded
   coordinates). They are superseded by the ``.tvd`` direct-read
   pipeline (:func:`telemed.export_video` / :func:`telemed.process`)
   and will be removed in telemed v0.2.0.

Telemed MP4 recordings combine a left- and right-side ultrasound view
side-by-side in one frame. ``crop_video`` splits one such MP4 into the
two per-side clips using the lab's standard 706x558 crop windows at
x=777 (left) / x=72 (right), y=42. ``crop_folder`` walks a study data
tree and crops every Telemed MP4 found.

Crop geometry is hardcoded — it reflects the Telemed device's frame
layout, not a per-study parameter. If the Telemed acquisition settings
change (resolution, layout, overlay placement) the constants below
need re-derivation.

Encoder default since 2026-05-23 is **h265 4:0:0 monochrome** (``mono=True``):
the crop output drops chroma planes entirely (``-c:v libx265 -pix_fmt
gray -crf 24 -an``), which fixes the chroma-noise-into-DLC-inference
penalty the older yuv420p crops carried (the bench at
``S:/_corpus/telemed/_bench/`` showed the yuv420p path was costing
~0.7 px median / 1.9 px p95 DLC keypoint error vs lossless). Sub-pixel
median preserved at CRF 24 (0.47 px vs lossless); p95 1.29 px. Drops
audio along with chroma.

Pass ``mono=False`` to fall back to the pre-graduation libx264
yuv420p invocation (``-c:v libx264 -preset slow``, no explicit ``-crf``).
A past cv2/decord frame-extraction inconsistency on training-data
videos pointed tentatively at NVENC, so libx264 stayed the conservative
default during the graduation period; callers who want NVENC (or any
other ffmpeg encoder) pass it via the ``encoder`` kwarg, which routes
through the internal :func:`encoder_flags` helper. ``encoder`` is
ignored in the mono branch (libx264 cannot produce true 4:0:0).
"""
from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import pyfilemanager


def _run_ffmpeg(
    cmd: list[str],
    *,
    label: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run an ffmpeg/ffprobe command with stdout+stderr captured.

    Args:
        cmd: Argument list passed to ``subprocess.run`` (never a string —
            keeps shell-quoting safe).
        label: Short description used in the error message.
        check: When True, re-raises ``CalledProcessError`` on non-zero
            exit with the captured stderr appended to the message.

    Returns:
        The completed ``subprocess.CompletedProcess`` (stdout / stderr
        decoded as text).

    Raises:
        subprocess.CalledProcessError: When ``check`` is True and the
            command exited non-zero. The exception message includes the
            captured stderr (truncated to 4000 chars).
    """
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        tag = f"[{label}] " if label else ""
        stderr = (result.stderr or "").strip()
        if len(stderr) > 4000:
            stderr = stderr[:4000] + "... (truncated)"
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=(
                f"{tag}ffmpeg/ffprobe exited {result.returncode}.\n"
                f"cmd: {' '.join(str(c) for c in cmd)}\n"
                f"stderr:\n{stderr}"
            ),
        )
    return result


def encoder_flags(
    encoder: str,
    crf: int = 28,
    preset: str = "slow",
) -> list[str]:
    """Return the ``-c:v ...`` flag list for the named encoder.

    ``h264_nvenc`` uses VBR with constant-quality; ``libx264`` uses CRF.
    Other encoders pass through with a generic preset.
    """
    if encoder == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc",
            "-rc:v", "vbr",
            "-cq:v", str(crf),
            "-b:v", "0",
            "-preset", preset,
        ]
    if encoder == "libx264":
        return [
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", preset,
        ]
    return ["-c:v", encoder, "-preset", preset]


_DEPRECATION_MSG = (
    "telemed.crop_video / telemed.crop_folder is deprecated; use "
    "telemed.export_video or telemed.process() against .tvd recordings "
    "instead. This module will be removed in telemed v0.2.0."
)

CROP_W, CROP_H, CROP_Y = 706, 558, 42
X_LEFT, X_RIGHT = 777, 72

# Default CRF for the mono (libx265) branch. Picked by the 2026-05-23
# bench (S:/_corpus/telemed/_bench, interosseous_pn24-x model, 2010
# frames): vs a lossless mono crop reference, CRF 24 holds median DLC
# error at 0.47 px (p95 1.29) and produces files ~28% smaller than CRF
# 22 -- the dustrack.batch.convert_to_mono knob position, which was
# tuned for "standalone re-encode of an already-captured clip" rather
# than the one-pass crop+mono workflow this file owns. Sits at the
# conservative end of the bench-validated CRF window (22-26 all beat
# the legacy libx264 yuv420p workflow strictly on DLC accuracy); 24
# trades a small file-size cut for the highest sub-pixel quality among
# them. CRF 28 was tested but its p95 ~2 px starts to confound
# downstream LK refinement.
_MONO_DEFAULT_CRF = 24


def _build_crop_cmd(src, dst, side, *, encoder, crf, preset, mono=True):
    """Build the ffmpeg argument list for one Telemed crop.

    Pulled out as a pure helper so the test suite can pin the
    byte-identical default-encoder invocation as a regression guard.

    When ``mono=True``, the encoder is forced to libx265 with
    ``-pix_fmt gray`` (true h265 4:0:0 monochrome) and audio is
    dropped via ``-an``. The ``encoder`` kwarg is ignored in this
    branch because libx264 cannot produce true 4:0:0; a non-None
    ``encoder`` together with ``mono=True`` raises rather than
    silently swallowing the choice.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    x = X_LEFT if side == "left" else X_RIGHT
    if mono:
        if encoder is not None and encoder != "libx265":
            raise ValueError(
                f"mono=True requires libx265 (true h265 4:0:0); "
                f"encoder={encoder!r} would not produce monochrome. "
                f"Drop the encoder kwarg or pass encoder='libx265'."
            )
        mono_crf = _MONO_DEFAULT_CRF if crf is None else crf
        return [
            "ffmpeg", "-i", str(src),
            "-vf", f"crop={CROP_W}:{CROP_H}:{x}:{CROP_Y}",
            "-c:v", "libx265",
            "-pix_fmt", "gray",
            "-crf", str(mono_crf),
            "-preset", preset,
            "-fps_mode", "passthrough",
            "-an",
            str(dst),
        ]
    if encoder is None:
        codec_flags = ["-c:v", "libx264", "-preset", preset]
    else:
        codec_flags = encoder_flags(
            encoder, crf=28 if crf is None else crf, preset=preset,
        )
    return [
        "ffmpeg", "-i", str(src),
        "-vf", f"crop={CROP_W}:{CROP_H}:{x}:{CROP_Y}",
        *codec_flags,
        "-c:a", "copy",
        str(dst),
    ]


def crop_video(src, dst, side, *, encoder=None, crf=None, preset="slow", mono=True):
    """Crop one Telemed MP4 to the left or right view; skip if ``dst`` exists.

    Args:
        src: Source MP4 path.
        dst: Output MP4 path. If it already exists, the call is a no-op.
        side: ``"left"`` (x=777) or ``"right"`` (x=72).
        encoder: ffmpeg video encoder name. Only honoured when
            ``mono=False`` -- in the mono branch the encoder is forced
            to libx265 (libx264 can't produce true 4:0:0). With
            ``mono=False, encoder=None`` you get the pre-graduation
            invocation: ``-c:v libx264 -preset {preset}`` with no
            ``-crf``. Pass an explicit encoder (e.g. ``"h264_nvenc"``)
            to route through the internal :func:`encoder_flags` helper.
        crf: Quality. When ``mono=True`` (default), defaults to 24
            (median DLC pixel error 0.47 px vs lossless on
            interosseous_pn24-x; see bench in
            ``S:/_corpus/telemed/_bench/``). Lower values (e.g. 22)
            buy tighter parity at larger file size; higher (e.g. 26 or
            28) save more space. When ``encoder`` is not ``None``
            (``mono=False`` branch), defaults to 28 (consistent with
            ``video.py``).
        preset: ffmpeg ``-preset`` value. Default ``"slow"``.
        mono: True (default since 2026-05-23) encodes as h265 4:0:0
            monochrome (``-c:v libx265 -pix_fmt gray``) so the crop
            output is chroma-noise-free in one pass instead of needing
            a follow-up :func:`dustrack.batch.convert_to_mono` step.
            Drops audio along with chroma. Pass ``mono=False`` to
            restore the pre-graduation libx264 yuv420p path.

    .. deprecated:: 0.1.0
       Use :func:`telemed.export_video` or :func:`telemed.process`
       against ``.tvd`` recordings instead. Will be removed in v0.2.0.
    """
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
    dst = Path(dst)
    if dst.exists():
        print(f"Skipping {dst.name}, already exists.")
        return
    cmd = _build_crop_cmd(
        src, dst, side, encoder=encoder, crf=crf, preset=preset, mono=mono,
    )
    print(f"Processing {src} -> {dst.name}")
    _run_ffmpeg(cmd, label=f"crop_telemed({Path(str(src)).name}:{side})")
    print(f"[OK] Done: {dst.name}")


def crop_folder(
    data_dir,
    dest_dir,
    *,
    left_suffix,
    right_suffix,
    stem_split=" ",
    encoder=None,
    crf=None,
    preset="slow",
    mono=True,
):
    """Crop every ``*telemed*.mp4`` under ``data_dir`` into ``dest_dir``.

    File discovery uses ``pyfilemanager.FileManager`` with
    ``include="telemed"`` and ``exclude="archive"`` — preserved from the
    pre-graduation pia02 / chi01 implementations so existing on-disk
    layouts continue to match.

    Output filenames are ``<stem_core><left_suffix>.mp4`` and
    ``<stem_core><right_suffix>.mp4`` where ``stem_core`` is the source
    stem split on ``stem_split`` (default: first whitespace-delimited
    token, which strips the device-emitted trailing description +
    timestamp).

    Default is ``mono=True`` (h265 4:0:0 monochrome). Pass
    ``mono=False`` to fall back to the pre-graduation libx264 yuv420p
    path; see :func:`crop_video` for the trade-offs.

    .. deprecated:: 0.1.0
       Use :func:`telemed.export_video` or :func:`telemed.process`
       against ``.tvd`` recordings instead. Will be removed in v0.2.0.
    """
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    fm = pyfilemanager.FileManager(str(data_dir)).add(
        "*.mp4", include="telemed", exclude="archive",
    )
    for fname in fm.all_files:
        stem_core = Path(fname).stem.split(stem_split)[0]
        crop_video(
            fname,
            dest_dir / f"{stem_core}{left_suffix}.mp4",
            "left",
            encoder=encoder, crf=crf, preset=preset, mono=mono,
        )
        crop_video(
            fname,
            dest_dir / f"{stem_core}{right_suffix}.mp4",
            "right",
            encoder=encoder, crf=crf, preset=preset, mono=mono,
        )
