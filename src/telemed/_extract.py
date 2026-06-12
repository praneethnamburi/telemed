"""COM-backed reader for Telemed `.tvd` (Telemed Video Data) files.

Wraps the AutoInt1 automation interface that ships with EchoWave II
(see ``C:/Program Files/Telemed/Echo Wave II Application/EchoWave II/
Config/Plugins/AutoInt1Client.txt`` for the underlying API docs).

This is the **only** chroma-free + native-VFR-timing path for Telemed
device data: bypasses the lossy mp4 export entirely, gives true uint8
grayscale arrays and per-frame timestamps at the device's native ~100 ns
precision. (DICOM exports also carry the timing in FrameTimeVector, but
truncate above ~10k frames -- unusable for typical pia02 recordings.)

One-time setup (per machine):

1. **Register the COM ProgID** -- open an Administrator PowerShell::

    cd "C:\\Program Files\\Telemed\\Echo Wave II Application\\EchoWave II\\Config\\Plugins"
    .\\AutoInt1_regasm.bat

   You should see "Types registered successfully".

Per-session setup:

2. **Start Echo Wave II as administrator** (right-click -> "Run as
   administrator"). Get it to its normal main window.
3. **Run Python from an Administrator shell** -- the COM connection
   only binds when both processes share elevation.

Network-drive note: EchoWave's OpenFile fails on UNC / mapped network
paths in our setup. :func:`extract_recording_folder` handles this
transparently by copying each source file to a local temp directory,
processing locally, and writing results back to the source folder.

Example::

    import telemed

    # Single file -- writes a sibling .tvd.h5
    telemed.export("C:/data/some.tvd")

    # Timing-only (much faster; skip pixel extraction)
    telemed.export("C:/data/some.tvd", frames=False)

    # Batch a folder, even when on a network drive
    telemed.export("M:/data/pia02")

    # Mix folders and individual files
    telemed.export(["M:/data/pia02", "M:/data/pia03", "C:/scratch/x.tvd"])

Known win32com gotcha (wrapped inside this module): zero-argument COM
methods on the .NET CCW are exposed as **properties**, not callables.
Attribute access invokes the call. See
``feedback_win32com_dotnet_ccw_zero_arg_property`` in auto-memory.
"""

from __future__ import annotations

import concurrent.futures
import os
import shutil
import struct
import tempfile
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

# ---------- Shared logging helper ----------


# Single lock for every timestamped print across the package. Background
# workers (stage / unstage / postprocess / TOC) share this with the main
# thread so their output lines don't interleave.
_LOG_LOCK = threading.Lock()


def _log(msg: str, *, tag: str = "", progress: bool = True) -> None:
    """Thread-safe ``[HH:MM:SS] [tag] msg`` print, gated on ``progress``.

    ``tag`` is a short phase prefix (``"stage"`` / ``"upload"`` /
    ``"encode"`` / ``"toc"`` / ``"cleanup"``) so interleaved bg/main
    output is grep-able. Pass ``tag=""`` for a bare timestamped line.
    """
    if not progress:
        return
    ts = time.strftime("%H:%M:%S")
    prefix = f"[{tag}] " if tag else ""
    with _LOG_LOCK:
        print(f"[{ts}] {prefix}{msg}", flush=True)


def _size_human(n_bytes: float) -> str:
    """Format ``n_bytes`` as a human-friendly KB / MB / GB string."""
    n = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ---------- .tvd container header (recorded frame count) ----------
#
# A Telemed ``.tvd`` is a RIFF-like container, but where standard RIFF
# uses a 32-bit chunk size, the .tvd "UIFF" form uses a **64-bit** LE
# size on every chunk (the recordings routinely exceed 4 GB, so a
# 32-bit size couldn't address them). The per-stream ``strh`` header
# carries the number of frames the *device* recorded at a fixed payload
# offset. That count is independent of how many frames EchoWave manages
# to load into RAM -- which matters because EchoWave silently truncates
# the load to fit available memory, leaving a short ``.tvd.h5`` with no
# error. Comparing the extracted ``n_frames`` against this header count
# (see ``telemed.verify_complete``) catches that truncation.
#
# Reading the header is also safe against the "what if EchoWave rewrites
# the file" worry: the extract pipeline copies each ``.tvd`` to a local
# temp dir and opens the *copy*, so the source ``.tvd`` is never written
# and its header stays pristine.

_TVD_FORM_MAGIC = b"UIFF"
_TVD_CONTAINER_IDS = (b"UIFF", b"LIST")
# Offset of the recorded-frame-count uint32 within the strh payload.
# Verified constant across the pia02 cohort (2026-05-27). The stored
# value runs ~2 frames above EchoWave's GetFramesCount on complete
# recordings, so completeness checks compare with a small tolerance
# rather than for exact equality (see telemed.verify_complete).
_TVD_STRH_NFRAMES_OFFSET = 0x14


def _walk_uiff_chunks(buf: bytes):
    """Yield ``(chunk_id, payload_offset, payload_size)`` for the UIFF
    chunk tree in ``buf``, descending into container chunks.

    UIFF chunk layout: 4-byte ASCII id, 8-byte LE size, then payload.
    Container chunks (``UIFF`` / ``LIST``) prefix their payload with a
    4-byte form/list type before child chunks begin. Sizes are clamped
    to the buffer length so a header-only read (we never load the whole
    multi-GB file) walks cleanly.
    """

    def rec(start: int, end: int):
        i = start
        while i + 12 <= end:
            cid = buf[i : i + 4]
            if not all(32 <= b < 127 for b in cid):
                return
            size = int.from_bytes(buf[i + 4 : i + 12], "little")
            payload = i + 12
            yield cid, payload, size
            if cid in _TVD_CONTAINER_IDS:
                yield from rec(payload + 4, min(payload + size, end))
            i = payload + size + (size & 1)

    yield from rec(0, len(buf))


def read_tvd_n_frames(tvd_path: Union[str, Path], *, _header_bytes: int = 65536) -> Optional[int]:
    """Recorded frame count declared in a ``.tvd`` container header.

    Parses the RIFF-like "UIFF" header (64-bit chunk sizes; see the
    module-level comment) and returns the frame count from the first
    per-stream ``strh`` chunk. This is the count the device wrote,
    independent of EchoWave's memory-limited load -- compare it against
    the extracted ``n_frames`` to detect a truncated extraction
    (``telemed.verify_complete`` does this for you).

    Only the first ``_header_bytes`` of the file are read (the headers
    live in the first few KB), so this is cheap even on a 20 GB
    recording on a network drive.

    Returns:
        The declared frame count, or ``None`` if the file isn't a
        readable ``.tvd`` (wrong magic, no ``strh`` chunk in the header
        window, truncated header). The value runs ~2 frames above
        EchoWave's ``GetFramesCount`` on complete recordings, so treat
        it as a tolerance reference, not an exact one.
    """
    p = Path(tvd_path)
    try:
        with open(p, "rb") as f:
            head = f.read(_header_bytes)
    except OSError:
        return None
    if head[:4] != _TVD_FORM_MAGIC:
        return None
    for cid, payload, _size in _walk_uiff_chunks(head):
        if cid == b"strh":
            off = payload + _TVD_STRH_NFRAMES_OFFSET
            if off + 4 <= len(head):
                return int(struct.unpack_from("<I", head, off)[0])
            return None
    return None


# Frames extracted ~this far below the .tvd-declared count are treated
# as a memory-truncated load (vs the benign ~+2 header overcount).
_TVD_TRUNCATION_TOLERANCE = 16


# ---------- .tvd per-frame timing (COM-free, bit-exact) ----------
#
# Each per-stream frame chunk (``00bb`` = stream 0 / primary B panel, ``01bb`` = stream 1) carries a
# 64-byte header before its pixel payload; at **offset 24** is a uint64 little-endian **end tick** in
# 100 ns units. The frames tile a contiguous tick timeline (frame k's start == frame k-1's end + 1).
# EchoWave's per-frame ``GetCurrentFrameTime`` -- the ``time_ms`` the COM path writes into the
# ``.tvd.h5`` -- is exactly
#
#     time_ms[k] = (end_tick[k] - end_tick[ref]) / 10000.0
#
# so the whole timing array is recoverable from the container alone: no COM, no Admin EchoWave, no
# per-frame GoToFrame1n walk. Verified bit-for-bit (``==``, not tolerance) against the COM export on
# the pia02 cohort (2025 EchoWave) AND the 2026 channel-C collection -- the container format is stable
# across those EchoWave versions. Caveat: the .tvd stores every *declared* frame; the COM export trims
# 1-4 warm-up / edge frames (a runtime decision NOT recorded in the container), so this returns a
# *superset* of the COM array with identical inter-frame intervals. Exactly reproducing a specific
# past COM export's trim would need its stored frame count, which isn't in the file.
#
# The read is seek-based: only the 12-byte chunk headers + the 8-byte tick are touched (never the
# multi-GB pixel payloads), so a short clip parses in ~1 s and a 13 GB / 50k-frame recording in a
# couple of minutes off a network drive -- vs the COM walk's hours.

_TVD_FRAME_STREAM = b"00bb"          # stream 0 == primary B-mode panel (what GetCurrentFrameTime tracks)
_TVD_FRAME_TICK_OFFSET = 24          # uint64 LE end-tick within the per-frame chunk header
_TVD_TICKS_PER_MS = 10000.0          # 100 ns ticks -> ms


def _walk_frame_ticks(f, start: int, end: int, stream: bytes, ticks: list) -> None:
    """Append each ``stream`` frame chunk's end tick (uint64, 100 ns) by SEEKing the UIFF chunk tree.

    Reads only chunk headers + the tick field -- never the pixel payloads -- so it stays cheap on
    multi-GB recordings on a network drive. ``f.seek(i)`` at the end of each iteration keeps the
    position correct across child-container recursion and tick reads.
    """
    f.seek(start)
    i = start
    while i + 12 <= end:
        hdr = f.read(12)
        if len(hdr) < 12:
            return
        cid = hdr[:4]
        if not all(32 <= b < 127 for b in cid):
            return
        size = int.from_bytes(hdr[4:12], "little")
        payload = i + 12
        if cid == stream:
            f.seek(payload + _TVD_FRAME_TICK_OFFSET)
            ticks.append(int.from_bytes(f.read(8), "little"))
        if cid in _TVD_CONTAINER_IDS:
            _walk_frame_ticks(f, payload + 4, min(payload + size, end), stream, ticks)
        i = payload + size + (size & 1)
        f.seek(i)


def read_tvd_frame_ticks(tvd_path: Union[str, Path], *, stream: bytes = _TVD_FRAME_STREAM):
    """Per-frame **end ticks** (uint64, 100 ns units) for every *declared* frame, COM-free.

    Returns a ``list[int]`` (one per frame chunk of ``stream``; ``00bb`` = primary B panel), or
    ``None`` if the file isn't a readable ``.tvd`` / has no such frames. See the module comment for
    the timing model. Seek-based -- never reads the pixel payloads, so it is cheap even on a
    multi-GB recording on a network drive.
    """
    p = Path(tvd_path)
    ticks: list = []
    try:
        size = p.stat().st_size
        with open(p, "rb") as f:
            if f.read(4) != _TVD_FORM_MAGIC:
                return None
            _walk_frame_ticks(f, 0, size, stream, ticks)
    except OSError:
        return None
    return ticks or None


def read_tvd_time_ms(tvd_path: Union[str, Path], *, stream: bytes = _TVD_FRAME_STREAM, ref: int = 0):
    """Bit-exact per-frame ``time_ms`` for every *declared* frame, COM-free.

    ``time_ms[k] = (end_tick[k] - end_tick[ref]) / 10000`` -- reproduces EchoWave's COM
    ``GetCurrentFrameTime`` bit-for-bit, with no EchoWave / COM round-trip. Returns a float64 numpy
    array of length = the declared frame count (a *superset* of the trimmed ``.tvd.h5`` ``time_ms``;
    see the module comment), or ``None`` if the file has no readable frame timing.
    """
    import numpy as np

    ticks = read_tvd_frame_ticks(tvd_path, stream=stream)
    if not ticks:
        return None
    e = np.asarray(ticks, dtype=np.int64)
    return (e - e[ref]) / _TVD_TICKS_PER_MS


# ---------- LUT-inversion guard (EchoWave < 4.4.0) ----------
#
# B-mode ultrasound has a predominantly dark background (the anatomy is
# sparse bright echoes on near-black). EchoWave builds before 4.4.0
# return ``GetLoadedFrameGray`` with the grayscale LUT inverted
# (``255 - x``), so the background comes back bright -- the .tvd device
# data is fine, but the extracted pixels are wrong. The separation is
# huge (primary-panel median ~9 normal vs ~245 inverted on the pia02
# cohort), so a midpoint threshold detects it with enormous margin.

_LUT_INVERSION_MEDIAN_THRESHOLD = 127.0


def _samples_look_lut_inverted(frames, roi) -> bool:
    """True if the sampled frames look LUT-inverted (bright background).

    ``frames`` is an iterable of full-frame 2-D uint8 arrays; ``roi`` is
    the primary B-mode panel (a :class:`TelemedRoi`-shaped object with
    1-based ``x1`` / ``x2`` / ``y1`` / ``y2``). Pure helper (no COM /
    HDF5) so it's unit-testable on synthetic arrays.
    """
    import numpy as np

    meds = []
    for fr in frames:
        crop = fr[roi.y1 - 1 : roi.y2, roi.x1 - 1 : roi.x2]
        if crop.size:
            meds.append(float(np.median(crop)))
    if not meds:
        return False
    return float(np.median(meds)) > _LUT_INVERSION_MEDIAN_THRESHOLD


def _reader_looks_lut_inverted(reader, meta, *, n_samples: int = 3) -> bool:
    """Sample up to ``n_samples`` frames via ``reader`` and run the
    LUT-inversion test against the primary panel ROI.

    Returns ``False`` (can't tell) when the recording has no frames or
    no img_id=1 panel rather than guessing.
    """
    n = meta.n_frames
    if n <= 0 or 1 not in meta.b_mode_rois:
        return False
    roi = meta.b_mode_rois[1]
    idxs = sorted({0, n // 2, n - 1})[: max(1, n_samples)]
    frames = [reader.get_frame_gray(i) for i in idxs]
    return _samples_look_lut_inverted(frames, roi)


# ---------- Metadata structures ----------


@dataclass
class TelemedRoi:
    """B-mode region-of-interest within the full Telemed display frame.

    ``img_id`` follows the AutoInt1Client convention: 1=B, 2=B2, 3=B3,
    4=B4 (additional B-mode panels light up when a second / third / etc.
    probe is active or a multi-image scan mode is in use). ``physical_dx
    / dy`` are cm/pixel for *this panel* -- per-axis spatial calibration,
    which can differ between panels if probes have different geometries.

    Units of ``x1`` / ``x2`` / ``y1`` / ``y2`` are pixels in the full-frame
    display coordinate system. Note the COM API uses 1-based pixel
    indexing (so x1=73 means the ROI's leftmost column is the 73rd pixel
    from the left edge); ``width`` and ``height`` are inclusive pixel
    counts (``x2 - x1 + 1``, ``y2 - y1 + 1``).
    """

    img_id: int
    x1: int
    x2: int
    y1: int
    y2: int
    width: int
    height: int
    physical_dx_cm_per_px: float
    physical_dy_cm_per_px: float

    @classmethod
    def from_cmd(cls, cmd, img_id: int = 1) -> Optional["TelemedRoi"]:
        """Probe one img_id; ``None`` if the panel isn't active.

        AutoInt1's Get* calls behave inconsistently for inactive panels:
        - Sometimes they raise (we catch).
        - Sometimes they return the zero-rect sentinel ``(0,0,0,0)``
          (observed 2026-05-24 on the usl02 single-probe probe -- img_ids
          2/3/4 all came back as ``(0,0,0,0)`` rather than raising).
        - Inverted or negative-coordinate rectangles also indicate
          'not present'.

        We reject anything that isn't a strict positive-area rectangle.
        """
        try:
            x1 = int(cmd.GetUltrasoundX1(img_id))
            x2 = int(cmd.GetUltrasoundX2(img_id))
            y1 = int(cmd.GetUltrasoundY1(img_id))
            y2 = int(cmd.GetUltrasoundY2(img_id))
            dx = float(cmd.GetUltrasoundPhysicalDeltaX(img_id))
            dy = float(cmd.GetUltrasoundPhysicalDeltaY(img_id))
        except Exception:  # noqa: BLE001
            return None
        # Strict: require positive width AND height. Catches both the
        # ``(0,0,0,0)`` sentinel (would have given width=1/height=1 with
        # the old ``<`` check) and any inverted / negative rectangle.
        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
            return None
        return cls(
            img_id=img_id,
            x1=x1,
            x2=x2,
            y1=y1,
            y2=y2,
            width=x2 - x1 + 1,
            height=y2 - y1 + 1,
            physical_dx_cm_per_px=dx,
            physical_dy_cm_per_px=dy,
        )


# B-mode img_ids per AutoInt1Client.txt (1=B, 2=B2, 3=B3, 4=B4).
# Higher ids (7=M, 8=PW, 9=CW) are non-B-mode and intentionally excluded
# -- this module is the B-mode imaging pipeline.
_B_MODE_IMG_IDS: tuple[int, ...] = (1, 2, 3, 4)


def _collect_b_mode_rois(cmd) -> dict[int, TelemedRoi]:
    """Enumerate every active B-mode panel.

    Returns ``{img_id: TelemedRoi}`` for img_ids that report valid
    coordinates. Single-probe recordings -> ``{1: ...}``; dual-probe
    (B+B2 side-by-side) -> ``{1: ..., 2: ...}``; etc.

    The presence of a key here is the authoritative dual-probe signal
    (more reliable than parsing the ``scanning_state`` enum, since the
    same enum can appear with one or two physically-mounted probes).
    """
    out: dict[int, TelemedRoi] = {}
    for img_id in _B_MODE_IMG_IDS:
        roi = TelemedRoi.from_cmd(cmd, img_id=img_id)
        if roi is not None:
            out[img_id] = roi
    return out


# ---------- Per-recording ParamGet snapshot (schema v2) ----------


@dataclass(frozen=True)
class _ParamSpec:
    """One Telemed AutoInt1 ParamGet probe.

    ``name`` is the HDF5-attr suffix (full attr key is ``"param_" + name``);
    ``param_id`` is the numeric identifier defined in
    ``AutoInt1Client.txt``; ``kind`` picks the ParamGet* variant.
    """

    name: str
    param_id: int
    kind: str  # "int" | "bool" | "string"


# Per-recording acquisition parameters snapshotted at extract time
# via the AutoInt1 ParamGet* interface. Failures are absorbed; missing
# values are simply absent from the sidecar (Log.params returns None
# via dict.get). IDs come from ``C:/Program Files/Telemed/Echo Wave
# II Application/EchoWave II/Config/Plugins/AutoInt1Client.txt``.
# Cost is ~15 calls per recording (~<1 s total, negligible against the
# per-file extract).
_PARAM_SPECS: tuple[_ParamSpec, ...] = (
    # Provenance / hardware identity
    _ParamSpec("beamformer_code", 915, "int"),
    _ParamSpec("beamformer_name", 916, "string"),
    _ParamSpec("probe_code", 917, "int"),
    _ParamSpec("probe_name", 918, "string"),
    # Absolute clock anchor -- format "yyyy.MM.dd HH:mm:ss.ffffff".
    # Combined with time_ms[-1] this gives the absolute timestamp of
    # every frame (cine *start* = end - time_ms[-1]/1000).
    _ParamSpec("cine_end_datetime_str", 690, "string"),
    # B-mode acquisition settings (the knobs that affect pixel
    # statistics + cross-recording calibration).
    _ParamSpec("b_depth", 305, "int"),
    _ParamSpec("b_frequency", 300, "int"),
    _ParamSpec("b_gain", 309, "int"),
    _ParamSpec("b_power", 307, "int"),
    _ParamSpec("b_dynamic_range", 311, "int"),
    _ParamSpec("b_focus_depth", 302, "int"),
    _ParamSpec("b_focuses_count", 334, "int"),
    _ParamSpec("b_is_dynamic_focus", 171, "bool"),
    _ParamSpec("b_thi", 177, "bool"),
    _ParamSpec("b_frame_averaging", 326, "int"),
    _ParamSpec("b_rejection", 327, "int"),
    _ParamSpec("b_image_enhancement", 328, "bool"),
    _ParamSpec("b_image_enhancement_method", 336, "int"),
    _ParamSpec("b_speckle_reduction", 330, "bool"),
    _ParamSpec("b_speckle_reduction_level", 337, "int"),
    _ParamSpec("b_palette", 338, "int"),
    _ParamSpec("b_palette_gamma", 313, "int"),
    _ParamSpec("b_palette_brightness", 315, "int"),
    _ParamSpec("b_palette_contrast", 317, "int"),
    _ParamSpec("b_palette_negative", 319, "bool"),
    # B-mode geometry / orientation -- cross-machine consistency
    # detector. The L/R-flip class of bug shows up here:
    # ``b_is_scan_direction_changed`` differs across machines when
    # operators have toggled the scan-direction button before saving.
    # (No companion getter exists for vertical flip / id_b_flip_up_down
    # (105) -- AutoInt1 exposes it as a toggle command only, so U/D
    # mismatches can't be detected from the sidecar.)
    _ParamSpec("b_is_scan_direction_changed", 133, "bool"),
    _ParamSpec("b_rotate", 132, "int"),
    _ParamSpec("b_view_area", 324, "int"),
    _ParamSpec("b_scan_type", 340, "int"),
    _ParamSpec("b_steering_trapezoid_angle", 339, "int"),
    _ParamSpec("b_lines_density", 332, "int"),
    _ParamSpec("b_zoom_factor", 112, "int"),
    # Sanity / scanning-state context at extract time. Saved .tvd
    # should report file-opened=True, probe-active=False; a probe-
    # active=True snapshot means we extracted while a live probe was
    # attached (weird but not necessarily wrong).
    _ParamSpec("is_usg_file_opened", 191, "bool"),
    _ParamSpec("scanning_state", 200, "int"),
    _ParamSpec("is_probe_active", 103, "bool"),
)


def _safe_param_get(cmd, spec: _ParamSpec) -> Optional[Any]:
    """One ParamGet call; ``None`` on any failure.

    Some IDs are only valid during a live scan (not on a saved .tvd
    once the probe is detached) -- the COM will raise; we degrade
    silently so the rest of the snapshot still lands.
    """
    try:
        if spec.kind == "int":
            return int(cmd.ParamGetInt(spec.param_id))
        if spec.kind == "bool":
            return bool(cmd.ParamGetBool(spec.param_id))
        if spec.kind == "string":
            return str(cmd.ParamGetString(spec.param_id))
    except Exception:  # noqa: BLE001
        return None
    return None


def _collect_params(cmd) -> dict[str, Any]:
    """Best-effort ParamGet sweep over :data:`_PARAM_SPECS`.

    Returns ``{attr_key: value}`` ready to merge into HDF5 root attrs;
    failed probes are absent (rather than carrying a sentinel) because
    HDF5 attrs can't carry None and any sentinel would collide with
    valid values on some param.
    """
    out: dict[str, Any] = {}
    for spec in _PARAM_SPECS:
        v = _safe_param_get(cmd, spec)
        if v is not None:
            out[f"param_{spec.name}"] = v
    return out


@dataclass
class TelemedRecordingMeta:
    """Per-recording metadata captured alongside per-frame timing.

    Persisted into the HDF5 sidecar's root attributes so downstream
    code can reproduce crops + scale physical measurements without
    re-opening the .tvd through EchoWave.

    Schema v2 (2026-05-23) added an opportunistic ``params`` dict
    populated via :data:`_PARAM_SPECS` -- best-effort probe / beamformer
    identity + cine-end timestamp + B-mode acquisition settings. The
    keys are pre-prefixed (``"param_..."``); failed probes are absent.

    Schema v1a3 (2026-05-24, formerly v3) expanded :data:`_PARAM_SPECS`
    with B-mode geometry / orientation probes (scan-direction-changed,
    rotate, view-area, scan-type, steering angle, lines density, zoom),
    pixel-semantics gaps (power, rejection, palette ID, palette-
    negative, dynamic-focus, enhancement / speckle-reduction *levels*
    alongside the existing enable bools), and sanity probes
    (file-opened, scanning-state, probe-active). Lets cross-machine
    cohort audits flag silent geometry mismatches.

    Schema v1a4 (2026-05-24, formerly v4) replaced the single
    ``b_mode_roi`` with a dict of ROIs keyed by ``img_id`` (1=B, 2=B2,
    3=B3, 4=B4) so dual-probe / multi-image recordings can be split
    losslessly at the encode step. Pixel-resolution lifted into each
    ROI (per-axis cm/px can differ between panels when probes have
    different geometries). Root attrs gained ``n_b_images`` (count)
    and ``roi{N}_*`` / ``physical_d{x,y}{N}_cm_per_px`` blocks per
    active img_id.

    Schema v1a5 (2026-05-24) adds **display-scale** root attrs
    ``image_dx_cm_per_px`` and ``image_dy_cm_per_px``: cm-per-pixel
    derived from ``b_depth_mm / 10 / panel_height_px`` per Telemed
    support's "trust the depth setting" calibration. These are the
    scales downstream measurement code should use for tracked-point
    cm conversion (``physical_d{x,y}{N}_cm_per_px`` is the
    beamformer-native scale, kept for hardware provenance but ~2%
    off the display scale on typical acquisitions). Global per
    recording (not per-img_id), because the depth knob is global in
    EchoWave and the square-pixel display assumption holds across
    all probed Telemed configurations.

    **Inner-image autocrop is intentionally NOT in the schema.** The
    encoder detects the inner ultrasound image (depth ruler, side
    margins, bottom-tick row stripped) from frame pixels at encode
    time, not extract time. Keeping the autocrop bounds out of the
    sidecar means a detector tweak only requires a re-encode (offline),
    not a re-extract (Admin EchoWave). See ``_encode._detect_image_roi``
    for the algorithm; the result is deterministic given the same
    panel pixels.

    Versioning: ``schema_version`` is a string. During pre-release
    iteration it carries an alpha suffix (``"v1aN"``); at public
    release it collapses to ``"v1"``. ``Log`` accepts both the
    current string form and legacy integer values (1-4 = v1a1-v1a4)
    via the back-compat path.
    """

    n_frames: int
    full_frame_width: int
    full_frame_height: int
    b_mode_rois: dict[int, TelemedRoi]
    image_dx_cm_per_px: Optional[float]
    image_dy_cm_per_px: Optional[float]
    source_tvd_path: str
    extracted_at_iso: str
    schema_version: str = "v1"
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def n_b_images(self) -> int:
        """Number of active B-mode panels (1 single-probe, 2 dual-probe)."""
        return len(self.b_mode_rois)

    def to_flat_attrs(self) -> dict:
        """Flatten for HDF5 root-attribute persistence (no nested dicts).

        Per-ROI fields expand to ``roi{N}_x1`` / ``...`` /
        ``physical_dx{N}_cm_per_px`` blocks; ``params`` is merged in
        as-is (keys already carry the ``param_`` prefix);
        ``image_d{x,y}_cm_per_px`` are skipped if None.
        """
        d = {
            k: v
            for k, v in asdict(self).items()
            if k not in ("b_mode_rois", "params", "image_dx_cm_per_px", "image_dy_cm_per_px")
        }
        d["n_b_images"] = self.n_b_images
        if self.image_dx_cm_per_px is not None:
            d["image_dx_cm_per_px"] = self.image_dx_cm_per_px
        if self.image_dy_cm_per_px is not None:
            d["image_dy_cm_per_px"] = self.image_dy_cm_per_px
        for img_id, roi in self.b_mode_rois.items():
            for k, v in asdict(roi).items():
                if k == "img_id":
                    continue
                if k in ("physical_dx_cm_per_px", "physical_dy_cm_per_px"):
                    axis = "dx" if k.startswith("physical_dx") else "dy"
                    d[f"physical_{axis}{img_id}_cm_per_px"] = v
                else:
                    d[f"roi{img_id}_{k}"] = v
        d.update(self.params)
        return d

    @classmethod
    def from_cmd(cls, cmd, source_tvd_path: Union[str, Path]) -> "TelemedRecordingMeta":
        # Need to load frame 1 once to populate width/height.
        cmd.GoToFrame1n(1, True)
        rois = _collect_b_mode_rois(cmd)
        if not rois:
            # Defensive: should never happen on a real recording -- at
            # minimum img_id=1 is always populated for B-mode. If it
            # does, fail loud rather than write a sidecar without any
            # spatial reference.
            raise RuntimeError(
                "No B-mode ROIs detected via AutoInt1 (img_id 1..4 all "
                "returned invalid coordinates). Is a recording loaded?"
            )
        params = _collect_params(cmd)
        # Display scale = depth_cm / panel_height_px ; per Telemed
        # support's "trust the depth setting" calibration. b_depth is
        # reported in mm (verified on usl02 + pia02 cohorts: 5 cm
        # depth setting => params['b_depth'] == 50). Skip if b_depth
        # wasn't captured. Square-pixel assumption holds for all
        # known Telemed configurations (probe physical_dx ==
        # physical_dy across both cohorts probed 2026-05-24), so
        # image_dx == image_dy globally.
        depth_mm = params.get("param_b_depth")
        primary_height = rois[1].height
        if depth_mm is not None and primary_height > 0:
            image_d = (float(depth_mm) / 10.0) / float(primary_height)
        else:
            image_d = None
        return cls(
            n_frames=int(cmd.GetFramesCount),
            full_frame_width=int(cmd.GetLoadedFrameWidth),
            full_frame_height=int(cmd.GetLoadedFrameHeight),
            b_mode_rois=rois,
            image_dx_cm_per_px=image_d,
            image_dy_cm_per_px=image_d,
            source_tvd_path=str(source_tvd_path),
            extracted_at_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
            params=params,
        )


# ---------- Reader ----------


_PROGID = "EchoWave2.CmdInt1"


class TelemedTvdReader:
    """Wraps the EchoWave II AutoInt1 COM interface.

    A single reader instance is fine for the lifetime of the EchoWave
    process; ``open()`` can be called repeatedly to switch files
    (Echo Wave is single-document, so the previously-open file is
    closed implicitly).

    See module docstring for setup prereqs.
    """

    def __init__(self):
        self._cmd = None
        self._opened: Optional[Path] = None

    def connect(self) -> "TelemedTvdReader":
        """Attach to the running Echo Wave II instance via the COM ROT.

        Raises:
            RuntimeError: If GetActiveObject fails (Echo Wave not
                running, COM ProgID not registered, or elevation
                mismatch).
        """
        try:
            import win32com.client
        except ImportError as e:
            raise RuntimeError(
                "pywin32 is required for the COM .tvd path. "
                "Install with: conda install -c conda-forge pywin32"
            ) from e
        try:
            self._cmd = win32com.client.GetActiveObject(_PROGID)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"GetActiveObject('{_PROGID}') failed: {e}\n"
                "Causes:\n"
                "  - Echo Wave II is not running -- start it.\n"
                "  - Echo Wave II is not 'Run as administrator'.\n"
                "  - This Python process is not 'Run as administrator'.\n"
                "  - AutoInt1.dll is not registered (one-time fix:\n"
                "    run AutoInt1_regasm.bat as Admin)."
            ) from e
        return self

    def _require_cmd(self):
        if self._cmd is None:
            raise RuntimeError("Not connected. Call .connect() first.")
        return self._cmd

    def _require_open(self):
        cmd = self._require_cmd()
        if self._opened is None:
            raise RuntimeError("No file open. Call .open(path) first.")
        return cmd

    def open(self, tvd_path: Union[str, Path]) -> "TelemedTvdReader":
        """Open a .tvd file in Echo Wave (idempotent freeze/stop).

        Stops any running scan, recording, or cine playback before
        opening (per AutoInt1Client.txt those states prevent
        GoToFrame1n navigation).

        Raises:
            FileNotFoundError: If ``tvd_path`` doesn't exist.
            RuntimeError: If EchoWave's OpenFile returns -1 (common
                cause: the file is on a network drive -- EchoWave's
                OpenFile fails on UNC / mapped network paths in our
                setup, so callers should copy to local first).
        """
        cmd = self._require_cmd()
        p = Path(tvd_path)
        if not p.is_file():
            raise FileNotFoundError(p)

        # Property-style access fires the call. See module docstring.
        if cmd.IsRecordingState == 1:
            _ = cmd.RecordStop
        if cmd.IsRunState == 1:
            _ = cmd.FreezeRun
        if cmd.IsPlayState == 1:
            _ = cmd.PlayPause

        if cmd.OpenFile(str(p)) == -1:
            raise RuntimeError(
                f"OpenFile failed for {p}. If the file is on a network "
                f"drive, copy it to a local path first -- EchoWave's "
                f"OpenFile fails on UNC / mapped network paths."
            )
        self._opened = p
        return self

    @property
    def opened_path(self) -> Optional[Path]:
        return self._opened

    @property
    def n_frames(self) -> int:
        return int(self._require_open().GetFramesCount)

    @property
    def b_mode_rois(self) -> dict[int, TelemedRoi]:
        """All active B-mode ROIs keyed by img_id (1=B, 2=B2, ...)."""
        return _collect_b_mode_rois(self._require_open())

    @property
    def b_mode_roi(self) -> TelemedRoi:
        """Convenience: the primary (img_id=1) B-mode ROI.

        Equivalent to ``self.b_mode_rois[1]``. Raises ``KeyError`` if
        img_id=1 is somehow not present (unexpected on any valid
        recording).
        """
        return self.b_mode_rois[1]

    def get_frame_time_ms(self, frame_idx_0n: int) -> float:
        """Time of frame ``frame_idx_0n`` in ms, with frame 0 -> 0.0.

        ``frame_idx_0n`` is 0-indexed for Python convention; the
        underlying COM API is 1-indexed and the conversion happens
        here.
        """
        cmd = self._require_open()
        if not (0 <= frame_idx_0n < self.n_frames):
            raise IndexError(f"frame_idx_0n {frame_idx_0n} out of range " f"[0, {self.n_frames})")
        cmd.GoToFrame1n(frame_idx_0n + 1, False)
        return float(cmd.GetCurrentFrameTime)

    def get_frame_gray(self, frame_idx_0n: int):
        """Get uint8 grayscale pixel array for the given frame.

        Returns a numpy array of shape (H, W) covering the FULL Echo
        Wave display, not the B-mode ROI. Crop yourself via
        :attr:`b_mode_roi`.
        """
        import numpy as np

        cmd = self._require_open()
        if not (0 <= frame_idx_0n < self.n_frames):
            raise IndexError(f"frame_idx_0n {frame_idx_0n} out of range " f"[0, {self.n_frames})")
        cmd.GoToFrame1n(frame_idx_0n + 1, True)
        return np.asarray(cmd.GetLoadedFrameGray, dtype=np.uint8)

    def extract_metadata(self) -> TelemedRecordingMeta:
        """Snapshot per-recording metadata for sidecar persistence."""
        return TelemedRecordingMeta.from_cmd(self._require_open(), source_tvd_path=self._opened)


# ---------- Module-level conveniences ----------


def connect() -> TelemedTvdReader:
    """Build + connect a :class:`TelemedTvdReader` in one call."""
    r = TelemedTvdReader()
    r.connect()
    return r


def _sidecar_h5_path(tvd_path: Path) -> Path:
    """Composite-suffix sidecar name: ``<stem>.tvd.h5``.

    Composite suffix (matches the ``.dnav-toc`` convention) so
    downstream tools walking ``*.h5`` don't accidentally pick these
    up as unrelated HDF5 data.
    """
    return tvd_path.with_suffix(tvd_path.suffix + ".h5")


def _is_network_path(p: Path) -> bool:
    """Detect Windows UNC paths and mapped network drives.

    UNC: starts with ``\\\\``. Mapped drives: we'd need ``net use`` to
    distinguish from local letters, which is heavier than warranted --
    so we conservatively only flag UNC here, plus drive letters not
    in ``{C}`` by default. Caller can override via the explicit
    ``copy_to_local`` flag.
    """
    s = str(p)
    if s.startswith("\\\\") or s.startswith("//"):
        return True
    # Heuristic: anything not on C: is treated as potentially-network
    # for the copy-to-local path. Praneeth's M:, S:, etc. are mapped
    # network shares in his setup.
    if len(s) >= 2 and s[1] == ":" and s[0].upper() != "C":
        return True
    return False


# ---------- Staging (background-prefetch primitives) ----------


@dataclass
class _StagedFile:
    """One .tvd ready for COM extraction.

    Produced by :func:`_stage_one` on a background I/O thread so the
    main-thread COM extract of the previous file overlaps with the
    network copy of this one.
    """

    src_tvd: Path  # original (possibly network) path
    dst_h5: Path  # where the final .h5 must end up (sibling of src)
    local_tvd: Path  # what _extract_one should open
    local_h5: Path  # what _extract_one should write
    stage_dir: Optional[Path]  # temp dir to clean up; None if no local copy


def _stage_one(
    src_tvd: Path,
    dst_h5: Path,
    *,
    use_copy: bool,
    temp_root: Path,
    progress: bool = False,
) -> _StagedFile:
    """Prepare one .tvd for extraction; copy to local temp if requested.

    When ``progress=True`` the copy is bracketed with two timestamped
    log lines (start + duration) so the user sees what the bg worker
    is doing while the main thread is busy with COM extraction.
    """
    if not use_copy:
        return _StagedFile(src_tvd, dst_h5, src_tvd, dst_h5, None)
    stage_dir = Path(tempfile.mkdtemp(prefix="telemed_tvd_", dir=temp_root))
    local_tvd = stage_dir / src_tvd.name
    try:
        size_bytes = src_tvd.stat().st_size
    except OSError:
        size_bytes = 0
    _log(
        f"staging {src_tvd.name} -> {stage_dir} ({_size_human(size_bytes)})...",
        tag="stage",
        progress=progress,
    )
    t0 = time.perf_counter()
    shutil.copy2(src_tvd, local_tvd)
    _log(
        f"staged {src_tvd.name} in {time.perf_counter() - t0:.1f} s",
        tag="stage",
        progress=progress,
    )
    return _StagedFile(
        src_tvd=src_tvd,
        dst_h5=dst_h5,
        local_tvd=local_tvd,
        local_h5=_sidecar_h5_path(local_tvd),
        stage_dir=stage_dir,
    )


def _unstage_one(
    staged: _StagedFile,
    *,
    upload: bool,
    progress: bool = False,
) -> None:
    """Copy the resulting .h5 back next to the source (if requested)
    and clean up the local staging dir.

    ``upload=False`` skips the copy-back -- used when the extract
    failed and there's nothing usable to upload. ``progress=True``
    bracket-logs the upload + cleanup phases.
    """
    try:
        if upload and staged.stage_dir is not None and staged.local_h5.exists():
            try:
                size_bytes = staged.local_h5.stat().st_size
            except OSError:
                size_bytes = 0
            _log(
                f"uploading {staged.local_h5.name} "
                f"({_size_human(size_bytes)}) -> {staged.dst_h5.parent}...",
                tag="upload",
                progress=progress,
            )
            t0 = time.perf_counter()
            shutil.copy2(staged.local_h5, staged.dst_h5)
            _log(
                f"uploaded {staged.local_h5.name} in " f"{time.perf_counter() - t0:.1f} s",
                tag="upload",
                progress=progress,
            )
    finally:
        if staged.stage_dir is not None:
            _log(
                f"cleaning up {staged.stage_dir}",
                tag="cleanup",
                progress=progress,
            )
            shutil.rmtree(staged.stage_dir, ignore_errors=True)


def _extract_one(
    tvd_path: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    *,
    reader: Optional[TelemedTvdReader] = None,
    frames: bool = True,
    compression: str = "gzip",
    compression_opts: int = 4,
    progress: bool = True,
) -> Path:
    """Extract one .tvd's timing + metadata + (optionally) frames to HDF5.

    Internal single-file primitive. Public callers should use
    :func:`export`, which accepts the same kwargs plus handles
    folders / lists and the network-drive copy-to-local dance.

    Output HDF5 schema (v1):

    * Root attributes (flat): ``n_frames``, ``full_frame_width``,
      ``full_frame_height``, ``n_b_images``,
      ``image_dx_cm_per_px``, ``image_dy_cm_per_px`` (display scale --
      omit if ``params["b_depth"]`` was not captured),
      ``source_tvd_path``, ``extracted_at_iso``,
      ``schema_version="v1"``, plus per-active-img_id blocks
      ``roi{N}_x1`` / ``roi{N}_x2`` / ``roi{N}_y1`` / ``roi{N}_y2`` /
      ``roi{N}_width`` / ``roi{N}_height`` and
      ``physical_dx{N}_cm_per_px`` / ``physical_dy{N}_cm_per_px``
      for N in 1..4 (1=B, 2=B2, ...). Plus ``tvd_declared_n_frames``
      (the frame count from the .tvd container header, when parseable;
      compared against ``n_frames`` by ``telemed.verify_complete`` to
      flag memory-truncated loads). Plus opportunistic ``param_*``
      attrs from :data:`_PARAM_SPECS` (probe / beamformer identity,
      cine-end timestamp, B-mode acquisition + geometry / orientation
      + sanity probes); failed probes are absent. **Inner-image
      autocrop bounds are NOT stored** -- the encoder detects the
      inner ultrasound image from frame pixels at encode time so
      detector tweaks don't require re-extraction.
    * ``/timing/frame_idx_1n`` -- int32 (N,)
    * ``/timing/time_ms`` -- float64 (N,)
    * ``/timing/ifi_ms`` -- float64 (N,)
    * ``/frames/gray`` -- uint8 (N, H, W) [only when ``frames=True``]

    Args:
        tvd_path: Source .tvd file. Must be on a local drive --
            EchoWave's OpenFile fails on UNC / mapped-network paths
            (the :func:`export` wrapper handles this).
        out_path: Output HDF5 path. Defaults to ``<stem>.tvd.h5``
            next to the source.
        reader: Optional already-connected reader (amortise the COM
            connect step across many files).
        frames, compression, compression_opts, progress: As for
            :func:`export`.

    Returns:
        Path to the written HDF5 file.
    """
    import h5py
    import numpy as np

    p = Path(tvd_path)
    out = Path(out_path) if out_path is not None else _sidecar_h5_path(p)

    r = reader if reader is not None else connect()
    r.open(p)
    cmd = r._cmd
    meta = r.extract_metadata()
    n = meta.n_frames

    # Fail fast on the EchoWave < 4.4.0 LUT-inversion bug before paying
    # for the full frame walk. The .tvd device data is fine; this
    # EchoWave build just inverts the grayscale on read, so the fix is
    # to re-extract on a 4.4.0+ machine. Only meaningful when pulling
    # pixels (timing-only extracts have nothing to invert).
    if frames and _reader_looks_lut_inverted(r, meta):
        raise RuntimeError(
            f"{p.name}: extracted frames look LUT-inverted (bright "
            f"background; primary-panel median > "
            f"{_LUT_INVERSION_MEDIAN_THRESHOLD:.0f}). This is the known "
            f"EchoWave < 4.4.0 grayscale-LUT bug -- the .tvd device data "
            f"is fine, but this EchoWave build inverts it on read. "
            f"Re-extract on a machine running EchoWave 4.4.0+."
        )

    # Recorded frame count from the .tvd header (independent of how many
    # frames EchoWave actually loaded). Stored alongside the data so a
    # later telemed.verify_complete() can flag a memory-truncated load.
    declared_n_frames = read_tvd_n_frames(p)
    if declared_n_frames is not None and declared_n_frames - n > _TVD_TRUNCATION_TOLERANCE:
        _log(
            f"WARNING: {p.name} extracted {n} frames but the .tvd header "
            f"declares {declared_n_frames} (~{declared_n_frames - n} "
            f"missing). EchoWave most likely truncated the load to fit "
            f"available memory -- free RAM and re-extract, or audit with "
            f"telemed.verify_complete().",
            progress=progress,
        )

    # Pre-allocate timing arrays (cheaper than building a DataFrame
    # per frame inside the hot loop).
    times = np.zeros(n, dtype=np.float64)
    frames_arr = None
    if frames:
        # Full-frame stack; consumer crops via root attrs.
        h = meta.full_frame_height
        w = meta.full_frame_width
        frames_arr = np.empty((n, h, w), dtype=np.uint8)

    # tqdm if available + requested; gracefully degrade to a silent
    # loop otherwise. The bar's ``desc`` carries the file stem so
    # batch logs are readable.
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(
                total=n,
                desc=p.stem,
                unit="frame",
                unit_scale=False,
                leave=False,
            )
        except ImportError:
            bar = None

    t0 = time.perf_counter()
    try:
        for i in range(1, n + 1):
            # load_frame_data only matters when we'll pull pixels.
            cmd.GoToFrame1n(i, frames)
            times[i - 1] = float(cmd.GetCurrentFrameTime)
            if frames:
                frames_arr[i - 1] = np.asarray(cmd.GetLoadedFrameGray, dtype=np.uint8)
            if bar is not None:
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()
    walk_s = time.perf_counter() - t0
    if progress:
        print(f"  walk: {walk_s:.1f} s  ({n / walk_s:.1f} fps)", flush=True)

    ifi = np.zeros(n, dtype=np.float64)
    ifi[1:] = np.diff(times)

    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as h5:
        for k, v in meta.to_flat_attrs().items():
            h5.attrs[k] = v
        if declared_n_frames is not None:
            h5.attrs["tvd_declared_n_frames"] = int(declared_n_frames)
        tg = h5.create_group("timing")
        tg.create_dataset("frame_idx_1n", data=np.arange(1, n + 1, dtype=np.int32))
        tg.create_dataset("time_ms", data=times)
        tg.create_dataset("ifi_ms", data=ifi)
        if frames:
            fg = h5.create_group("frames")
            kwargs = {}
            if compression is not None:
                kwargs["compression"] = compression
                if compression == "gzip":
                    kwargs["compression_opts"] = compression_opts
            fg.create_dataset(
                "gray",
                data=frames_arr,
                chunks=(1, meta.full_frame_height, meta.full_frame_width),
                **kwargs,
            )
    return out


def _normalize_sources(
    source: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    recursive: bool,
    pattern: str,
) -> list[Path]:
    """Resolve ``source`` to a de-duplicated list of .tvd file paths.

    ``source`` may be: a single file path, a single directory, or an
    iterable mixing both. Directories are walked for ``pattern`` files
    (recursively when ``recursive=True``); files matching ``pattern``
    are taken as-is even if they wouldn't match the glob (caller
    explicitly named them). De-duplication is by ``Path.resolve()`` so
    overlapping roots / repeated entries don't double-process.
    """
    if isinstance(source, (str, Path)):
        entries = [Path(source)]
    else:
        entries = [Path(s) for s in source]

    seen: set = set()
    files: list[Path] = []
    for entry in entries:
        if entry.is_file():
            candidates = [entry]
        elif entry.is_dir():
            candidates = sorted(entry.rglob(pattern) if recursive else entry.glob(pattern))
        else:
            # Non-existent or special: skip silently. The caller's
            # results dict will simply lack the entry.
            continue
        for fp in candidates:
            key = fp.resolve()
            if key in seen:
                continue
            seen.add(key)
            files.append(fp)
    return files


def export_h5(
    source: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    recursive: bool = True,
    pattern: str = "*.tvd",
    skip_existing: bool = True,
    frames: bool = True,
    compression: str = "gzip",
    compression_opts: int = 4,
    copy_to_local: Optional[bool] = None,
    local_temp_root: Optional[Union[str, Path]] = None,
    keep_full_speed: bool = True,
    postprocess: Optional[Callable[["_StagedFile", bool], None]] = None,
    progress: bool = True,
    progress_callback: Optional[Callable[[int, int, Path, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict:
    """Extract Telemed ``.tvd`` recording(s) to HDF5 sidecar(s).

    Stage 1 of the canonical pipeline (``.tvd -> .tvd.h5 -> .mp4``).
    Requires Administrator-mode EchoWave II + Administrator-mode
    Python. The encode stage (``export_video``) is offline and runs
    in any Python. The composite :func:`process` does both in one
    call.

    ``source`` may be:

    * A path to a single ``.tvd`` file.
    * A directory (walked for ``pattern`` files; ``recursive=True`` by
      default).
    * An iterable of any combination of the above.

    For each .tvd, a sibling ``<stem>.tvd.h5`` is written. With
    ``skip_existing=True`` (default), files whose sidecar already
    exists are skipped, so re-running on a partly-processed corpus
    picks up where the previous run left off.

    Opens **one** Echo Wave COM connection for the entire job (cheap
    per-file overhead).

    **Network-drive handling.** EchoWave's ``OpenFile`` fails on UNC /
    mapped-network paths. When ``copy_to_local`` is True (or ``None``
    -- the default -- and the source looks like a network path: non-C:
    drive letter, or starts with ``\\\\``), each source is copied to a
    unique subdir of ``local_temp_root`` (default: system temp dir),
    processed there, and the resulting HDF5 is copied back next to the
    original ``.tvd``. The local staging dir is cleaned up after each
    file (even on error).

    Args:
        source: File path, directory, or iterable of either. See
            above for shape semantics.
        recursive: If True (default), recurse into subdirectories
            when walking directories. Ignored for individual file
            entries.
        pattern: Glob for file selection inside walked directories
            (default ``"*.tvd"``). Ignored for individual file
            entries.
        skip_existing: If True (default), skip files whose ``.tvd.h5``
            sidecar already exists at the destination.
        frames: If True (default), include raw grayscale frames in
            the HDF5 sidecar. Pass False for a fast timing-only
            extraction (~3x faster, much smaller output).
        compression: HDF5 compression for the frames dataset.
            ``"gzip"`` (default) is lossless and ~5x smaller for
            typical ultrasound; ``"lzf"`` is faster but ~30% larger;
            ``None`` skips compression entirely.
        compression_opts: gzip level [0-9]; default 4.
        copy_to_local: Force-on/off the network-aware copy. ``None``
            (default) auto-detects per source path.
        local_temp_root: Where to stage local copies. ``None`` uses
            the system temp directory.
        keep_full_speed: If True (default), opt the Python process and
            the running ``EchoWave.exe`` out of Windows background power
            throttling (EcoQoS) before extracting, and inhibit system
            sleep, so the ~5 fps rate holds even when the driving console
            is backgrounded or the RDP session is disconnected. Best-effort
            and no-op off Windows; see :func:`telemed.keep_full_speed`.
        postprocess: Optional hook called on a background worker after
            every COM extract. Signature
            ``fn(staged: _StagedFile, success: bool) -> None``;
            invoked with ``success=True`` when the extract wrote the
            local .h5, ``False`` when it raised. Default (``None``)
            is the legacy "upload .h5 + cleanup local temp"
            behaviour. The dispatcher (:func:`telemed.process`)
            replaces this with a richer hook that also encodes mp4s,
            uploads them, and builds the dnav TOC sidecar -- so the
            cost of encode + TOC hides inside the COM extract window
            of the *next* file. Custom hooks must not raise; they own
            their own error reporting.
        progress: If True (default), print ``[i/N] <filename>`` before
            each file and let the per-file tqdm bar render. False
            suppresses both.
        progress_callback: Optional ``fn(idx, total, path, status)``
            for machine-readable progress -- matches the
            ``dustrack.batch`` convention.
        cancel_check: Optional zero-arg callable polled between
            files. If truthy, the loop exits early; the partial
            results dict is returned; any in-flight local staging
            dir is cleaned up.

    Returns:
        ``{path: status}`` where status is ``"built"`` (just
        extracted), ``"hit"`` (skipped existing), or ``f"error: {msg}"``.

    Examples::

        # One file
        telemed.export("C:/data/scan.tvd")

        # One folder (recursive walk for *.tvd)
        telemed.export("M:/data/pia02")

        # Mix of folders and individual files
        telemed.export([
            "M:/data/pia02",
            "M:/data/pia03",
            "C:/scratch/single.tvd",
        ])

        # Timing only -- fast pass for bulk metadata extraction
        telemed.export("M:/data/pia02", frames=False)
    """
    files = _normalize_sources(source, recursive=recursive, pattern=pattern)
    if not files:
        return {}

    temp_root = Path(local_temp_root) if local_temp_root else Path(tempfile.gettempdir())
    temp_root.mkdir(parents=True, exist_ok=True)

    total = len(files)
    results: dict[str, str] = {}

    # Phase 1 -- triage. Pull out the skip-existing files up front so
    # we don't pay staging cost on hits. The two-phase split also keeps
    # the prefetch pipeline below simpler: it only iterates the files
    # that actually need work.
    to_process: list[tuple[int, Path]] = []  # (global_idx, src_tvd)
    for idx, src_tvd in enumerate(files):
        dst_h5 = _sidecar_h5_path(src_tvd)
        if skip_existing and dst_h5.exists():
            results[str(src_tvd)] = "hit"
            if progress:
                print(f"[{idx + 1}/{total}] {src_tvd.name}  (hit, skip)", flush=True)
            if progress_callback is not None:
                progress_callback(idx, total, src_tvd, "hit")
        else:
            to_process.append((idx, src_tvd))

    if not to_process:
        return results

    def _should_copy(p: Path) -> bool:
        return _is_network_path(p) if copy_to_local is None else copy_to_local

    # Postprocess hook -- default mirrors today's behaviour (upload
    # .h5 + cleanup local temp). The dispatcher overrides with a hook
    # that also encodes mp4s + builds TOCs inside the next file's
    # extract window.
    if postprocess is None:

        def postprocess(staged: _StagedFile, success: bool) -> None:
            _unstage_one(staged, upload=success, progress=progress)

    # Suppress Windows background-throttling of the COM loop so the rate
    # holds when the driving console is backgrounded or the RDP session is
    # disconnected. Best-effort; must never break the extract. EchoWave is
    # already running by here, so its process gets opted out too.
    if keep_full_speed:
        try:
            from . import _winperf

            _winperf.keep_full_speed(log=lambda m: _log(m, progress=progress))
        except Exception as e:  # noqa: BLE001
            _log(f"keep-full-speed guard skipped: {e}", progress=progress)

    _log("connecting to EchoWave...", progress=progress)
    reader = connect()
    _log("connected to EchoWave.", progress=progress)

    # Phase 2 -- prefetch-pipelined extraction. The pool runs two
    # background I/O slots:
    #   * stage(N+1): copy next file from network to local temp
    #   * unstage(N-1): copy this file's .h5 back to network + cleanup
    # while the main thread is busy with the COM extraction of file N.
    # COM ops stay on the main thread (EchoWave + COM has thread
    # affinity); the workers only touch the filesystem.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="telemed-io"
    ) as pool:
        in_flight: deque = deque()

        def _submit_stage(p: Path) -> None:
            in_flight.append(
                pool.submit(
                    _stage_one,
                    p,
                    _sidecar_h5_path(p),
                    use_copy=_should_copy(p),
                    temp_root=temp_root,
                    progress=progress,
                )
            )

        # Prime: stage the first file before entering the loop so the
        # first iteration has work waiting.
        _submit_stage(to_process[0][1])

        for pos, (idx, src_tvd) in enumerate(to_process):
            if cancel_check is not None and cancel_check():
                break

            # Wait for the current file's stage to finish.
            try:
                staged = in_flight.popleft().result()
            except Exception as e:  # noqa: BLE001
                results[str(src_tvd)] = f"error: staging: {e}"
                if progress:
                    print(f"[{idx + 1}/{total}] {src_tvd.name}  (error: staging)", flush=True)
                if progress_callback is not None:
                    progress_callback(idx, total, src_tvd, results[str(src_tvd)])
                continue

            # Pipeline: kick off the next file's stage now so the
            # download runs in parallel with this file's COM extract.
            if pos + 1 < len(to_process):
                _submit_stage(to_process[pos + 1][1])

            if progress:
                print(f"[{idx + 1}/{total}] {src_tvd.name}", flush=True)

            try:
                _extract_one(
                    staged.local_tvd,
                    out_path=staged.local_h5,
                    reader=reader,
                    frames=frames,
                    compression=compression,
                    compression_opts=compression_opts,
                    progress=progress,
                )
                # Fire-and-forget: postprocess (upload + cleanup, plus
                # any caller-supplied work like mp4 encode + TOC) runs
                # on the second background worker while the main
                # thread starts the next file's COM extract.
                pool.submit(postprocess, staged, True)
                results[str(src_tvd)] = "built"
            except Exception as e:  # noqa: BLE001
                # Extraction failed -- caller's postprocess gets
                # success=False so it can skip upload / encode and
                # just clean up the local staging dir.
                pool.submit(postprocess, staged, False)
                results[str(src_tvd)] = f"error: {e}"

            if progress_callback is not None:
                progress_callback(idx, total, src_tvd, results[str(src_tvd)])

        # On cancel: drain any prefetched-but-unprocessed stage so we
        # don't leak its temp dir. (The pool's context-manager exit
        # also waits for any submitted postprocess tasks to complete.)
        while in_flight:
            try:
                staged = in_flight.popleft().result()
            except Exception:  # noqa: BLE001
                continue
            _unstage_one(staged, upload=False)

    return results


# The composite ``process()`` orchestrator lives in ``_dispatch.py``
# (per-file pipelining: encode + TOC + upload hide inside the next
# file's COM extract window). Import via the package: ``from
# telemed import process``.
