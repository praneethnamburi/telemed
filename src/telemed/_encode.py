"""Encode Telemed ``.tvd.h5`` sidecars into mp4 video files.

Consumed by DUSTrack / DLC etc., but the output is just an mp4 -- this
module is named for what it does, not the downstream tool.

Public surface (re-exported from ``telemed``)::

    telemed.export_video(source)         # file | folder | list -> mp4(s)
    telemed.Log("rec.tvd.h5").to_video() # single-recording convenience

Dispatcher convenience::

    telemed.export(source, kind="h5")    # tvd -> tvd.h5 (default)
    telemed.export(source, kind="video") # tvd.h5 -> mp4(s)
    telemed.export(source, kind="both")  # tvd -> tvd.h5 -> mp4(s)

Design choices baked in here:

* **Lossless by default**. The source frames in the .tvd.h5 are uint8
  grayscale straight off the device; there's no upstream lossy step to
  reclaim quality from, so a CRF-tuned encode is buying file size at the
  cost of DLC accuracy that the device gave us for free. The cropped ROI
  is small enough that lossless h265 mono is tolerable -- a typical
  20k-frame pia02 recording lands at ~1-2 GB. Pass ``lossless=False``
  with an explicit ``crf=`` if you want to trade a few percent accuracy
  for ~50x smaller files.

* **Per-panel split for multi-image recordings**. If the sidecar has
  ``n_b_images > 1`` (dual-probe scans), we write one mp4 per active
  img_id -- ``<stem>_b{N}.mp4`` -- each cropped to its own ROI. Single-
  probe stays ``<stem>.mp4``.

* **Autocrop to the inner ultrasound image**. The AutoInt1-reported
  panel ROI is the *full* B-mode panel (depth ruler, side margins,
  bottom-tick row, inner image). The encoder detects the inner image
  by content (gray-margin step + bottom tick-row peel) from a
  16-frame mean of the sidecar's ``/frames/gray`` and crops to that
  box by default (``crop="image"``); recordings where detection
  can't identify the inner box fall back to the panel ROI with a
  warning. ``crop="panel"`` opts out explicitly. **The autocrop
  bounds are NOT persisted in the sidecar** -- detection happens at
  encode time so detector tweaks only need a re-encode, not a
  re-extract.

* **Orientation normalisation**. When ``b_is_scan_direction_changed``
  is True on the sidecar, the output is L/R-flipped during encode so
  every cohort mp4 lands in a canonical orientation regardless of which
  EchoWave operator toggled scan direction. U/D flip has no API getter
  so cannot be detected; rotation handling is deferred until we see a
  recording with non-zero ``b_rotate``.

* **No timing CSV**. The .tvd.h5 already carries /timing/time_ms +
  /timing/ifi_ms; downstream consumers that need real time round-trip
  via ``telemed.Log(<stem>.tvd.h5).time_ms[frame_idx]``. The mp4
  declares CFR at the recording's mean fps so DLC / cv2 index by frame
  number unchanged.
"""

from __future__ import annotations

import os
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import h5py
import numpy as np

# Optional dep: datanavigator's TOC precompute. ``export_video`` builds
# a per-mp4 ``.dnav-toc`` sidecar by default so downstream tools
# (``dnav.VideoReader`` / DUSTrack / DLC via ``patch_dlc_decoder``)
# open without paying the cold-open demux pass. Missing dnav is
# non-fatal -- the encode still succeeds; we warn once and skip the
# sidecar so users without dnav installed aren't blocked.
try:
    from datanavigator.video_reader import precompute_toc as _dnav_precompute_toc

    _HAS_DNAV = True
except ImportError:  # pragma: no cover -- exercised via monkeypatch in tests
    _dnav_precompute_toc = None
    _HAS_DNAV = False

_DNAV_WARNED = False


def _warn_no_dnav_once() -> None:
    """Emit a one-time UserWarning that TOC building was skipped."""
    global _DNAV_WARNED
    if _DNAV_WARNED:
        return
    _DNAV_WARNED = True
    warnings.warn(
        "telemed.export_video: build_toc=True but datanavigator is not "
        "importable; TOC sidecars will not be built. Install datanavigator "
        "to enable, or pass build_toc=False to suppress this notice.",
        stacklevel=3,
    )


def _ensure_toc_sidecar(mp4_path: Path) -> str:
    """Build the dnav ``.dnav-toc`` sidecar for ``mp4_path``; idempotent.

    Returns one of: ``"built"`` (TOC just written), ``"hit"`` (already
    cached), ``"built (uncached)"`` (built but sidecar write failed),
    ``"skipped: no dnav"`` (dnav not importable), or
    ``f"error: {msg}"``. Never raises -- TOC build is best-effort,
    distinct from the encode contract.
    """
    if not _HAS_DNAV:
        _warn_no_dnav_once()
        return "skipped: no dnav"
    try:
        results = _dnav_precompute_toc(
            [str(mp4_path)],
            show_progress=False,
        )
        return results.get(str(mp4_path), "built")
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


# Default CRF matches ``telemed.crop``'s tuning for the lossy fallback
# path. Ignored in the default ``lossless=True`` branch.
_VIDEO_DEFAULT_CRF = 24
# Default preset is ``ultrafast``. For *lossless* h265 the preset only
# trades file size against encode + decode speed -- reconstructed
# pixels are bit-exact regardless. The 2026-05-24 pia02 bench on a
# 19248-frame dual-probe recording landed:
#
#   preset    encode    size       cv2 decode    dnav linear   dnav seeks
#   slow      55 fps    1.00x      507 fps       277 fps       2.2/s
#   medium   110 fps    1.04x      548 fps       290 fps       2.4/s
#   fast     150 fps    1.08x      578 fps       292 fps       2.4/s
#   ultrafast 400 fps   1.15x     1293 fps       553 fps       4.4/s
#
# ultrafast wins on every time axis (encode, linear decode, random
# seek) at ~6% larger files than fast -- the simpler bitstream
# structure is cheaper for both encoder + decoder to process.
# Power users override via ``preset=`` kwarg (e.g. ``"slow"`` for the
# smallest lossless files, at ~7x slower encode + ~2.5x slower
# decode).
_VIDEO_DEFAULT_PRESET = "ultrafast"
_VIDEO_SUPPORTED_CODECS = ("h265_mono",)


# ---------- Inner-image (autocrop) detection ----------
#
# The panel ROI returned by AutoInt1's GetUltrasoundX{1,2}/Y{1,2} is
# the *full B-mode panel*: depth ruler + side margins + bottom-tick
# row + the inner ultrasound image. For DLC / DUSTrack consumption we
# want *just* the inner image -- the depth ruler eats ~25% of the
# panel width on single-probe acquisitions and the bottom-tick row
# pollutes the bottom edge with sharp non-anatomical contrast that
# confuses trackers.
#
# The detector runs at **encode time**, not extract time, so detector
# tweaks only require a re-encode (offline) -- not a re-extract
# (Admin EchoWave). For a fixed (probe, depth, view_area, panel_dims)
# the output is deterministic, so this trades nothing for the
# flexibility of skipping a schema bump.
#
# Strategy: aggregate ~16 evenly-sampled frames into a mean panel,
# then:
#   * Cols: estimate the EchoWave UI gray level from the panel's
#     leftmost+rightmost five columns (median to ignore digit noise
#     in the depth-ruler band), threshold by |col_mean - gray| and a
#     low col-std cap, take the longest contiguous run of non-margin
#     cols as the inner-image width. Tick rows are pre-trimmed for
#     this pass so a saturated bright stripe doesn't inflate col_std.
#   * Rows: walk up from the last panel row, peeling off rows whose
#     col-restricted mean exceeds the tick threshold
#     (``max(60, 2*median + 20)``). Trust the panel ROI for the top
#     (probed Telemed configurations place the depth-ruler "0" digit
#     *above* the panel ROI vertically).
#
# Returns None when no clear inner box is detectable (fully-black
# recording, no clear margin step, unfamiliar UI theme); callers
# fall back to the panel ROI.


def _otsu(values) -> float:
    """1-D Otsu threshold; returns the histogram edge that maximises
    between-class variance. Falls back to ``min(values)`` on a
    degenerate (constant) input so callers always get a defined cutoff.
    """
    import numpy as np

    v = np.asarray(values, dtype=np.float64)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin < 1e-9:
        return vmin
    hist, edges = np.histogram(v, bins=256, range=(vmin, vmax))
    p = hist.astype(np.float64) / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom = np.where(denom < 1e-12, 1.0, denom)
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    sigma_b2[(omega <= 0) | (omega >= 1)] = 0
    k = int(np.argmax(sigma_b2))
    return float(edges[k])


def _longest_run(mask, min_run: int) -> Optional[tuple[int, int]]:
    """Longest contiguous True run in a 1-D bool mask.

    Returns ``(start, end)`` as 0-based half-open coords, or ``None``
    if no run meets ``min_run``.
    """
    import numpy as np

    if not mask.any():
        return None
    m = mask.view(np.int8)
    diff = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lengths = ends - starts
    if lengths.max() < min_run:
        return None
    i = int(np.argmax(lengths))
    return int(starts[i]), int(ends[i])


def _detect_image_roi(
    panel,
    *,
    gray_tol: float = 12.0,
    std_tol: float = 12.0,
    tick_floor: float = 60.0,
    tick_factor: float = 2.0,
    tick_offset: float = 20.0,
    min_run_x: int = 20,
    min_run_y: int = 20,
    min_frac: float = 0.2,
) -> Optional[tuple[int, int, int, int]]:
    """Detect the inner ultrasound image within an aggregated panel.

    ``panel`` is a 2D float array -- typically a multi-frame mean of
    the pixels inside one B-mode panel ROI. Returns
    ``(x_s, x_e, y_s, y_e)`` in **panel-local 0-based half-open**
    coords, or ``None`` if no clear inner box is detectable (caller
    falls back to the panel ROI).

    See the module-level comment above for the algorithm + tuning
    rationale. Defaults were tuned against the 2026-05-24 usl02
    (single-probe) + pia02 (dual-probe) cohorts with the LF9-5N60-A3
    probe at depth=50 mm, view_area=100; both produce ~700x557 inner
    boxes inside their 1409x558 / 705x558 panels.
    """
    import numpy as np

    panel = np.asarray(panel, dtype=np.float64)
    if panel.ndim != 2:
        return None
    H, W = panel.shape
    if H < min_run_y or W < min_run_x:
        return None

    # Pre-trim candidate tick rows for the col-detection pass. A tick
    # row has saturated bright marks spanning the full panel width,
    # which contributes a uniform stripe in every column and inflates
    # col_std (margin cols look "noisy" enough to slip past the
    # margin filter). Trimming on full-row mean is independent of the
    # eventual col bounds, so it's safe to do first.
    full_row_mean = panel.mean(axis=1)
    panel_median = float(np.median(full_row_mean))
    pre_tick_thr = max(tick_floor, tick_factor * panel_median + tick_offset)
    keep_rows = full_row_mean <= pre_tick_thr
    trimmed = panel[keep_rows] if keep_rows.sum() >= min_run_y else panel

    col_mean = trimmed.mean(axis=0)
    col_std = trimmed.std(axis=0)
    edge_cols = np.concatenate([col_mean[:5], col_mean[-5:]])
    gray = float(np.median(edge_cols))
    is_margin_col = (np.abs(col_mean - gray) < gray_tol) & (col_std < std_tol)
    xrange = _longest_run(~is_margin_col, min_run_x)
    if xrange is None:
        return None
    x_s, x_e = xrange
    if (x_e - x_s) < max(min_run_x, int(min_frac * W)):
        return None

    # Row detection: use the ORIGINAL panel restricted to inner cols
    # (so the tick band is still visible and can be peeled off the
    # bottom). Walk up while col-restricted row_mean exceeds the
    # tick threshold derived from this narrower mean.
    sub = panel[:, x_s:x_e]
    row_mean = sub.mean(axis=1)
    med = float(np.median(row_mean))
    tick_thr = max(tick_floor, tick_factor * med + tick_offset)
    i = H - 1
    while i >= 0 and row_mean[i] > tick_thr:
        i -= 1
    y_e = i + 1
    y_s = 0
    if (y_e - y_s) < max(min_run_y, int(min_frac * H)):
        return None
    return (x_s, x_e, y_s, y_e)


def _aggregate_panel_from_h5(
    h5_path: Union[str, os.PathLike],
    roi,
    n_samples: int = 16,
):
    """Mean of ``n_samples`` evenly-spaced panel-cropped frames from
    a sidecar's ``/frames/gray`` dataset. Returns a float64 2D array.

    The output is sized from the *actual* sliced frame shape, not the
    nominal ROI dims, so a panel ROI that extends past the full-frame
    bounds (a few test fixtures construct this impossible geometry,
    and h5py + numpy silently clip) gives a consistent result rather
    than a broadcast error.
    """
    import numpy as np

    with h5py.File(h5_path, "r") as h5:
        ds = h5["frames/gray"]
        n = ds.shape[0]
        idxs = np.linspace(0, n - 1, min(n_samples, n)).astype(int)
        ys = slice(roi.y1 - 1, roi.y2)
        xs = slice(roi.x1 - 1, roi.x2)
        acc = ds[int(idxs[0])][ys, xs].astype(np.float64)
        for i in idxs[1:]:
            acc += ds[int(i)][ys, xs]
    return acc / float(len(idxs))


# ---------- ffmpeg cmd builder + runner (pure helpers; testable) ----------


def _build_ffmpeg_cmd(
    out_path: Union[str, os.PathLike],
    *,
    width: int,
    height: int,
    fps: float,
    codec: str,
    lossless: bool,
    crf: int,
    preset: str,
    vf_chain: Optional[list[str]] = None,
    overwrite: bool,
) -> list[str]:
    """Build the ffmpeg argv that reads raw gray frames from stdin.

    Pure helper (no I/O) so tests can pin the byte-identical default
    invocation as a regression guard.

    ``codec="h265_mono"`` always means ``-c:v libx265 -pix_fmt gray``;
    quality is controlled by ``lossless`` (True -> ``-x265-params
    lossless=1``, no ``-crf``) or ``crf`` (False -> ``-crf {crf}``).
    ``vf_chain`` is an ordered list of ffmpeg ``-vf`` filters (e.g.
    ``["hflip"]``); empty / None skips the ``-vf`` flag entirely.

    ``-hide_banner -loglevel error`` is mandatory: libx265 floods stderr
    with per-frame stats, which fills the Popen PIPE buffer (~64 KB) and
    deadlocks the stdin writer on long encodes.
    """
    if codec not in _VIDEO_SUPPORTED_CODECS:
        raise ValueError(f"codec={codec!r} not supported; options: {_VIDEO_SUPPORTED_CODECS}")
    quality_flags: list[str] = ["-x265-params", "lossless=1"] if lossless else ["-crf", str(crf)]
    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-c:v",
        "libx265",
        "-pix_fmt",
        "gray",
        *quality_flags,
        "-preset",
        preset,
        "-fps_mode",
        "cfr",
        "-an",
    ]
    if vf_chain:
        cmd += ["-vf", ",".join(vf_chain)]
    cmd.append(str(out_path))
    return cmd


def _encode_frames(cmd: list[str], frames_iter: Iterable[np.ndarray]) -> None:
    """Pipe a frame iterator through ffmpeg's stdin.

    Each yielded frame must be C-contiguous uint8 of the shape implied
    by the ``-s WxH`` flag in ``cmd``. Surfaces ffmpeg's stderr on
    non-zero exit so failures don't print silently.
    """
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for fr in frames_iter:
            proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        stderr = (proc.stderr.read() or b"").decode("utf-8", errors="replace")
        raise subprocess.CalledProcessError(
            proc.returncode,
            cmd,
            stderr=(
                f"[telemed.export_video] ffmpeg exited {proc.returncode}.\n"
                f"cmd: {' '.join(str(c) for c in cmd)}\n"
                f"stderr:\n{stderr}"
            ),
        )


# ---------- Orientation normalisation ----------


def _orientation_vf(params: dict) -> list[str]:
    """Return the ``-vf`` filter chain that normalises the recording's
    orientation to canonical (no L/R flip, no rotation).

    Driven by sidecar ``param_*`` attrs:
    * ``b_is_scan_direction_changed`` True -> apply ``hflip``.
    * Rotation handling is deferred -- both probed cohorts (usl02,
      pia02) report ``b_rotate=0``, and the int->degrees enum mapping
      from AutoInt1 isn't documented. We log a warning if a non-zero
      ``b_rotate`` is encountered so it surfaces for follow-up rather
      than silently producing a misoriented mp4.
    """
    vf: list[str] = []
    if params.get("b_is_scan_direction_changed"):
        vf.append("hflip")
    rot = params.get("b_rotate", 0)
    if rot:
        # Don't guess the enum convention; warn loud + leave the pixels
        # alone so the caller can investigate.
        import warnings

        warnings.warn(
            f"telemed.export_video: b_rotate={rot} on this recording, but "
            f"the AutoInt1 rotation enum->degrees mapping is undocumented. "
            f"Output will NOT be rotation-corrected. Inspect the source "
            f"recording and update _orientation_vf() when the convention "
            f"is verified.",
            stacklevel=3,
        )
    return vf


# ---------- Output naming ----------


@dataclass(frozen=True)
class _VideoTarget:
    """One mp4 output for one ROI of one recording.

    ``stem`` is the recording stem (``<basename>`` with ``.tvd.h5``
    stripped). ``img_id`` is the AutoInt1 panel id (1=B, 2=B2, ...).
    Single-probe recordings use ``stem.mp4``; multi-probe use
    ``stem_b{img_id}.mp4``.
    """

    h5_path: Path
    img_id: int
    out_path: Path


def _stem_from_h5(h5_path: Path) -> str:
    """Strip the ``.tvd.h5`` composite suffix; fallback to ``.h5``."""
    name = h5_path.name
    if name.endswith(".tvd.h5"):
        return name[: -len(".tvd.h5")]
    return h5_path.stem


def _plan_targets(
    h5_path: Path,
    img_ids: list[int],
    out_dir: Optional[Path] = None,
) -> list[_VideoTarget]:
    """Build the ``_VideoTarget`` list for one recording.

    Single img_id -> ``<stem>.mp4``. Multiple -> ``<stem>_b{N}.mp4``.
    """
    stem = _stem_from_h5(h5_path)
    base_dir = out_dir if out_dir is not None else h5_path.parent
    single = len(img_ids) == 1
    targets: list[_VideoTarget] = []
    for img_id in sorted(img_ids):
        name = f"{stem}.mp4" if single else f"{stem}_b{img_id}.mp4"
        targets.append(
            _VideoTarget(
                h5_path=h5_path,
                img_id=img_id,
                out_path=base_dir / name,
            )
        )
    return targets


# ---------- Single-recording encode ----------


def _encode_one_panel(
    h5_path: Path,
    img_id: int,
    out_path: Path,
    *,
    codec: str = "h265_mono",
    lossless: bool = True,
    crf: int = _VIDEO_DEFAULT_CRF,
    preset: str = _VIDEO_DEFAULT_PRESET,
    fps: Optional[float] = None,
    normalize_orientation: bool = True,
    crop: str = "image",
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    """Encode one ROI from one .tvd.h5 to one mp4.

    Read frames lazily from ``/frames/gray`` (no full-stack load), crop
    each to the panel's ROI, optionally apply orientation normalisation,
    pipe through ffmpeg. Output dimensions are taken from the ROI
    (cropped) not the full frame.

    ``crop="image"`` (default) crops to the inner ultrasound image
    (depth ruler + side margins + bottom-tick row stripped) when the
    sidecar has v1a6 inner-ROI fields; otherwise falls back to the
    full panel ROI with a warning. ``crop="panel"`` always uses the
    outer B-mode panel (legacy behaviour).

    When ``progress=True`` (default) and ``tqdm`` is importable, a
    per-frame bar shows during the encode. The bar's ``desc`` is the
    output stem so batch logs stay readable across many panels.

    Raises:
        RuntimeError: Sidecar has no ``/frames/gray`` group (was
            extracted with ``frames=False``).
        FileNotFoundError: img_id not present in the sidecar.
        FileExistsError: ``out_path`` exists and ``overwrite=False``.
        ValueError: ``crop`` is neither ``"image"`` nor ``"panel"``.
    """
    if crop not in ("image", "panel"):
        raise ValueError(
            f"crop={crop!r} not supported; use 'image' (inner ultrasound "
            f"image, depth-ruler-free) or 'panel' (outer B-mode panel)."
        )
    # Lazy import to avoid a top-of-module cycle: log -> _encode -> log.
    from .log import Log

    lf = Log(h5_path)
    if not lf.has_frames:
        raise RuntimeError(
            f"{h5_path.name} has no /frames/gray (extracted with "
            f"frames=False). Re-extract with frames=True before encoding."
        )
    if img_id not in lf.b_mode_rois:
        raise FileNotFoundError(
            f"img_id={img_id} not present in {h5_path.name}; available "
            f"img_ids: {sorted(lf.b_mode_rois)}"
        )
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists; pass overwrite=True to clobber.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    roi = lf.b_mode_rois[img_id]
    if crop == "image":
        # Detect on the sidecar's frames; fall back to the panel ROI
        # with a warning when the detector returns None.
        panel_mean = _aggregate_panel_from_h5(h5_path, roi)
        inner = _detect_image_roi(panel_mean)
        if inner is None:
            import warnings

            warnings.warn(
                f"telemed.export_video: detector couldn't identify "
                f"inner ultrasound image for {h5_path.name} "
                f"img_id={img_id}; encoding the outer panel ROI "
                f"(the mp4 will include the depth ruler / side "
                f"margins). Pass crop='panel' to suppress this "
                f"warning when the full panel is intentional.",
                stacklevel=3,
            )
            ys, xs = roi.as_slice()
        else:
            x_s, x_e, y_s, y_e = inner
            ys = slice(roi.y1 - 1 + y_s, roi.y1 - 1 + y_e)
            xs = slice(roi.x1 - 1 + x_s, roi.x1 - 1 + x_e)
    else:
        ys, xs = roi.as_slice()
    width = xs.stop - xs.start
    height = ys.stop - ys.start
    effective_fps = float(fps) if fps is not None else lf.mean_fps
    vf_chain = _orientation_vf(lf.params) if normalize_orientation else []

    cmd = _build_ffmpeg_cmd(
        out_path,
        width=width,
        height=height,
        fps=effective_fps,
        codec=codec,
        lossless=lossless,
        crf=crf,
        preset=preset,
        vf_chain=vf_chain,
        overwrite=overwrite,
    )

    # Per-frame tqdm if available + requested; the ffmpeg subprocess
    # pulls frames lazily from this generator, so wrapping it with the
    # bar gives a true encode-progress signal (libx265's own stderr
    # logging is suppressed by ``-loglevel error`` to keep the Popen
    # PIPE from deadlocking on long encodes -- see _build_ffmpeg_cmd).
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(
                total=lf.n_frames,
                desc=out_path.stem,
                unit="frame",
                unit_scale=False,
                leave=False,
            )
        except ImportError:
            bar = None

    def _iter_frames():
        try:
            with h5py.File(h5_path, "r") as h5:
                ds = h5["frames/gray"]
                for i in range(lf.n_frames):
                    yield ds[i][ys, xs]
                    if bar is not None:
                        bar.update(1)
        finally:
            if bar is not None:
                bar.close()

    _encode_frames(cmd, _iter_frames())
    return out_path


# ---------- Source normalisation (file / folder / list of either) ----------


def _resolve_h5_sources(
    source: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    recursive: bool,
    pattern: str,
) -> list[Path]:
    """Resolve ``source`` to a de-duplicated list of .tvd.h5 paths.

    Accepts:
    * a single .tvd.h5 file (used directly),
    * a single .tvd file (the sibling .tvd.h5 is substituted; missing
      sidecars are silently skipped),
    * a directory (walked for ``pattern`` -- default ``*.tvd.h5``),
    * an iterable of any combination of the above.

    De-duplication is by ``Path.resolve()`` so overlapping roots /
    repeats don't double-encode.
    """
    entries: list[Path]
    if isinstance(source, (str, Path)):
        entries = [Path(source)]
    else:
        entries = [Path(s) for s in source]

    seen: set = set()
    out: list[Path] = []
    for entry in entries:
        if entry.is_file():
            if entry.suffixes[-2:] == [".tvd", ".h5"]:
                candidates = [entry]
            elif entry.suffix == ".tvd":
                sidecar = entry.with_suffix(entry.suffix + ".h5")
                candidates = [sidecar] if sidecar.is_file() else []
            else:
                continue
        elif entry.is_dir():
            candidates = sorted(entry.rglob(pattern) if recursive else entry.glob(pattern))
        else:
            continue
        for fp in candidates:
            key = fp.resolve()
            if key in seen:
                continue
            seen.add(key)
            out.append(fp)
    return out


# ---------- Public batch entry point ----------


def export_video(
    source: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    recursive: bool = True,
    pattern: str = "*.tvd.h5",
    out_dir: Optional[Union[str, Path]] = None,
    codec: str = "h265_mono",
    lossless: bool = True,
    crf: int = _VIDEO_DEFAULT_CRF,
    preset: str = _VIDEO_DEFAULT_PRESET,
    fps: Optional[float] = None,
    normalize_orientation: bool = True,
    crop: str = "image",
    skip_existing: bool = True,
    overwrite: bool = False,
    build_toc: bool = True,
    progress: bool = True,
    progress_callback: Optional[Callable[[int, int, Path, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict:
    """Encode Telemed ``.tvd.h5`` recording(s) to ``.mp4`` file(s).

    Single unified entry point. ``source`` may be:

    * A path to a single ``.tvd.h5`` file (encoded as-is).
    * A path to a ``.tvd`` file (its sibling ``.tvd.h5`` is encoded;
      missing sidecar -> silently skipped).
    * A directory (walked for ``pattern`` -- default ``*.tvd.h5``).
    * An iterable of any combination of the above.

    Per recording, ``n_b_images`` outputs are written: single-probe ->
    ``<stem>.mp4``, multi-probe (n>=2) -> ``<stem>_b{img_id}.mp4`` per
    active panel.

    Args:
        source: File path, directory, or iterable of either.
        recursive: When True (default), recurse into subdirectories
            during ``pattern`` walk.
        pattern: Glob filter for directory walks. Default ``"*.tvd.h5"``.
        out_dir: Output directory. ``None`` (default) co-locates each
            mp4 next to its source ``.tvd.h5``.
        codec: Output codec preset. ``"h265_mono"`` (default).
        lossless: When True (default), produce a lossless h265 mono
            encode (``-x265-params lossless=1``). False uses ``crf``.
            See module docstring for rationale (raw uint8 gray source
            -> nothing to gain from CRF quantisation).
        crf: ffmpeg CRF for the lossy branch. Default 24.
        preset: ffmpeg ``-preset`` value. Default ``"ultrafast"``
            (lossless bit-exact regardless of preset; ``ultrafast``
            trades ~15% larger files for ~7x faster encode + ~2.5x
            faster decode vs ``"slow"``). Pass ``"slow"`` to reclaim
            the smallest lossless files. See module docstring for the
            bench numbers backing the default.
        fps: CFR fps declared in the mp4 container. ``None`` (default)
            uses each recording's ``mean_fps``. Real per-frame timing
            stays in the ``.tvd.h5`` (``/timing/time_ms``).
        normalize_orientation: When True (default), L/R-flip the
            output if the sidecar reports ``b_is_scan_direction_changed=True``
            so every cohort mp4 lands in a canonical orientation. False
            ships pixels as-stored in the sidecar.
        crop: ``"image"`` (default) crops each output to the inner
            ultrasound image (depth ruler / side margins / bottom-tick
            row stripped) when the sidecar carries v1a6 inner-ROI
            fields; falls back to the outer panel with a warning on
            legacy sidecars. ``"panel"`` keeps the outer B-mode panel
            (the pre-v1a6 behaviour) for inspection / debugging.
        skip_existing: When True (default), per-output: skip if the
            target ``.mp4`` already exists. (Each panel checked
            independently for multi-probe sidecars.)
        overwrite: When True, clobber any existing target ``.mp4``.
            Mutually exclusive with ``skip_existing=True``; if both,
            ``overwrite`` wins.
        build_toc: When True (default), build the
            ``<mp4>.dnav-toc`` sidecar after each successful encode
            (and rebuild a missing sidecar on the skip-existing path)
            so ``dnav.VideoReader`` / DUSTrack open the mp4 without
            the cold-open demux pass. Silently skipped (with a one-
            time warning) when ``datanavigator`` isn't importable.
            False suppresses the sidecar build entirely.
        progress: When True (default), print a line per panel encoded
            AND render a per-frame tqdm bar during each panel's encode
            (the bar's ``desc`` is the output mp4 stem). False
            suppresses both.
        progress_callback: Optional ``fn(idx, total, mp4_path, status)``
            -- matches the ``dustrack.batch`` convention. ``idx`` /
            ``total`` are per-PANEL counts, not per-recording.
        cancel_check: Zero-arg callable polled between panels; truthy
            -> exit early.

    Returns:
        ``{mp4_path_str: status}`` where status is ``"built"`` /
        ``"hit"`` (skipped existing) / ``f"error: {msg}"``.

    Examples::

        # One file -- writes a sibling .mp4 (or per-panel .mp4s)
        telemed.export_video("M:/data/scan.tvd.h5")

        # One folder (recursive walk for *.tvd.h5)
        telemed.export_video("M:/data/pia02")

        # Lossy branch
        telemed.export_video("M:/data/pia02", lossless=False, crf=22)
    """
    h5_files = _resolve_h5_sources(
        source,
        recursive=recursive,
        pattern=pattern,
    )
    if not h5_files:
        return {}

    # Lazy import (only when we have work to do).
    from .log import Log

    out_dir_path = Path(out_dir) if out_dir is not None else None

    # Phase 1 -- expand each recording into one target per panel.
    all_targets: list[_VideoTarget] = []
    for h5 in h5_files:
        try:
            img_ids = sorted(Log(h5).b_mode_rois.keys())
        except Exception as e:  # noqa: BLE001
            # Bad sidecar (corrupt / missing required attrs / etc.):
            # surface as a single error result for the recording, no
            # targets queued.
            key = str(h5)
            results: dict = getattr(export_video, "_pending_errors", {})
            results[key] = f"error: load_sidecar: {e}"
            setattr(export_video, "_pending_errors", results)
            continue
        all_targets.extend(_plan_targets(h5, img_ids, out_dir=out_dir_path))

    results: dict[str, str] = {}
    # Surface any pre-loop load errors.
    if hasattr(export_video, "_pending_errors"):
        results.update(export_video._pending_errors)
        del export_video._pending_errors

    total = len(all_targets)
    for idx, tgt in enumerate(all_targets):
        if cancel_check is not None and cancel_check():
            break

        out_str = str(tgt.out_path)
        if skip_existing and not overwrite and tgt.out_path.exists():
            results[out_str] = "hit"
            if progress:
                print(f"[{idx + 1}/{total}] {tgt.out_path.name}  (hit, skip)", flush=True)
            # Still ensure the TOC -- catches the case where someone
            # deleted only the sidecar (or upgraded to a dnav-aware
            # workflow on an older mp4 cohort). Idempotent on hit.
            if build_toc:
                toc_status = _ensure_toc_sidecar(tgt.out_path)
                if progress and toc_status not in ("built", "hit"):
                    print(f"  toc: {toc_status}", flush=True)
            if progress_callback is not None:
                progress_callback(idx, total, tgt.out_path, "hit")
            continue

        if progress:
            print(f"[{idx + 1}/{total}] {tgt.out_path.name}", flush=True)

        encoded_ok = False
        try:
            _encode_one_panel(
                tgt.h5_path,
                tgt.img_id,
                tgt.out_path,
                codec=codec,
                lossless=lossless,
                crf=crf,
                preset=preset,
                fps=fps,
                normalize_orientation=normalize_orientation,
                crop=crop,
                overwrite=overwrite,
                progress=progress,
            )
            results[out_str] = "built"
            encoded_ok = True
        except Exception as e:  # noqa: BLE001
            results[out_str] = f"error: {e}"

        if encoded_ok and build_toc:
            toc_status = _ensure_toc_sidecar(tgt.out_path)
            if progress and toc_status not in ("built", "hit"):
                print(f"  toc: {toc_status}", flush=True)

        if progress_callback is not None:
            progress_callback(idx, total, tgt.out_path, results[out_str])

    return results
