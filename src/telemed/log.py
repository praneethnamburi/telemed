"""``Log`` -- entry point for analysis of an exported Telemed recording.

Loads the HDF5 sidecar produced by
:func:`immersionlab.telemed.export.extract_recording`. Mirrors the
``Log`` pattern used elsewhere in immersionlab (delsys, ot, atem,
etc.): construct with a single file path, get typed attributes for
the data + small methods that do the typical analysis / inspection
work directly on the instance.

Example::

    from immersionlab import telemed

    lf = telemed.Log("M:/data/054/telemed/scan.tvd.h5")
    print(lf.n_frames, lf.duration_s, lf.b_mode_roi)
    lf.view()                 # interactive frame browser
    img = lf.frame(0)         # uint8 H x W
    cropped = lf.frame(0, crop=True)  # uint8 roi_h x roi_w

Frame data is read lazily from the HDF5 (random-access; no need to
load 20k frames just to peek at one).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import h5py
import numpy as np


@dataclass(frozen=True)
class Roi:
    """B-mode region-of-interest, mirroring the export-side dataclass.

    Stored as immutable so callers can pass it around without
    accidental mutation. The slice helpers are convenient when
    indexing into a full-frame numpy array.

    ``img_id`` follows the AutoInt1 convention (1=B, 2=B2, 3=B3, 4=B4);
    ``physical_d{x,y}_cm_per_px`` is per-panel since multi-probe
    recordings can have different physical resolutions per transducer.
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

    def as_slice(self) -> tuple:
        """Return ``(y_slice, x_slice)`` for indexing a (H, W) array.

        Telemed's COM API uses 1-based pixel indexing; we convert to
        0-based Python slices here. End points are inclusive in the
        source convention, so the slice end gets +1.
        """
        return (slice(self.y1 - 1, self.y2), slice(self.x1 - 1, self.x2))


def _load_rois(attrs: dict) -> dict[int, Roi]:
    """Build ``{img_id: Roi}`` from HDF5 root attrs, handling v1-v3 and v4.

    v4 sidecars write per-img_id ``roi{N}_*`` + ``physical_d{x,y}{N}_cm_per_px``
    blocks plus an ``n_b_images`` count. v1-v3 wrote a single unprefixed
    block (``roi_x1``, ..., ``physical_dx_cm_per_px``, ...); we
    collapse those to ``{1: Roi(...)}`` so callers can always use the
    new multi-image API.
    """
    out: dict[int, Roi] = {}
    # Probe v4 blocks first (1..4 = B, B2, B3, B4).
    for img_id in (1, 2, 3, 4):
        key_x1 = f"roi{img_id}_x1"
        if key_x1 not in attrs:
            continue
        out[img_id] = Roi(
            img_id=img_id,
            x1=int(attrs[key_x1]),
            x2=int(attrs[f"roi{img_id}_x2"]),
            y1=int(attrs[f"roi{img_id}_y1"]),
            y2=int(attrs[f"roi{img_id}_y2"]),
            width=int(attrs[f"roi{img_id}_width"]),
            height=int(attrs[f"roi{img_id}_height"]),
            physical_dx_cm_per_px=float(attrs[f"physical_dx{img_id}_cm_per_px"]),
            physical_dy_cm_per_px=float(attrs[f"physical_dy{img_id}_cm_per_px"]),
        )
    if out:
        return out
    # Legacy fallback: v1-v3 single unprefixed block -> img_id=1.
    if "roi_x1" in attrs:
        out[1] = Roi(
            img_id=1,
            x1=int(attrs["roi_x1"]),
            x2=int(attrs["roi_x2"]),
            y1=int(attrs["roi_y1"]),
            y2=int(attrs["roi_y2"]),
            width=int(attrs["roi_width"]),
            height=int(attrs["roi_height"]),
            physical_dx_cm_per_px=float(attrs["physical_dx_cm_per_px"]),
            physical_dy_cm_per_px=float(attrs["physical_dy_cm_per_px"]),
        )
    return out


class Log:
    """Load a Telemed ``.tvd.h5`` sidecar.

    Args:
        fname: Path to the HDF5 sidecar (``<stem>.tvd.h5``) produced
            by :func:`~immersionlab.telemed.export.extract_recording`.

    Attributes:
        fname (Path): Full HDF5 path passed in.
        name (str): File stem (extensions stripped) for use as a
            recording identifier.
        n_frames (int): Number of frames in the recording.
        full_frame_width / full_frame_height (int): Pixel dims of the
            Echo Wave display frames stored in ``/frames/gray``.
        b_mode_rois (dict[int, Roi]): All active B-mode ROIs keyed by
            ``img_id`` (1=B, 2=B2, ...). Single-probe recordings get
            ``{1: Roi(...)}``; dual-probe (B+B2 side-by-side) gets
            ``{1: ..., 2: ...}``. Each Roi carries its own
            ``physical_d{x,y}_cm_per_px`` (per-panel calibration).
            v1-v3 sidecars collapse to ``{1: ...}``.
        n_b_images (int): Count of active B-mode panels (1 for single-
            probe, 2+ for multi-probe / multi-image).
        b_mode_roi (Roi): Backward-compat alias for ``b_mode_rois[1]``.
        physical_dx_cm_per_px / physical_dy_cm_per_px (float):
            Backward-compat aliases for the img_id=1 panel's per-axis
            spatial resolution.
        time_ms (np.ndarray): Absolute time of each frame in ms, with
            frame 0 -> 0.0. Shape ``(n_frames,)``.
        ifi_ms (np.ndarray): Inter-frame intervals in ms. ``ifi_ms[0]``
            is 0 (frame 1 anchor). Shape ``(n_frames,)``.
        source_tvd_path (str): Path the data was extracted from.
        extracted_at_iso (str): When the HDF5 was written.
        schema_version (str): HDF5 schema version. Pre-release alpha
            track: ``"v1a1"`` (initial single-ROI), ``"v1a2"`` (added
            ParamGet sweep), ``"v1a3"`` (expanded ParamGet to ~36
            fields), ``"v1a4"`` (per-img_id multi-ROI for dual-probe),
            ``"v1a5"`` (adds stored display-scale ``image_d{x,y}_cm_per_px``).
            Collapses to ``"v1"`` at public release. Legacy sidecars
            stored an int (1-4 = v1a1-v1a4); Log normalises both
            forms to a single string. See BENCHMARKING.md for the
            decision log of what each iteration added.
        params (dict[str, Any]): Per-recording acquisition parameters
            captured at export time via the AutoInt1 ParamGet*
            interface (schema v2+). Keys are short (no ``param_``
            prefix); use ``.get(name)`` since failed-probe params are
            absent. Common keys when populated: ``probe_name`` /
            ``probe_code``, ``beamformer_name`` / ``beamformer_code``,
            ``cine_end_datetime_str``, B-mode acquisition
            ``b_depth`` / ``b_frequency`` / ``b_gain`` / ``b_power``
            / ``b_dynamic_range`` / ``b_focus_depth`` /
            ``b_focuses_count`` / ``b_is_dynamic_focus`` / ``b_thi`` /
            ``b_frame_averaging`` / ``b_rejection`` /
            ``b_image_enhancement{,_method}`` /
            ``b_speckle_reduction{,_level}`` / ``b_palette`` /
            ``b_palette_{gamma,brightness,contrast,negative}``,
            geometry / orientation
            ``b_is_scan_direction_changed`` (L/R flip) / ``b_rotate`` /
            ``b_view_area`` / ``b_scan_type`` /
            ``b_steering_trapezoid_angle`` / ``b_lines_density`` /
            ``b_zoom_factor``, sanity ``is_usg_file_opened`` /
            ``scanning_state`` / ``is_probe_active``. Empty ``{}`` on
            schema v1 sidecars.

    Notes:
        Frame data is loaded lazily (random-access via h5py). The
        timing and metadata arrays ARE eagerly loaded on construction
        because they're tiny.
    """

    def __init__(self, fname: Union[str, os.PathLike]):
        self.fname: Path = Path(fname)
        if not self.fname.is_file():
            raise FileNotFoundError(self.fname)
        self.name: str = self.fname.name.split(".")[0]

        with h5py.File(self.fname, "r") as h5:
            a = dict(h5.attrs)
            self.n_frames: int = int(a["n_frames"])
            self.full_frame_width: int = int(a["full_frame_width"])
            self.full_frame_height: int = int(a["full_frame_height"])
            self.source_tvd_path: str = str(a["source_tvd_path"])
            self.extracted_at_iso: str = str(a["extracted_at_iso"])
            # Schema version is a string in v1a5+ (e.g. "v1a5", will
            # collapse to "v1" at public release). Legacy sidecars
            # store an int (1=v1a1, 2=v1a2, 3=v1a3, 4=v1a4); normalise
            # both forms to a single string. Bytes -> decoded; ints
            # -> "v1aN".
            raw_v = a["schema_version"]
            if isinstance(raw_v, bytes):
                raw_v = raw_v.decode("utf-8", errors="replace")
            if isinstance(raw_v, (int, np.integer)):
                self.schema_version: str = f"v1a{int(raw_v)}"
            else:
                self.schema_version = str(raw_v)

            # Display scale (v1a5+). Optional -- legacy sidecars
            # (v1a1..v1a4) don't have these stored; Log's property
            # accessors fall back to computing from b_depth +
            # panel_height_px.
            self._stored_image_dx: Optional[float] = (
                float(a["image_dx_cm_per_px"])
                if "image_dx_cm_per_px" in a else None
            )
            self._stored_image_dy: Optional[float] = (
                float(a["image_dy_cm_per_px"])
                if "image_dy_cm_per_px" in a else None
            )

            # ROI block: v4+ writes per-img_id ``roi{N}_*`` blocks +
            # ``n_b_images``; v1-v3 wrote a single unprefixed ``roi_*``
            # block + flat ``physical_d{x,y}_cm_per_px``. Read both;
            # legacy collapses to {1: ...}.
            self.b_mode_rois: dict[int, Roi] = _load_rois(a)
            self.n_b_images: int = len(self.b_mode_rois)

            # Schema v2: opportunistic ParamGet snapshot under
            # param_* attrs. Strip the prefix for ergonomic access.
            # h5py returns numpy scalars + bytes for HDF5-native types;
            # coerce to plain Python so .params behaves like a normal
            # dict when serialised / printed.
            self.params: dict[str, Any] = {}
            for k, v in a.items():
                if not k.startswith("param_"):
                    continue
                if isinstance(v, bytes):
                    v = v.decode("utf-8", errors="replace")
                elif isinstance(v, np.generic):
                    v = v.item()
                self.params[k[len("param_"):]] = v

            self.time_ms: np.ndarray = h5["timing/time_ms"][...]
            self.ifi_ms: np.ndarray = h5["timing/ifi_ms"][...]

            # /frames/gray is optional (export with frames=False omits
            # it). Cache a flag so view() can fail with a clear message.
            self._has_frames: bool = "frames/gray" in h5

    # ---------- Convenience scalar properties ----------

    @property
    def duration_s(self) -> float:
        """Recording duration in seconds (last frame's absolute time)."""
        return float(self.time_ms[-1] / 1000.0)

    @property
    def mean_fps(self) -> float:
        """Implied average fps from the recording duration."""
        if self.duration_s == 0:
            return 0.0
        return float((self.n_frames - 1) / self.duration_s)

    @property
    def has_frames(self) -> bool:
        """True if the HDF5 has pixel data (i.e. wasn't extracted with frames=False)."""
        return self._has_frames

    # ---------- Back-compat aliases for the v3 single-ROI surface ----------

    @property
    def b_mode_roi(self) -> Roi:
        """Primary B-mode ROI (img_id=1). Alias for ``b_mode_rois[1]``."""
        return self.b_mode_rois[1]

    @property
    def physical_dx_cm_per_px(self) -> float:
        """Per-pixel horizontal cm of the img_id=1 panel.

        This is the **beamformer-native** spacing reported by
        AutoInt1's ``GetUltrasoundPhysicalDeltaX``. It does NOT match
        the on-screen display scale (the two differ by ~2% on typical
        Telemed acquisitions). For pixel-to-cm conversion in tracked-
        point analysis, prefer :attr:`image_dx_cm_per_px`.
        """
        return self.b_mode_rois[1].physical_dx_cm_per_px

    @property
    def physical_dy_cm_per_px(self) -> float:
        """Per-pixel vertical cm of the img_id=1 panel.

        Beamformer-native spacing -- see notes on
        :attr:`physical_dx_cm_per_px`. For measurements, use
        :attr:`image_dy_cm_per_px`.
        """
        return self.b_mode_rois[1].physical_dy_cm_per_px

    # ---------- Display (image) scale ----------

    @property
    def image_dy_cm_per_px(self) -> Optional[float]:
        """**Display** vertical scale -- cm per panel pixel.

        Schema v1a5+ stores this as a root attr (computed at extract
        time as ``b_depth_mm / 10 / panel_height_px`` per Telemed
        support's "trust the depth setting" calibration). Legacy
        sidecars (v1a1..v1a4) lack the stored value; this property
        derives it on the fly when possible. Returns ``None`` when
        the sidecar has no ``b_depth`` param either (v1a1 schemas).

        Use this for cm conversions on tracked-point coordinates --
        ``physical_dy{N}_cm_per_px`` is the beamformer-native scale,
        kept for hardware provenance but ~2% off the display scale
        on typical Telemed acquisitions.
        """
        if self._stored_image_dy is not None:
            return self._stored_image_dy
        depth_mm = self.params.get("b_depth")
        if depth_mm is None:
            return None
        return (float(depth_mm) / 10.0) / float(self.b_mode_rois[1].height)

    @property
    def image_dx_cm_per_px(self) -> Optional[float]:
        """**Display** horizontal scale -- cm per panel pixel.

        Telemed renders square display pixels (1:1 aspect, so anatomy
        isn't squished), and AutoInt1 reports ``physical_dx ==
        physical_dy`` for all probed acquisitions. So the display x
        scale equals :attr:`image_dy_cm_per_px`. If a future probe is
        found to break this assumption it would surface as anatomy
        rendered with a non-1:1 aspect in :meth:`view`; revisit then.

        Stored as a root attr in v1a5+; back-compat fallback via
        :attr:`image_dy_cm_per_px` for legacy sidecars.
        """
        if self._stored_image_dx is not None:
            return self._stored_image_dx
        return self.image_dy_cm_per_px

    # ---------- Frame access ----------

    def frame(self, frame_idx_0n: int, *, crop: bool = False) -> np.ndarray:
        """Read a single frame as uint8.

        Args:
            frame_idx_0n: 0-indexed frame number.
            crop: If True, return only the B-mode ROI region; if False
                (default), return the full Echo Wave display frame.

        Returns:
            ``np.ndarray`` of shape ``(H, W)`` -- full frame or
            cropped depending on ``crop``.

        Raises:
            RuntimeError: If the HDF5 was written without frames
                (``extract_recording(..., frames=False)``).
            IndexError: If ``frame_idx_0n`` is out of range.
        """
        if not self._has_frames:
            raise RuntimeError(
                f"{self.fname.name} contains no frame data "
                "(extracted with frames=False). Re-extract with "
                "frames=True to enable frame access."
            )
        if not (0 <= frame_idx_0n < self.n_frames):
            raise IndexError(
                f"frame_idx_0n {frame_idx_0n} out of range "
                f"[0, {self.n_frames})"
            )
        # Re-open the HDF5 per call to keep the file handle short-lived
        # (avoids issues if the file lives on a network drive).
        with h5py.File(self.fname, "r") as h5:
            full = h5["frames/gray"][frame_idx_0n]
        if crop:
            ys, xs = self.b_mode_roi.as_slice()
            return full[ys, xs]
        return full

    # ---------- Encode to mp4 (single-recording convenience) ----------

    def to_video(
        self,
        out_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ) -> dict:
        """Encode this recording as mp4 file(s).

        Thin convenience wrapper around
        :func:`immersionlab.telemed.export_video` operating on this
        sidecar's path. Single-probe -> ``<stem>.mp4``; dual-probe ->
        ``<stem>_b{img_id}.mp4`` per active panel. Lossless h265 mono
        by default; orientation normalised to canonical.

        Args:
            out_dir: Output directory. ``None`` (default) co-locates the
                mp4(s) next to ``self.fname``.
            **kwargs: Forwarded to ``export_video`` -- ``lossless``,
                ``crf``, ``preset``, ``fps``, ``normalize_orientation``,
                ``overwrite``, etc. See ``export_video`` docstring.

        Returns:
            ``{mp4_path_str: status}`` (one entry per panel encoded).
        """
        # Lazy import (avoid a top-of-module cycle log <-> _encode).
        from ._encode import export_video

        return export_video(self.fname, out_dir=out_dir, **kwargs)

    # ---------- View ----------

    def view(self, *, crop: bool = True, frame_idx_0n: int = 0):
        """Interactive frame browser using matplotlib.

        Opens a window with the current frame + a slider for scrubbing
        and left/right arrow-key bindings for single-frame steps.
        Returns the matplotlib ``Figure`` so the caller can keep a
        reference (or call ``plt.show()`` afterwards in non-interactive
        backends).

        Args:
            crop: If True (default), show the B-mode ROI only. If
                False, show the full Echo Wave display frame.
            frame_idx_0n: Initial frame to display.
        """
        if not self._has_frames:
            raise RuntimeError(
                f"{self.fname.name} contains no frame data "
                "(extracted with frames=False). Cannot view."
            )
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        fig, (ax_img, ax_slider) = plt.subplots(
            nrows=2,
            gridspec_kw={"height_ratios": [20, 1]},
            figsize=(10, 7),
        )
        fig.canvas.manager.set_window_title(f"telemed.Log: {self.name}")

        img0 = self.frame(frame_idx_0n, crop=crop)
        im = ax_img.imshow(img0, cmap="gray", vmin=0, vmax=255,
                           interpolation="nearest")
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # Scale bar (lower-left). Uses ``image_dy_cm_per_px`` (the
        # display scale, which is what the operator's depth ruler is
        # calibrated against). Bar is 10 mm long; we adapt to 5 mm /
        # 20 mm if 10 mm is too small / large to read. Silently skip
        # if scale isn't derivable (v1 sidecars with no b_depth).
        dy = self.image_dy_cm_per_px
        if dy and dy > 0:
            img_h, img_w = img0.shape[:2]
            for bar_mm in (10, 5, 20, 2):
                bar_px = int(round(bar_mm * 0.1 / dy))
                if 30 <= bar_px <= img_w * 0.5:
                    break
            else:
                bar_mm, bar_px = 10, int(round(10 * 0.1 / dy))
            # Inset from the corner -- 5% of width / height of img.
            x0 = int(0.04 * img_w)
            y0 = int(0.96 * img_h)
            ax_img.plot(
                [x0, x0 + bar_px], [y0, y0],
                color="yellow", linewidth=2, solid_capstyle="butt",
            )
            ax_img.text(
                x0 + bar_px / 2, y0 - 0.02 * img_h, f"{bar_mm} mm",
                color="yellow", fontsize=9, ha="center", va="bottom",
            )

        def _title_for(i: int) -> str:
            return (
                f"frame {i + 1}/{self.n_frames}  "
                f"t = {self.time_ms[i] / 1000.0:.3f} s  "
                f"(IFI = {self.ifi_ms[i]:.3f} ms)"
            )

        ax_img.set_title(_title_for(frame_idx_0n))

        slider = Slider(
            ax=ax_slider,
            label="frame",
            valmin=0,
            valmax=max(self.n_frames - 1, 1),
            valinit=frame_idx_0n,
            valstep=1,
            valfmt="%d",
        )

        # Holder for the most recently displayed frame index so the
        # arrow-key handler can mutate it (and the slider's on_changed
        # can write it).
        state = {"idx": int(frame_idx_0n)}

        def _show_frame(i: int):
            i = int(np.clip(i, 0, self.n_frames - 1))
            state["idx"] = i
            im.set_data(self.frame(i, crop=crop))
            ax_img.set_title(_title_for(i))
            fig.canvas.draw_idle()

        def _on_slider(val):
            i = int(val)
            if i != state["idx"]:
                _show_frame(i)

        def _on_key(event):
            if event.key == "right":
                slider.set_val(min(state["idx"] + 1, self.n_frames - 1))
            elif event.key == "left":
                slider.set_val(max(state["idx"] - 1, 0))
            elif event.key in ("pagedown", "down"):
                slider.set_val(min(state["idx"] + 10, self.n_frames - 1))
            elif event.key in ("pageup", "up"):
                slider.set_val(max(state["idx"] - 10, 0))
            elif event.key == "home":
                slider.set_val(0)
            elif event.key == "end":
                slider.set_val(self.n_frames - 1)

        slider.on_changed(_on_slider)
        fig.canvas.mpl_connect("key_press_event", _on_key)

        # Keep refs alive on the figure so they don't get GC'd before
        # the user interacts.
        fig._telemed_view_slider = slider  # type: ignore[attr-defined]
        fig._telemed_view_state = state  # type: ignore[attr-defined]

        plt.tight_layout()
        return fig

    # ---------- Repr / debug ----------

    def __repr__(self) -> str:
        return (
            f"telemed.Log(name={self.name!r}, n_frames={self.n_frames}, "
            f"duration={self.duration_s:.3f}s, "
            f"roi={self.b_mode_roi.width}x{self.b_mode_roi.height})"
        )
