"""``Log`` -- entry point for analysis of an exported Telemed recording.

Loads the HDF5 sidecar produced by :func:`telemed.export_h5`.
Construct with a single file path, get typed attributes for the data
plus small methods that do the typical analysis / inspection work
directly on the instance.

Example::

    import telemed

    lf = telemed.Log("M:/data/054/telemed/scan.tvd.h5")
    print(lf.n_frames, lf.duration_s, lf.b_mode_roi)
    lf.view()                 # interactive browser (full frame, all probes)
    lf.view("right")          # just the right-hand probe (dual-probe scans)
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

_UNSET = object()  # sentinel: distinguishes "not yet computed" from a cached None (no .tvd)


@dataclass(frozen=True)
class Roi:
    """B-mode region-of-interest, mirroring the export-side dataclass.

    Stored as immutable so callers can pass it around without
    accidental mutation. The slice helpers are convenient when
    indexing into a full-frame numpy array.

    ``img_id`` follows the AutoInt1 convention (1=B, 2=B2, 3=B3, 4=B4);
    ``physical_d{x,y}_cm_per_px`` is per-panel since multi-probe
    recordings can have different physical resolutions per transducer.

    The Roi describes the **outer B-mode panel** from AutoInt1
    (depth ruler + side margins + inner image). The inner-ultrasound-
    image sub-rectangle (depth ruler / margins / tick row stripped)
    is computed on demand from frame pixels -- see
    :meth:`Log.image_roi`.
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
        """Return ``(y_slice, x_slice)`` for the **outer panel ROI**.

        Telemed's COM API uses 1-based pixel indexing; we convert to
        0-based Python slices here. End points are inclusive in the
        source convention, so the slice end gets +1.
        """
        return (slice(self.y1 - 1, self.y2), slice(self.x1 - 1, self.x2))


def _load_rois(attrs: dict) -> dict[int, Roi]:
    """Build ``{img_id: Roi}`` from HDF5 root attrs, handling v1-v3 and v4+.

    v4+ sidecars write per-img_id ``roi{N}_*`` +
    ``physical_d{x,y}{N}_cm_per_px`` blocks plus an ``n_b_images``
    count. v1-v3 wrote a single unprefixed block (``roi_x1``, ...,
    ``physical_dx_cm_per_px``, ...); we collapse those to
    ``{1: Roi(...)}`` so callers can always use the new multi-image
    API.
    """
    out: dict[int, Roi] = {}
    # Probe v4+ blocks first (1..4 = B, B2, B3, B4).
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
            by :func:`~telemed.export.extract_recording`.

    Attributes:
        fname (Path): Full HDF5 path passed in.
        name (str): File stem (extensions stripped) for use as a
            recording identifier.
        n_frames (int): Number of frames stored in this sidecar (==
            :attr:`n_frames_stored`).
        n_frames_stored (int): Frames EchoWave stored here (alias of
            :attr:`n_frames`).
        n_frames_declared (Optional[int]): Frames the device declared in
            the ``.tvd`` header (``strh`` count); from the stored attr
            when present (no ``.tvd`` read), else from the sibling
            ``.tvd``. ``None`` if neither is available.
        time_ms_declared / time_ms_comfree (Optional[np.ndarray]): COM-
            free per-frame ``time_ms`` for every *declared* frame, read
            from the sibling ``.tvd`` (cached ticks sidecar). The
            declared-frame superset of :attr:`time_ms` (the stored
            subset). ``None`` when the ``.tvd``/sidecar is absent.
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
        tvd_declared_n_frames (Optional[int]): Frame count recorded in
            the source ``.tvd`` container header, stored at extract
            time. ``None`` on sidecars extracted before the
            completeness-QC feature. When this sits well above
            :attr:`n_frames`, EchoWave truncated the load to fit memory
            -- audit a cohort with ``telemed.verify_complete``.
        schema_version (str): HDF5 schema version. Always reports
            ``"v1"`` -- production extracts write that label and Log
            also normalises the legacy in-development variants
            (``"v1a1"`` through ``"v1a5"``, or integers ``1..4``) to
            ``"v1"`` on load. The historical labels reflect what each
            iteration added (single-ROI -> ParamGet sweep -> expanded
            ParamGet -> per-img_id multi-ROI -> stored display-scale)
            and are documented in the changelog; downstream code should
            branch on the data, not the label.
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
            # Recorded frame count from the .tvd container header, stored
            # at extract time (absent on sidecars predating the
            # completeness-QC feature). When present and well above
            # n_frames, EchoWave truncated the load -- audit with
            # telemed.verify_complete(). None when not stored.
            self.tvd_declared_n_frames: Optional[int] = (
                int(a["tvd_declared_n_frames"]) if "tvd_declared_n_frames" in a else None
            )
            # Production extracts write the public ``"v1"`` baseline.
            # Legacy in-development sidecars carry either the alpha-
            # track strings (``"v1a{1..5}"``) or an integer
            # (``1..4`` = v1a1..v1a4); normalise all of them to ``"v1"``
            # so downstream code branches on the data, not the label.
            raw_v = a["schema_version"]
            if isinstance(raw_v, bytes):
                raw_v = raw_v.decode("utf-8", errors="replace")
            if isinstance(raw_v, (int, np.integer)):
                self.schema_version: str = "v1"
            elif isinstance(raw_v, str) and (raw_v == "v1" or raw_v.startswith("v1a")):
                self.schema_version = "v1"
            else:
                self.schema_version = str(raw_v)

            # Display scale (v1a5+). Optional -- legacy sidecars
            # (v1a1..v1a4) don't have these stored; Log's property
            # accessors fall back to computing from b_depth +
            # panel_height_px.
            self._stored_image_dx: Optional[float] = (
                float(a["image_dx_cm_per_px"]) if "image_dx_cm_per_px" in a else None
            )
            self._stored_image_dy: Optional[float] = (
                float(a["image_dy_cm_per_px"]) if "image_dy_cm_per_px" in a else None
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
                self.params[k[len("param_") :]] = v

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

    # ---------- Stored vs declared frame counts + COM-free timing ----------

    @property
    def n_frames_stored(self) -> int:
        """Frames EchoWave loaded/stored into this sidecar -- the same number as :attr:`n_frames`
        (the ``.h5`` is the *stored* subset). Named for symmetry with :attr:`n_frames_declared`."""
        return self.n_frames

    @property
    def time_ms_stored(self) -> np.ndarray:
        """The EchoWave-*stored* per-frame ``time_ms`` (alias of :attr:`time_ms`); named for symmetry
        with :attr:`time_ms_declared`."""
        return self.time_ms

    @property
    def n_frames_declared(self) -> Optional[int]:
        """Frames the device DECLARED in the ``.tvd`` container header (the ``strh`` count).

        Sourced from the stored ``tvd_declared_n_frames`` root attr when present -- so **no ``.tvd``
        read is needed**; otherwise read from the sibling ``.tvd`` (or its ticks sidecar). ``None``
        if neither is available. Runs ~2 frames above :attr:`n_frames_stored` on complete recordings
        (declared = pulsed+1 = stored+2); *well* above means a memory-truncated load
        (see :func:`telemed.verify_complete`)."""
        if self.tvd_declared_n_frames is not None:
            return self.tvd_declared_n_frames
        tvd = self._sibling_tvd
        if tvd is not None and tvd.is_file():
            from ._extract import read_tvd_n_frames
            return read_tvd_n_frames(tvd)
        tm = self.time_ms_declared
        return None if tm is None else int(len(tm))

    @property
    def _sibling_tvd(self) -> Optional[Path]:
        """The source ``.tvd`` sitting next to this ``<stem>.tvd.h5`` sidecar (``None`` if the name
        isn't the composite ``.tvd.h5`` form). Used for the COM-free declared timing."""
        s = str(self.fname)
        return Path(s[:-3]) if s.endswith(".tvd.h5") else None

    @property
    def time_ms_declared(self) -> Optional[np.ndarray]:
        """COM-free per-frame ``time_ms`` for every **declared** frame, read straight from the
        sibling ``.tvd`` (via its cached ``.tvd.ticks.npy`` sidecar; see
        :func:`telemed.read_tvd_time_ms`). This is the declared-frame *superset* of :attr:`time_ms`
        (which is the EchoWave-*stored* subset). ``None`` when the ``.tvd``/sidecar isn't available.

        Lazy + cached on the instance, and the first read populates the ticks sidecar, so repeat
        access (this session or a later one) is a fast ``.npy`` load rather than a container walk."""
        cached = getattr(self, "_time_ms_declared_cache", _UNSET)
        if cached is not _UNSET:
            return cached
        tvd = self._sibling_tvd
        val = None
        if tvd is not None:
            from ._extract import read_tvd_time_ms
            try:
                val = read_tvd_time_ms(tvd, cache=True)
            except Exception:  # noqa: BLE001 -- never let a cache/read hiccup break attribute access
                val = None
        object.__setattr__(self, "_time_ms_declared_cache", val)
        return val

    @property
    def time_ms_comfree(self) -> Optional[np.ndarray]:
        """Alias for :attr:`time_ms_declared` (the COM-free, read-from-the-``.tvd`` declared timing)."""
        return self.time_ms_declared

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

    def frame(
        self,
        frame_idx_0n: int,
        *,
        crop: Union[bool, str] = False,
        panel: int = 1,
    ) -> np.ndarray:
        """Read a single frame as uint8.

        Args:
            frame_idx_0n: 0-indexed frame number.
            crop: ``False`` (default) -> full Echo Wave display frame.
                ``True`` or ``"image"`` -> the inner ultrasound image
                (detected from frame pixels; depth ruler / side
                margins / bottom-tick row stripped). Falls back to the
                outer panel ROI when the detector can't identify the
                inner box. ``"panel"`` -> the outer B-mode panel ROI
                (depth ruler + margins included).
            panel: ``img_id`` of the panel to crop to (1=B, 2=B2, ...).
                Defaults to 1 so single-probe call sites work
                unchanged. Validated even when ``crop=False`` so a
                typo doesn't silently pass.

        Returns:
            ``np.ndarray`` of shape ``(H, W)`` -- full frame or
            cropped depending on ``crop``.

        Raises:
            RuntimeError: If the HDF5 was written without frames
                (``extract_recording(..., frames=False)``).
            IndexError: If ``frame_idx_0n`` is out of range.
            KeyError: If ``panel`` is not an active img_id.
            ValueError: If ``crop`` is not bool, ``"image"``, or
                ``"panel"``.
        """
        if not self._has_frames:
            raise RuntimeError(
                f"{self.fname.name} contains no frame data "
                "(extracted with frames=False). Re-extract with "
                "frames=True to enable frame access."
            )
        if not (0 <= frame_idx_0n < self.n_frames):
            raise IndexError(f"frame_idx_0n {frame_idx_0n} out of range " f"[0, {self.n_frames})")
        if panel not in self.b_mode_rois:
            raise KeyError(
                f"panel={panel} not in this recording; active img_ids: "
                f"{sorted(self.b_mode_rois)}"
            )
        # Re-open the HDF5 per call to keep the file handle short-lived
        # (avoids issues if the file lives on a network drive).
        with h5py.File(self.fname, "r") as h5:
            full = h5["frames/gray"][frame_idx_0n]
        if crop is False:
            return full
        roi = self.b_mode_rois[panel]
        if crop == "panel":
            ys, xs = roi.as_slice()
        elif crop is True or crop == "image":
            ys, xs = self.image_slice(panel)
        else:
            raise ValueError(f"crop={crop!r} not understood; use False/True/'image'/'panel'.")
        return full[ys, xs]

    def image_slice(self, panel: int = 1) -> tuple:
        """``(y_slice, x_slice)`` for the **inner ultrasound image** panel.

        Runs the content-based detector on a multi-frame mean of the
        panel and caches the resulting bounds on this Log instance.
        Returns the outer panel slice when the detector can't
        identify the inner box (warns once per panel).
        """
        cache = getattr(self, "_image_slice_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_image_slice_cache", cache)
        if panel in cache:
            return cache[panel]
        if panel not in self.b_mode_rois:
            raise KeyError(
                f"panel={panel} not in this recording; active img_ids: "
                f"{sorted(self.b_mode_rois)}"
            )
        roi = self.b_mode_rois[panel]
        # Lazy import: _encode owns the detector, but log.py is
        # imported by _encode (via from .log import Log). Defer to
        # call time to avoid the cycle.
        from ._encode import _aggregate_panel_from_h5, _detect_image_roi

        if not self._has_frames:
            cache[panel] = roi.as_slice()
            return cache[panel]
        panel_mean = _aggregate_panel_from_h5(self.fname, roi)
        inner = _detect_image_roi(panel_mean)
        if inner is None:
            import warnings

            warnings.warn(
                f"telemed.Log.image_slice: detector couldn't identify "
                f"inner ultrasound image for {self.fname.name} "
                f"img_id={panel}; using the outer panel ROI.",
                stacklevel=2,
            )
            cache[panel] = roi.as_slice()
            return cache[panel]
        x_s, x_e, y_s, y_e = inner
        # Panel-local 0-based half-open -> full-frame slices.
        cache[panel] = (
            slice(roi.y1 - 1 + y_s, roi.y1 - 1 + y_e),
            slice(roi.x1 - 1 + x_s, roi.x1 - 1 + x_e),
        )
        return cache[panel]

    # ---------- Panel selection (for view) ----------

    def _panels_left_to_right(self) -> list:
        """Active ``Roi``s ordered by on-screen x-position (left first).

        ``img_id`` order (1=B, 2=B2, ...) is *not* guaranteed to run
        left-to-right on screen, so anything that talks about "left" /
        "right" probes resolves position from ``Roi.x1`` instead.
        """
        return sorted(self.b_mode_rois.values(), key=lambda r: r.x1)

    def _panel_position_label(self, img_id: int) -> str:
        """Human label for a panel's screen position ("left"/"right"/"").

        Empty string for single-probe recordings (no left/right to
        disambiguate); ``"#N"`` for the interior panels of a 3+-probe
        layout.
        """
        ordered = self._panels_left_to_right()
        if len(ordered) == 1:
            return ""
        for i, r in enumerate(ordered):
            if r.img_id == img_id:
                if i == 0:
                    return "left"
                if i == len(ordered) - 1:
                    return "right"
                return f"#{i + 1}"
        return ""

    def _panel_help(self) -> str:
        """One-line description of the active probes for error messages."""
        ordered = self._panels_left_to_right()
        if len(ordered) == 1:
            return f"This recording has a single probe (img_id={ordered[0].img_id})."
        bits = ", ".join(
            f"{r.img_id} ({self._panel_position_label(r.img_id)})" for r in ordered
        )
        return f"Active probes (left to right): {bits}."

    def _resolve_panel(self, panel: Union[int, str, None]) -> Optional[int]:
        """Map a :meth:`view` ``panel`` selector to an ``img_id``.

        Returns ``None`` for the whole-frame views (``None`` / ``"all"``)
        and a concrete ``img_id`` for a single probe. ``"left"`` /
        ``"right"`` resolve by screen x-position (see
        :meth:`_panels_left_to_right`).
        """
        if panel is None:
            return None
        # bool is an int subclass; reject it so a leftover ``crop=True`` /
        # ``crop=False`` habit fails loudly instead of silently selecting
        # img_id 1 (True) or being treated as a missing panel (False).
        if isinstance(panel, bool):
            raise TypeError(
                f"panel={panel!r}: pass None/'all', an img_id int, or "
                "'left'/'right' (the crop= argument was removed)."
            )
        if isinstance(panel, str):
            key = panel.strip().lower()
            if key == "all":
                return None
            if key in ("left", "right"):
                ordered = self._panels_left_to_right()
                return (ordered[0] if key == "left" else ordered[-1]).img_id
            raise ValueError(
                f"panel={panel!r} not understood; use None/'all', "
                f"'left'/'right', or an img_id int. {self._panel_help()}"
            )
        if isinstance(panel, (int, np.integer)):
            img_id = int(panel)
            if img_id in self.b_mode_rois:
                return img_id
            raise KeyError(
                f"panel={panel} is not an active probe. {self._panel_help()}"
            )
        raise TypeError(
            f"panel must be None/'all', an img_id int, or 'left'/'right'; "
            f"got {type(panel).__name__}."
        )

    # ---------- Encode to mp4 (single-recording convenience) ----------

    def mp4_path(
        self,
        panel: int = 1,
        *,
        out_dir: Optional[Union[str, os.PathLike]] = None,
    ) -> Path:
        """Where the per-panel mp4 would land for this recording.

        Deterministic from ``n_b_images`` + the chosen ``out_dir``.
        Mirrors ``export_video``'s naming so downstream callers don't
        recreate the convention:

        * single-probe -> ``<stem>.mp4``
        * multi-probe  -> ``<stem>_b{panel}.mp4``

        Does NOT check whether the file exists -- pair with
        :meth:`ensure_mp4` to encode if missing.

        Args:
            panel: ``img_id`` of the panel (1=B, 2=B2, ...). Must be
                an active panel in :attr:`b_mode_rois`.
            out_dir: Output directory. ``None`` (default) co-locates
                next to ``self.fname`` -- matches :meth:`to_video`'s
                default.

        Raises:
            KeyError: ``panel`` is not an active img_id.
        """
        if panel not in self.b_mode_rois:
            raise KeyError(
                f"panel={panel} not in this recording; active img_ids: "
                f"{sorted(self.b_mode_rois)}"
            )
        # Strip the composite ``.tvd.h5`` -- matches ``_stem_from_h5``
        # in ``_encode.py``; ``Path.stem`` alone would leave ``.tvd``.
        name = self.fname.name
        stem = name[: -len(".tvd.h5")] if name.endswith(".tvd.h5") else self.fname.stem
        base_dir = Path(out_dir) if out_dir is not None else self.fname.parent
        if self.n_b_images == 1:
            return base_dir / f"{stem}.mp4"
        return base_dir / f"{stem}_b{panel}.mp4"

    def ensure_mp4(
        self,
        panel: int = 1,
        *,
        out_dir: Optional[Union[str, os.PathLike]] = None,
        **encode_kwargs,
    ) -> Path:
        """Return the per-panel mp4 path, encoding it if missing.

        Idempotent: a second call is a file-existence check, not a
        re-encode. The encode pass writes every active panel of the
        recording in one HDF5-open pass, so ``ensure_mp4(1)`` on a
        dual-probe recording also produces panel 2's mp4 as a side
        effect (and vice versa).

        Args:
            panel: ``img_id`` of the panel to return.
            out_dir: Output directory. ``None`` (default) co-locates
                with ``self.fname``. Forwarded to both the existence
                check and the encode pass so they agree on where to
                look.
            **encode_kwargs: Forwarded to ``export_video`` --
                ``lossless`` / ``crf`` / ``preset`` / ``fps`` /
                ``normalize_orientation`` / ``overwrite`` etc. See
                ``export_video``.

        Returns:
            Path to the per-panel mp4; guaranteed to exist on return.
        """
        target = self.mp4_path(panel, out_dir=out_dir)
        if not target.exists():
            self.to_video(out_dir=out_dir, **encode_kwargs)
        return target

    def to_video(
        self,
        out_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ) -> dict:
        """Encode this recording as mp4 file(s).

        Thin convenience wrapper around
        :func:`telemed.export_video` operating on this
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

    def view(self, panel: Union[int, str, None] = None, *, frame_idx_0n: int = 0, block: bool = False):
        """Interactive frame browser using matplotlib.

        Opens a window with the current frame + a slider for scrubbing
        and left/right arrow-key bindings for single-frame steps, then
        shows it (non-blocking by default) and returns the matplotlib
        ``Figure`` so the caller can keep a reference.

        Args:
            panel: Which probe to show.

                * ``None`` (default) or ``"all"`` -- the full Echo Wave
                  frame, i.e. *both* panels of a dual-probe (B + B2)
                  recording side by side.
                * an ``img_id`` int (``1``, ``2``, ...) -- that probe's
                  inner ultrasound image.
                * ``"left"`` / ``"right"`` -- the leftmost / rightmost
                  probe, resolved by on-screen position (not ``img_id``
                  order). On a single-probe recording both map to the
                  one probe.

                A selected probe is shown as its inner image (depth
                ruler / margins stripped); the scale bar conveys depth.
                For the outer panel ROI including the ruler, use
                ``frame(..., crop="panel")``.
            frame_idx_0n: Initial frame to display.
            block: If ``False`` (default), enable matplotlib interactive
                mode and show the window non-blocking, returning
                immediately -- the slider and arrow keys stay live as long
                as a REPL event loop is running (the usual case). If
                ``True``, run the GUI main loop until the window is closed
                (use this from non-interactive scripts).

        Raises:
            ValueError / KeyError / TypeError: If ``panel`` isn't a valid
                selector or names a probe this recording doesn't have.
        """
        if not self._has_frames:
            raise RuntimeError(
                f"{self.fname.name} contains no frame data "
                "(extracted with frames=False). Cannot view."
            )
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        # Resolve the probe selector once. ``img_id is None`` => show the
        # whole frame (crop=False); otherwise crop that panel to its
        # inner image. ``frame()`` ignores ``panel`` when ``crop=False``,
        # so the ``or 1`` placeholder is harmless in the whole-frame case.
        img_id = self._resolve_panel(panel)
        crop_mode: Union[bool, str] = False if img_id is None else "image"
        frame_panel = img_id or 1

        fig, (ax_img, ax_slider) = plt.subplots(
            nrows=2,
            gridspec_kw={"height_ratios": [20, 1]},
            figsize=(10, 7),
        )
        if img_id is None:
            view_label = "full frame"
        else:
            pos = self._panel_position_label(img_id)
            view_label = f"probe {img_id} ({pos})" if pos else f"probe {img_id}"
        fig.canvas.manager.set_window_title(f"telemed.Log: {self.name} -- {view_label}")

        img0 = self.frame(frame_idx_0n, crop=crop_mode, panel=frame_panel)
        im = ax_img.imshow(img0, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
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
                [x0, x0 + bar_px],
                [y0, y0],
                color="yellow",
                linewidth=2,
                solid_capstyle="butt",
            )
            ax_img.text(
                x0 + bar_px / 2,
                y0 - 0.02 * img_h,
                f"{bar_mm} mm",
                color="yellow",
                fontsize=9,
                ha="center",
                va="bottom",
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
            im.set_data(self.frame(i, crop=crop_mode, panel=frame_panel))
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

        # Show the window ourselves so ``lf.view()`` just works. A bare
        # ``plt.show(block=False)`` only *displays* the figure -- per its
        # own contract "you are responsible for ensuring that the event
        # loop is running to have responsive figures." With interactive
        # mode off and no REPL integration, Qt never processes the
        # slider/key events, so the window looks frozen on the initial
        # frame. ``plt.ion()`` installs matplotlib's REPL/event-loop hook
        # (the Qt input hook in a plain shell, IPython's ``enable_gui``
        # under IPython), which is what actually keeps the slider live.
        # Skip all of this on non-GUI backends (e.g. Agg under pytest /
        # in a headless script): there's nothing to show, and we don't
        # want to flip global interactive mode as a side effect.
        import matplotlib

        # Enumerate the interactive (GUI) backends. ``rcsetup.interactive_bk``
        # was deprecated in matplotlib 3.9 and removed in 3.11 in favour of the
        # backend registry; fall back to it on matplotlib < 3.9 (no registry).
        # Compare case-insensitively -- ``get_backend()`` and the backend lists
        # have varied in capitalisation across matplotlib versions.
        try:
            from matplotlib.backends import BackendFilter, backend_registry

            interactive_backends = backend_registry.list_builtin(BackendFilter.INTERACTIVE)
        except ImportError:  # matplotlib < 3.9
            from matplotlib import rcsetup

            interactive_backends = rcsetup.interactive_bk

        if matplotlib.get_backend().lower() in {b.lower() for b in interactive_backends}:
            if block:
                plt.show(block=True)
            else:
                plt.ion()
                plt.show(block=False)
        return fig

    # ---------- Repr / debug ----------

    def __repr__(self) -> str:
        return (
            f"telemed.Log(name={self.name!r}, n_frames={self.n_frames}, "
            f"duration={self.duration_s:.3f}s, "
            f"roi={self.b_mode_roi.width}x{self.b_mode_roi.height})"
        )
