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
from typing import Optional, Union

import h5py
import numpy as np


@dataclass(frozen=True)
class Roi:
    """B-mode region-of-interest, mirroring the export-side dataclass.

    Stored as immutable so callers can pass it around without
    accidental mutation. The slice helpers are convenient when
    indexing into a full-frame numpy array.
    """

    x1: int
    x2: int
    y1: int
    y2: int
    width: int
    height: int

    def as_slice(self) -> tuple:
        """Return ``(y_slice, x_slice)`` for indexing a (H, W) array.

        Telemed's COM API uses 1-based pixel indexing; we convert to
        0-based Python slices here. End points are inclusive in the
        source convention, so the slice end gets +1.
        """
        return (slice(self.y1 - 1, self.y2), slice(self.x1 - 1, self.x2))


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
        b_mode_roi (Roi): B-mode ROI within the full frame.
        physical_dx_cm_per_px / physical_dy_cm_per_px (float):
            Spatial resolution of the B-mode image.
        time_ms (np.ndarray): Absolute time of each frame in ms, with
            frame 0 -> 0.0. Shape ``(n_frames,)``.
        ifi_ms (np.ndarray): Inter-frame intervals in ms. ``ifi_ms[0]``
            is 0 (frame 1 anchor). Shape ``(n_frames,)``.
        source_tvd_path (str): Path the data was extracted from.
        extracted_at_iso (str): When the HDF5 was written.
        schema_version (int): HDF5 schema version.

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
            self.b_mode_roi: Roi = Roi(
                x1=int(a["roi_x1"]),
                x2=int(a["roi_x2"]),
                y1=int(a["roi_y1"]),
                y2=int(a["roi_y2"]),
                width=int(a["roi_width"]),
                height=int(a["roi_height"]),
            )
            self.physical_dx_cm_per_px: float = float(a["physical_dx_cm_per_px"])
            self.physical_dy_cm_per_px: float = float(a["physical_dy_cm_per_px"])
            self.source_tvd_path: str = str(a["source_tvd_path"])
            self.extracted_at_iso: str = str(a["extracted_at_iso"])
            self.schema_version: int = int(a["schema_version"])

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
