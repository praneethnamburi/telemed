"""Telemed ultrasound interop: extract .tvd recordings to HDF5 + per-panel mp4.

Three named entry points -- pick the one that matches what you want:

* :func:`telemed.export_h5` -- extract ``.tvd`` -> ``.tvd.h5`` sidecars
  via the AutoInt1 COM API. Requires Administrator-mode EchoWave II +
  Administrator-mode Python (Windows-only). Network-drive aware
  (auto-stages via local temp because EchoWave's ``OpenFile`` fails on
  UNC / mapped paths).

* :func:`telemed.export_video` -- encode ``.tvd.h5`` -> ``.mp4`` per
  active B-mode panel. Offline (no EchoWave needed). Lossless h265
  mono by default (raw uint8 gray frames -> nothing to gain from CRF
  quantisation; ``preset="ultrafast"`` is the bench-validated sweet
  spot for both encode AND decode speed at ~15% larger files than
  slow). Auto-splits dual-probe recordings per ``n_b_images``;
  normalises L/R-flip when ``b_is_scan_direction_changed`` is True so
  cohort mp4s land in a canonical orientation.

* :func:`telemed.process` -- end-to-end orchestrator for
  ``.tvd -> .tvd.h5 -> .mp4(s) + .dnav-toc(s)``. Triages sources into
  set A (need extraction) and set B (already have .h5); runs the
  appropriate pipeline(s), or both concurrently when the cohort is
  mixed (the COM-bound extract pipeline and the CPU/disk-bound
  encode-only pipeline have orthogonal bottlenecks). For Set A, each
  file's encode + TOC + upload runs in the background while the next
  file's COM extract executes, so wall time is bounded by extract
  alone. Returns ``{"h5": ..., "video": ..., "toc": ...}``.
  Idempotent under default ``skip_existing=True``.

Plus :mod:`telemed.crop` for the legacy mp4-crop workflow (side-by-side
EchoWave mp4 export -> per-side h265 monochrome; deprecated, removed
in v0.2.0) and :class:`telemed.Log` for loading + viewing a single
``.tvd.h5`` sidecar (``Log.view`` includes a depth-calibrated scale
bar; ``Log.to_video`` / ``Log.ensure_mp4`` are single-recording
conveniences around ``export_video``; ``Log.mp4_path`` reports where a
per-panel mp4 would land without forcing an encode).

Public surface (everything advertised here)::

    import telemed

    # Pipeline
    telemed.export_h5(source)         # tvd -> tvd.h5   (Admin + EchoWave)
    telemed.export_video(source)      # tvd.h5 -> mp4(s)  (offline)
    telemed.process(source)           # = export_h5 + export_video

    # Completeness QC (catch EchoWave memory-truncated extractions)
    telemed.verify_complete(source)   # compare extracted vs .tvd-declared
    telemed.backfill_tvd_n_frames(source)  # add declared count to old h5s
    telemed.read_tvd_n_frames(tvd)    # frame count from a .tvd header
    telemed.looks_lut_inverted(h5)    # detect EchoWave <4.4.0 LUT bug

    # Legacy mp4 cropping (deprecated; will be removed in v0.2.0)
    telemed.crop_video(...)
    telemed.crop_folder(...)

    # Analysis
    lf = telemed.Log("recording.tvd.h5")
    lf.view()
    lf.to_video()                     # encode every active panel
    mp4 = lf.ensure_mp4(panel=2)      # encode-if-missing, return path
    lf.mp4_path(panel=1)              # plan-only, no I/O
    lf.frame(0, crop=True, panel=2)   # per-panel cropped frame
"""

from __future__ import annotations

__version__ = "0.1.0"

# Submodule access for advanced users (no underscore in the public
# layout). ``_extract`` / ``_encode`` are intentionally underscored --
# callers reach the extraction + encode surfaces via the three named
# functions below rather than poking internals.
from . import crop, log  # noqa: F401
from ._dispatch import process  # noqa: F401
from ._encode import export_video  # noqa: F401
from ._extract import (  # noqa: F401
    _PARAM_SPECS,
    TelemedRecordingMeta,
    TelemedRoi,
    TelemedTvdReader,
    _ParamSpec,
    connect,
    export_h5,
    read_tvd_n_frames,
)
from ._qc import backfill_tvd_n_frames, looks_lut_inverted, verify_complete  # noqa: F401
from .crop import (  # noqa: F401
    _MONO_DEFAULT_CRF,
    CROP_H,
    CROP_W,
    CROP_Y,
    X_LEFT,
    X_RIGHT,
    _build_crop_cmd,
    crop_folder,
    crop_video,
)
from .log import Log  # noqa: F401
