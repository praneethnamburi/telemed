"""Telemed ultrasound device interop.

**Status: queued for graduation** to a standalone ``telemed`` package
(mirroring the delsys pattern -- the source still lives here, but
this subpackage is the canonical entry point and downstream consumers
should rely on the public surface listed below rather than reaching
into ``immersionlab.telemed.*`` submodules. See
``specs/immersionToolbox.md`` Roadmap.



Three named entry points -- pick the one that matches what you want:

* :func:`immersionlab.telemed.export_h5` -- extract ``.tvd`` ->
  ``.tvd.h5`` sidecars via the COM API. Requires Administrator-mode
  EchoWave II + Administrator-mode Python. Network-drive aware
  (auto-stages via local temp because EchoWave's ``OpenFile`` fails
  on UNC / mapped paths).

* :func:`immersionlab.telemed.export_video` -- encode ``.tvd.h5`` ->
  ``.mp4`` per active B-mode panel. Offline (no EchoWave needed).
  Lossless h265 mono by default (raw uint8 gray frames -> nothing to
  gain from CRF quantisation; ``preset="ultrafast"`` is the bench-
  validated sweet spot for both encode AND decode speed at ~15%
  larger files than slow). Auto-splits dual-probe recordings per
  ``n_b_images``; normalises L/R-flip when
  ``b_is_scan_direction_changed`` is True so cohort mp4s land in a
  canonical orientation.

* :func:`immersionlab.telemed.process` -- composite that calls
  ``export_h5`` then ``export_video`` on the same source. The
  one-shot pipeline for ``.tvd`` -> ``.tvd.h5`` -> ``.mp4(s)``.
  Idempotent under default ``skip_existing=True``.

Plus :mod:`immersionlab.telemed.crop` for the legacy mp4-crop workflow
(side-by-side EchoWave mp4 export -> per-side h265 monochrome) and
:class:`immersionlab.telemed.Log` for loading + viewing a single
``.tvd.h5`` sidecar (``Log.view`` includes a depth-calibrated scale
bar; ``Log.to_video`` is a single-recording convenience around
``export_video``).

Public surface (everything advertised here)::

    from immersionlab import telemed

    # Pipeline
    telemed.export_h5(source)         # tvd -> tvd.h5   (Admin + EchoWave)
    telemed.export_video(source)      # tvd.h5 -> mp4(s)  (offline)
    telemed.process(source)           # = export_h5 + export_video

    # Legacy mp4 cropping (Telemed side-by-side mp4 -> per-side h265)
    telemed.crop_video(...)
    telemed.crop_folder(...)

    # Analysis
    lf = telemed.Log("recording.tvd.h5")
    lf.view()
    lf.to_video()                     # single-recording mp4 encode
"""
from __future__ import annotations

# Submodule access for advanced users (no underscore in the public
# layout). ``_extract`` / ``_encode`` are intentionally underscored --
# callers reach the extraction + encode surfaces via the three named
# functions below rather than poking internals.
from . import crop, log  # noqa: F401

from .crop import (  # noqa: F401
    CROP_H,
    CROP_W,
    CROP_Y,
    X_LEFT,
    X_RIGHT,
    _build_crop_cmd,
    _MONO_DEFAULT_CRF,
    crop_folder,
    crop_video,
)
from ._extract import (  # noqa: F401
    TelemedRecordingMeta,
    TelemedRoi,
    TelemedTvdReader,
    _PARAM_SPECS,
    _ParamSpec,
    connect,
    export_h5,
    process,
)
from ._encode import export_video  # noqa: F401
from .log import Log  # noqa: F401
