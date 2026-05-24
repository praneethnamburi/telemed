"""Telemed ultrasound device interop.

**Status: queued for graduation** to a standalone ``telemed`` package
(mirroring the delsys pattern -- the source still lives here, but
this subpackage is the canonical entry point and downstream consumers
should rely on the public surface listed below rather than reaching
into ``immersionlab.telemed.*`` submodules. See
``specs/immersionToolbox.md`` Roadmap.



Three concerns, each in its own submodule:

* :mod:`immersionlab.telemed.crop` -- ffmpeg-based crop of telemed
  side-by-side mp4 exports into per-side h265 monochrome (or legacy
  libx264 yuv420p). The default-encoding workflow the lab has been
  using for data acquisition.

* :func:`immersionlab.telemed.export` -- single unified entry point
  for COM-backed extraction of Telemed ``.tvd`` files into HDF5
  sidecars (``<stem>.tvd.h5``: chroma-free uint8 grayscale frames +
  native VFR per-frame timing + B-mode ROI + physical resolution).
  Accepts a file path, a folder, or any iterable of files / folders.
  Network-drive aware (auto-stages via local temp because EchoWave's
  OpenFile rejects UNC / mapped-network paths). Requires Administrator
  privileges and a running EchoWave II instance; see the internal
  ``_extract`` submodule docstring for setup details.

* :mod:`immersionlab.telemed.log` -- :class:`Log` for loading the
  ``.tvd.h5`` sidecar produced by ``export``. Same per-recording
  shape as :class:`delsys.Log` / :class:`immersionlab.ot.Log` etc.

Public surface (everything advertised here)::

    from immersionlab import telemed

    # Cropping (mp4)
    telemed.crop_video(...)
    telemed.crop_folder(...)

    # Extraction (tvd -> h5)
    telemed.export(source)         # file | folder | list of either

    # Analysis
    lf = telemed.Log("recording.tvd.h5")
    lf.view()
"""
from __future__ import annotations

# Submodule access for advanced users (no underscore in the public
# layout). ``_extract`` is intentionally underscored -- callers should
# reach the extraction surface via ``telemed.export(...)`` rather than
# poking the internals.
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
    connect,
    export,
)
from .log import Log  # noqa: F401
