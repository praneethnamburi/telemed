"""Telemed ultrasound device interop.

Three submodules, each with a focused responsibility:

* :mod:`immersionlab.telemed.crop` -- ffmpeg-based crop of telemed
  side-by-side mp4 exports into per-side h265 monochrome (or legacy
  libx264 yuv420p). The default-encoding workflow that lab data
  acquisition has been using.

* :mod:`immersionlab.telemed.export` -- COM-backed reader for the
  device-native ``.tvd`` files via EchoWave II's AutoInt1 automation
  interface. Extracts the chroma-free uint8 grayscale frames + native
  VFR per-frame timing + B-mode ROI + physical resolution into one
  HDF5 sidecar (``<stem>.tvd.h5``). Network-drive aware (copies via
  local temp because EchoWave's OpenFile rejects UNC / mapped-network
  paths). Requires Administrator privileges and a running EchoWave II
  instance; see the submodule docstring for setup details.

* :mod:`immersionlab.telemed.log` -- :class:`Log` for loading the
  ``.tvd.h5`` sidecar produced by ``export``. Same per-recording
  shape as :class:`delsys.Log` / :class:`immersionlab.ot.Log` etc.;
  use this as the entry point in analysis code.

Top-level re-exports preserve the pre-package call shape so existing
callers (``pn-projects``, etc.) keep working::

    from immersionlab import telemed
    telemed.crop_folder(...)
    telemed.Log("recording.tvd.h5")
"""
from __future__ import annotations

# Re-export the crop API at the package level for backward compatibility.
# Existing callers do ``from immersionlab import telemed; telemed.crop_folder(...)``;
# preserving that surface avoids a migration churn in pn-projects and friends.
from . import crop, export, log  # noqa: F401 — submodule access
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
from .export import (  # noqa: F401
    TelemedRecordingMeta,
    TelemedRoi,
    TelemedTvdReader,
    connect,
    extract_recording,
    extract_recording_folder,
)
from .log import Log  # noqa: F401
