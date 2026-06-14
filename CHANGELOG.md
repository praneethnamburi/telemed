# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Fully COM-free extraction path (`telemed.process(..., comfree=True)`).**
  A new backend (`telemed/_comfree.py`, kept cleanly separate from the COM
  `_extract.py`) that reads the device's **raw acoustic frames** + all
  per-recording metadata straight from the `.tvd` bytes — **no EchoWave, no
  Admin, no COM, no per-frame COM walk**. `process(src, comfree=True,
  out_dir=<local path>)` writes, per recording: one **mp4 per probe**
  (`<stem>.mp4` single / `<stem>_b1.mp4`+`<stem>_b2.mp4` dual) at the **native
  acoustic pixel grid** (`lines × active_depth`, depth-vertical, no gamma; a
  Sample-Aspect-Ratio tag carries the true display aspect while the stored
  pixels stay native; L/R-flipped when `b_is_scan_direction_changed`), **plus a
  `<stem>.tvd.h5` holding metadata + COM-free timing only — no `/frames/gray`**
  (the image data is in the mp4s, not duplicated into the h5). The h5 uses
  `schema_version="comfree-v1"` / `backend="comfree"` and carries the parsed
  acquisition `param_*` (a small .NET-BinaryFormatter reader lifts `UsgHWSettings`
  + probe/beamformer names + cine datetime), per-probe geometry + pixel scale
  (`probeN_{lines,samples,active,sar,axial_cm_per_px,lateral_cm_per_px,video}`),
  and `/timing/time_ms` (the COM-free **declared** frame set; `time_ms_com` is no
  longer needed). The container is streamed in a single constant-memory pass
  (large read buffer) so the 17 GB recordings extract without loading the file
  into RAM. Geometry (`lines`/`samples`/`active`) is read per-recording from
  `strf` — it varies across the cohort, so it is never assumed. Also exposed:
  `telemed.extract_comfree`, `telemed.read_geometry`, `telemed.read_metadata`.
  Note: EchoWave's spatial compounding + speckle filtration + enhancement are
  applied in the acoustic domain *before* the cine is stored, so the raw frames
  already carry them; only the display gray-mapping (DR window / palette) and the
  scan-conversion resampling are display-side and intentionally omitted.
- **End-tick caching + COM-free declared timing on `Log`.**
  `read_tvd_frame_ticks(tvd, cache=True)` (and `read_tvd_time_ms(..., cache=True)`)
  memoise the per-frame end-ticks to a sibling `<stem>.tvd.ticks.npy` sidecar,
  so the network-slow per-frame-chunk walk is paid once per `.tvd`; an
  up-to-date sidecar is loaded instead (and is honoured even if the `.tvd`
  is later absent). `.tvd` files are immutable, so the cache never goes
  stale (mtime-guarded anyway). **`export_h5` / `process()` now drop this
  `.tvd.ticks.npy` next to the `.h5` automatically** on every extract — read
  COM-free from the already-staged local `.tvd`, so it's ~free (no extra
  network read) and new recordings are cached without a separate pass.
  `telemed.Log` gains, on top of the existing
  `.time_ms` / `.n_frames` (the EchoWave-**stored** subset):
  `.time_ms_declared` (alias `.time_ms_comfree`) — the COM-free **declared**
  per-frame `time_ms`, read from the sibling `.tvd` via the cached sidecar
  (lazy; `None` if no `.tvd`/sidecar); `.n_frames_declared` — the `.tvd`
  declared count, from the stored `tvd_declared_n_frames` attr when present
  (no `.tvd` read) else the sibling `.tvd`; and `.n_frames_stored` — an
  explicit alias of `.n_frames`.
- `telemed.keep_full_speed()` — suppress Windows background-throttling of
  the COM extract loop. Opts the Python client **and** the running
  `EchoWave.exe` out of EcoQoS execution-speed throttling
  (`SetProcessInformation(ProcessPowerThrottling)`) and inhibits system
  sleep, so the ~5 fps extract rate holds when the driving console is
  backgrounded or the RDP session is **disconnected** (previously sagged
  to ~1-2 fps after a short grace period). `export_h5` / `process` call it
  by default (`keep_full_speed=True`); Windows-only, best-effort, no-op
  elsewhere, and never touches EchoWave's COM threading.
- `telemed.verify_complete(source)` — audit extracted `.tvd.h5`
  sidecar(s) for EchoWave **memory-truncated loads** (EchoWave silently
  loads only as many frames as fit in available RAM, leaving a short
  sidecar with no error). Compares the extracted `n_frames` against the
  frame count recorded in the source `.tvd` container header and, when
  present, the native `<stem>.mp4` export's `nb_frames`. Works on
  already-extracted sidecars without re-processing (parses the sibling
  `.tvd` header directly when the stored count is absent).
- `telemed.read_tvd_n_frames(tvd)` — parse the recorded frame count out
  of a `.tvd` container header (the RIFF-like "UIFF" form uses 64-bit
  chunk sizes). Independent of EchoWave's memory-limited load; the
  source `.tvd` is never written by the pipeline, so its header stays
  pristine.
- `telemed.backfill_tvd_n_frames(source)` — write `tvd_declared_n_frames`
  into already-extracted sidecars (header read + one attr write; no COM,
  no re-extraction) so older sidecars carry the same completeness
  metadata as future extractions.
- `telemed.looks_lut_inverted(source)` — run the LUT-inversion
  pixel-statistics test on an already-extracted `.tvd.h5` (the same
  check `export_h5` applies at extract time), so an existing cohort can
  be audited for the EchoWave < 4.4.0 inversion bug.
- HDF5 schema: `export_h5` now stores `tvd_declared_n_frames` (the `.tvd`
  header frame count) as a root attr when parseable, and emits a
  truncation **warning** during extraction when the loaded frame count
  falls well short of it. Surfaced on `Log.tvd_declared_n_frames`.

### Fixed

- `Log.view()` now **shows the window itself and stays responsive**.
  Previously it only built and returned the `Figure`, so the caller had
  to `plt.show()` manually; and a non-blocking `plt.show(block=False)`
  left the slider/arrow keys dead and the figure frozen on one frame,
  because nothing was pumping the GUI event loop (per `show`'s contract).
  `view()` now enables matplotlib interactive mode and shows non-blocking
  on a GUI backend (new `block=True` runs the main loop for scripts; a
  no-op on non-GUI backends like Agg).

### Changed

- `Log.view()` selects which probe to show with a new `panel` argument,
  replacing `crop`. `panel=None` (default) / `"all"` shows the full Echo
  Wave frame (both panels of a dual-probe recording); an `img_id` int
  (`1`, `2`, ...) or `"left"`/`"right"` (resolved by on-screen position,
  not `img_id` order) shows that probe's inner ultrasound image. The old
  `crop=False/True/"image"/"panel"` argument is gone; `Log.frame()` keeps
  `crop` for the outer-panel / ruler view.

- `export_h5` now **fails fast on the EchoWave < 4.4.0 grayscale-LUT
  bug**. Those builds return `GetLoadedFrameGray` inverted (`255 - x`),
  giving a bright-background sidecar from otherwise-fine `.tvd` device
  data. A pixel-statistics guard detects the inversion from the first
  few frames and raises with a clear "re-extract on EchoWave 4.4.0+"
  message instead of silently writing inverted pixels. Standardize on
  EchoWave 4.4.0+ for extraction.

### Docs

- README Install section now spells out the extract-path prerequisites
  (Echo Wave II installed + the one-time `AutoInt1_regasm.bat` COM
  registration + Admin EchoWave/Python per session) and cross-links the
  Per-session setup section, instead of leaving them only in the lower
  section. Clarified that the install path can vary by EchoWave version.

## [0.1.0] - 2026-05-24

Initial release. Graduated from `immersionlab.telemed` into a standalone
package; the development history (16 commits, from the original `crop_video`
helper through the `.tvd` direct-read pipeline and inner-image autocrop)
is preserved via `git filter-repo`.

### Added

- `telemed.export_h5(source)` — extract `.tvd` → `.tvd.h5` sidecars via
  the AutoInt1 COM API. Requires Administrator-mode EchoWave II +
  Administrator-mode Python. Network-drive aware (auto-stages via local
  temp because EchoWave's `OpenFile` fails on UNC / mapped paths).
- `telemed.export_video(source)` — encode `.tvd.h5` → `.mp4` per active
  B-mode panel. Lossless h265 mono by default (`preset="ultrafast"`).
  Inner-image autocrop strips depth ruler + side margins + bottom-tick
  row at encode time. Auto-splits dual-probe recordings.
- `telemed.process(source)` — end-to-end orchestrator
  (`.tvd → .tvd.h5 → .mp4(s) + .dnav-toc(s)`). Triages sources into
  Set A (extract + encode) and Set B (encode-only); runs both pipelines
  concurrently when the cohort is mixed.
- `telemed.Log(path)` — analysis entry point for `.tvd.h5` sidecars.
  Lazy frame loading; depth-calibrated `view()` browser;
  `Log.to_video()` / `Log.ensure_mp4()` / `Log.mp4_path()`
  single-recording conveniences around `export_video`;
  `Log.frame(idx, crop=, panel=)` per-panel access.
- HDF5 schema v1 — flat root attributes, per-img_id multi-ROI capture
  (`roi{N}_*`, `physical_d{x,y}{N}_cm_per_px`), opportunistic
  ParamGet sweep (~36 fields).
- `telemed.crop_video` / `telemed.crop_folder` — legacy
  EchoWave-mp4-export workflow (side-by-side mp4 → per-side h265
  monochrome). Ships under `DeprecationWarning`; will be removed in
  v0.2.0. Use `export_video` or `process()` against `.tvd` recordings
  instead.

### Notes

- HDF5 schema label collapsed from in-development `v1a{1..5}` series
  to a clean `v1` baseline for the package's v0.1.0 release. `Log`
  reads both old (`v1a{1..5}`) and new (`v1`) labels transparently for
  on-disk backcompat.
