# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
