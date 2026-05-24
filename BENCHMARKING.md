# telemed -- encode pipeline benchmarks

Wall-clock + accuracy numbers for the Telemed `.tvd` -> `.tvd.h5` ->
`.mp4` pipeline, tracked across the encoding-preset decision and the
lossless-vs-lossy decision. The pipeline lives in
`immersionlab.telemed` -- see [`README.md`](README.md) for the design
overview.

Two axes:

- **Encode-side throughput + output size** -- how fast does
  `export_video()` produce mp4(s) at each libx265 preset, and how
  large are the resulting files? Drives the `_VIDEO_DEFAULT_PRESET`
  choice.
- **Downstream consumption performance** -- per-preset linear decode
  (cv2 + dnav PyAV+TOC), random-access seek rate, TOC build time, and
  DLC inference fps. Drives whether the encode-side choice has any
  hidden cost for analysis users.

Companion document at
[`S:/_corpus/telemed/_bench/run_bench.py`](file:///S:/_corpus/telemed/_bench/run_bench.py)
holds the older 2026-05-23 lossy-source CRF-tuning bench that drove
`telemed.crop`'s `_MONO_DEFAULT_CRF`. That bench's input is the
already-lossy mp4 export from EchoWave -- different optimisation
target from the `export_video` bench documented here (raw uint8 gray
input from the .tvd.h5 sidecar).

## Encode + decode + seek + TOC -- summary

Real Telemed pia02 acquisition, dual-probe, 19248 frames per panel,
705x558 cropped ROI. Bench fixture:
`C:/data/temp2/018/pia02_s018_003 fav piece 20250528 142714_b1_*.mp4`.

env: `b4` (Python 3.10, h5py, dnav 1.5.0); GPU not involved in the
encode + decode + seek + TOC paths (CPU-only libx265, cv2 software,
PyAV software).

| preset | encode fps | size | cv2 linear decode | dnav linear decode | dnav 50 seeks | dnav TOC build |
|---|---:|---:|---:|---:|---:|---:|
| slow | 55 | 2.84 GB (1.00x) | 507 fps | 277 fps | 2.2/s | 67.1s |
| medium | 110 | 2.95 GB (1.04x) | 548 fps | 290 fps | 2.4/s | 62.3s |
| fast | 150 | 3.08 GB (1.08x) | 578 fps | 292 fps | 2.4/s | 63.4s |
| **ultrafast** | **400** | **3.26 GB (1.15x)** | **1293 fps** | **553 fps** | **4.4/s** | **34.4s** |
| (RFA2 legacy h264 yuv420p) | n/a | 60 MB | n/a* | n/a* | n/a* | 11.1s |

\* RFA2 legacy is a different codec / pix_fmt and not a candidate for
the encode-preset decision; only its TOC build time is comparable
(and it's faster only because it's 50x smaller).

**Decision (2026-05-24): `_VIDEO_DEFAULT_PRESET = "ultrafast"`.** The
faster preset wins on every time axis -- encode AND linear decode AND
random-access seeks AND TOC build -- at +6% size over `fast` and +15%
over `slow`. The simpler bitstream structure that ultrafast emits is
cheaper for both encoder and decoder to process.

### Why this is counterintuitive

For *lossy* h265, faster presets typically produce larger files with
*harder* bitstreams to decode (less optimised prediction structures
means more residual to invert). The opposite holds for *lossless*
h265: the reconstructed pixels must be bit-exact regardless of
preset, so the preset only controls how aggressively the encoder
searches the prediction-mode space. Faster presets pick simpler
prediction modes (larger transform blocks, less inter-frame motion
search) that the decoder can also invert with less work. So
lossless h265 + faster preset is genuinely faster on both sides --
the only cost is bitstream entropy efficiency (a few percentage
points of file size).

## DLC inference -- summary

env: `dlc3rc14` (DLC 3.0.0rc14, ResNet-50, batch=8).
Model: `interosseous_pn24-x-2025-10-24`, shuffle 1, snapshot 300.

| preset | DLC wall | DLC fps | relative to slow |
|---|---:|---:|---:|
| slow | 112.8s | 170.6 | 1.00x |
| medium | 110.9s | 173.5 | 1.02x |
| fast | 111.0s | 173.4 | 1.02x |
| **ultrafast** | **108.8s** | **176.8** | **1.04x** |
| (RFA2 legacy h264) | 102.3s | 188.2 | 1.10x |

**Conclusion: encoder preset is effectively neutral for DLC inference.**
The four lossless presets fall within 4% of each other; inference is
GPU-bound on the ResNet-50 forward pass (~175 fps ceiling), well
below any of the decoder's max throughput (277-1293 fps). The legacy
h264 path is ~4% faster because its decoder is lighter, but you'd be
paying that in real keypoint accuracy (see next section).

## DUSTrack UI fps -- summary

Per-frame `browser.update() + processEvents()` budget against each
candidate mp4, with `fast_render=True` (Tier 2 Qt-native image
pane). Adapter probe constructs `DUSTrack(video_path,
height_ratios=(3,1,1), fast_render=True)` directly so we can bench
videos that aren't in a DLC project; annotation load is just the
default buffer layer, so absolute fps numbers aren't directly
comparable to dustrack BENCHMARKING.md's published 27.8 fps (which
includes DLC trace layers on a 36715-frame video). The
**encoding-axis delta** is the apples-to-apples comparison.

env: `dlc3rc14` (DLC 3.0.0rc14, PySide6 6.4.2, matplotlib QtAgg).
N=185 measured per candidate (15 warmup discarded).

| candidate | median ms | fps | update ms | process_events ms |
|---|---:|---:|---:|---:|
| slow | 24.71 | 40.5 | 23.66 | 1.02 |
| medium | 24.88 | 40.2 | 23.80 | 1.04 |
| fast | 25.08 | 39.9 | 23.96 | 1.03 |
| **ultrafast** | **23.79** | **42.0** | **22.83** | 1.05 |
| **RFA2 legacy h264** | **29.11** | **34.4** | **28.12** | 1.06 |

**Conclusion**: lossless h265 mono is ~22% faster than legacy h264
yuv420p in the dustrack UI (42 vs 34 fps); within the lossless presets
the choice is noise (~5% spread). The encoding-axis win is the gap
between "lossless h265 mono" and "anything with chroma planes" -- in
dustrack's per-frame pipeline, dnav's RGB conversion runs the chroma
planes through swscale even when they're all-128, so h265 4:0:0
genuinely saves work the yuv420p path can't.

The preset-within-lossless invariance is expected: decode savings
between slow and ultrafast in the dnav linear bench were ~1.8 ms /
frame, but that's a small fraction of the ~24 ms per-frame UI
budget; it gets lost in process_events / Qt pixmap upload variance.

### Cross-reference: dustrack BENCHMARKING.md baseline

dustrack's own bench publishes 27.8 fps (36.0 ms median) for
1.5.0-fast-render Tier 2 on the production `pia02_s001_007_LFA2.mp4`
(36715 frames + 3 annotation layers). The 3.94x gain in that doc is
the **architectural** speedup from fast_render Tier 2 vs the 1.3.0
matplotlib baseline (7.1 fps); the encoding-axis improvement
documented here is an **additional ~22% on top** when the underlying
mp4 changes from legacy h264 yuv420p to lossless h265 mono. Both
gains stack.

The absolute fps gap between dustrack's 27.8 (production load) and
this bench's 42 (default buffer only) is the annotation-layer
overhead. To reproduce 27.8 with the new encoding for a direct
apples-to-apples comparison, the `pia02_s001_007_LFA2.mp4` source
would need re-encoding at each preset and `dustrack/tests/qt_learning/
14_benchmark_fast_render.py` re-run on the new mp4s.

## DLC accuracy parity -- summary

Reference: `slow` preset's DLC h5. The four lossless presets MUST
produce identical pixel decoding (h265 lossless guarantee), so any
non-zero error comes from non-deterministic CUDA convolution kernels
in the model forward pass -- below sub-pixel resolution.

| candidate | max |delta| (px) | mean |delta| (px) | median euclidean (px) | p99 euclidean (px) | max euclidean (px) |
|---|---:|---:|---:|---:|---:|
| medium  vs slow | 3.05e-5 | 5.72e-9 | 0.000 | 0.000 | 0.000 |
| fast    vs slow | 3.05e-5 | 6.64e-9 | 0.000 | 0.000 | 0.000 |
| ultrafast vs slow | 3.05e-5 | 5.71e-9 | 0.000 | 0.000 | 0.000 |
| **RFA2 legacy vs slow** | **32.8** | **0.28** | **0.58** | **2.26** | **41.2** |

**Conclusion: lossless preset choice is accuracy-free.** All four
lossless presets land within float32 numerical noise of the slow
baseline -- max absolute keypoint difference is 3.05e-5 px (well
below the sub-pixel DLC discrimination threshold). The four
candidates produce essentially identical keypoint outputs.

**The legacy lossy h264/yuv420p path costs ~0.58 px median, 2.26 px
p99, and outlier-frame errors up to 41 px.** This reproduces the
2026-05-23 crop-bench finding ([`S:/_corpus/telemed/_bench/`](file:///S:/_corpus/telemed/_bench/))
that the legacy mp4 export costs visible keypoint accuracy; the
lossless h265 mono pipeline closes that gap.

## Methodology

### Encode bench

Each preset encoded once from the cropped uint8 gray frames lazily
streamed out of the .tvd.h5 sidecar (i.e. the production
`export_video` code path), with `-hide_banner -loglevel error` to
prevent Popen pipe deadlock. Wall-clock time measured by the
`tqdm` bar reported in `_encode_one_panel`; file size from the
filesystem after each encode completes.

Single-rep numbers because each rep is ~20k frames at 55-400 fps
(~50s-400s per rep) -- effectively averaging across a large sample.
The encoder is CPU-bound and the test machine has no other load
during the bench.

### Linear decode + seek + TOC build

Each candidate decoded in two backends:

1. **cv2 linear** -- `cv2.VideoCapture` opened, `cap.read()` until
   EOF, frames discarded. Measures decode throughput in the standard
   DLC inference / cv2 consumer path.
2. **dnav linear** -- `datanavigator.VideoReader` opened (first call
   builds TOC; cached on disk for subsequent opens),
   `vr[i] for i in range(n)`. Measures decode throughput in the
   PyAV+TOC consumer path used by DUSTrack and any datanavigator
   power-user.

Random-access seeks: 50 frame indices selected from `np.random.default_rng(42)`
without replacement, sorted (mixes small and large seeks). Same
`dnav.VideoReader`. Measures the typical DUSTrack scrubbing /
labeling pattern.

TOC build: pre-existing `.dnav-toc` sidecar deleted before each
candidate; `dn.VideoReader(path)` call timed end-to-end (TOC is
built lazily inside `__init__` when no cached sidecar is found).

### DLC inference

`deeplabcut.analyze_videos(config, [mp4], shuffle=1, save_as_csv=False,
destfolder=stage)`. Wall-clock measured around the call; fps =
`n_frames / wall_s`. Each candidate writes to its own destfolder to
avoid DLC's "already analyzed" short-circuit.

### Accuracy parity

For each candidate's DLC h5, load the keypoint coordinates via
`pd.read_hdf` + `.xs('x', level='coords', axis=1)` / `.xs('y', ...)`,
stack to `(F, K, 2)`. Reference = the `slow` preset's h5. Report:

- **`max|delta|`** -- maximum absolute coordinate difference (px),
  per (F, K, coord)
- **`mean|delta|`** -- mean absolute coordinate difference (px),
  per (F, K, coord)
- **median / p99 / max euclidean** -- distribution of euclidean
  distance per (F, K), NaN-masked

NaN-masking matters because DLC emits NaN for low-confidence frames;
those drop out of the parity statistics.

## How to run

```powershell
# Sit in the bench dir; .tvd.h5 fixture must already exist (run
# telemed.export_h5 first if not).
Set-Location C:\data\temp2\018\_dlc_bench

# Encode bench (re-encode at each preset; ffmpeg in PATH required;
# uses the b4 env's immersionlab.telemed.export_video).
C:\Users\praneeth\anaconda3\envs\b4\python.exe -c @'
from pathlib import Path
from immersionlab import telemed
H5 = Path("C:/data/temp2/018/pia02_s018_003 fav piece 20250528 142714.tvd.h5")
OUT = Path("C:/data/temp2/018")
for preset in ("slow", "medium", "fast", "ultrafast"):
    out = OUT / f"{H5.stem.removesuffix('.tvd')}_b1_{preset}.mp4"
    if out.exists(): out.unlink()
    telemed.export_video(H5, preset=preset, overwrite=True)
'@

# Linear decode + seek + TOC bench (cv2 + dnav, all 4 presets):
C:\Users\praneeth\anaconda3\envs\b4\python.exe `
    C:\dev\immersionToolbox\immersionlab\telemed\_bench_decode.py

# DLC inference bench (dlc3rc14 env):
Set-Location C:\data\temp2\018\_dlc_bench
C:\Users\praneeth\anaconda3\envs\dlc3rc14\python.exe run_dlc_bench.py

# DLC accuracy parity (b4 env -- reads the h5s the bench above wrote):
C:\Users\praneeth\anaconda3\envs\b4\python.exe `
    C:\dev\immersionToolbox\immersionlab\telemed\_bench_dlc_parity.py
```

The bench scripts at `_bench_*.py` (h5 input + result tables) are
sketched in the **Bench artefacts** section below.

## Hardware / environment

- Machine: Windows 11, the development workstation
- Conda envs: `b4` (Python 3.10, h5py, dnav 1.5.0) for the encode +
  decode + accuracy benches; `dlc3rc14` (Python 3.10, DLC 3.0.0rc14,
  PySide6 6.4.2) for the DLC inference bench
- ffmpeg: `C:\ffmpeg\bin\ffmpeg.exe` (libx265 + h264 enabled)
- Video storage: local `C:` drive (no network-drive first-touch
  overhead). pia02 production lives on `M:` and first-touch TOC is
  ~37 s; that's a separate concern not benchmarked here
- Bench fixture: `pia02_s018_003 fav piece 20250528 142714` (dual-
  probe, b1 panel = 705x558 px, 19248 frames; representative of the
  pia02 cohort)

## Bench artefacts

Bench scripts that produce the tables above. Kept under the
`immersionlab/telemed/` folder so they travel with the package
(unlike the older `S:/_corpus/telemed/_bench/run_bench.py` which is
on a network drive).

- `_bench_decode.py` -- cv2 + dnav linear decode + random seeks + TOC
  build, all 4 presets. (Not in-repo yet; pasteable from this doc's
  raw-results section.)
- `_bench_dlc_parity.py` -- DLC h5 parity vs slow reference.
- DLC bench harness: `C:\data\temp2\018\_dlc_bench\run_dlc_bench.py`.

## Raw results

### Encode bench -- 2026-05-24

Fixture: `pia02_s018_003 fav piece 20250528 142714_b1_*.mp4` (one
mp4 per preset; cropped 705x558 ROI from the source .tvd.h5; 19248
frames).

| preset | encode fps | wall encode | file size |
|---|---:|---:|---:|
| slow | ~55 | ~350 s | 2840843 KB (2.84 GB) |
| medium | ~110 | ~175 s | 2947865 KB (2.95 GB) |
| fast | ~150 | ~128 s | 3076457 KB (3.08 GB) |
| ultrafast | ~400 | ~48 s | 3261113 KB (3.26 GB) |

### Decode + seek + TOC bench -- 2026-05-24

| preset | cv2 linear (s, fps) | dnav linear (s, fps) | dnav 50 seeks (s, seeks/s) | dnav TOC build (s) |
|---|---:|---:|---:|---:|
| slow | 37.9 / 507.3 | 69.5 / 277.0 | 22.5 / 2.2 | 67.05 |
| medium | 35.1 / 548.2 | 66.4 / 289.7 | 20.8 / 2.4 | 62.32 |
| fast | 33.3 / 578.2 | 66.0 / 291.5 | 21.1 / 2.4 | 63.44 |
| ultrafast | 14.9 / 1292.8 | 34.8 / 553.5 | 11.4 / 4.4 | 34.36 |
| RFA2 legacy h264 | n/a | n/a | n/a | 11.10 |

### DLC inference bench -- 2026-05-24

| preset | DLC wall (s) | DLC fps | output h5 size |
|---|---:|---:|---:|
| slow | 112.8 | 170.6 | -- |
| medium | 110.9 | 173.5 | -- |
| fast | 111.0 | 173.4 | -- |
| ultrafast | 108.8 | 176.8 | -- |
| RFA2 legacy h264 | 102.3 | 188.2 | -- |

### DUSTrack UI bench -- 2026-05-24

Per-frame `update() + processEvents()` budget against each
candidate. Adapter probe at `C:/data/temp2/018/_dlc_bench/14_encoding_axis_bench.py`;
env `dlc3rc14`; fast_render=True; default buffer annotation only;
N=185 measured per candidate (15 warmup discarded).

| candidate | median ms | fps | update median ms | process_events median ms |
|---|---:|---:|---:|---:|
| slow | 24.71 | 40.5 | 23.66 | 1.02 |
| medium | 24.88 | 40.2 | 23.80 | 1.04 |
| fast | 25.08 | 39.9 | 23.96 | 1.03 |
| ultrafast | 23.79 | 42.0 | 22.83 | 1.05 |
| RFA2 legacy h264 | 29.11 | 34.4 | 28.12 | 1.06 |

### DLC accuracy parity vs slow -- 2026-05-24

| candidate | max |delta| (px) | mean |delta| (px) | median euclidean (px) | p99 euclidean (px) | max euclidean (px) |
|---|---:|---:|---:|---:|---:|
| medium | 3.052e-5 | 5.715e-9 | 0.000 | 0.000 | 0.000 |
| fast | 3.052e-5 | 6.637e-9 | 0.000 | 0.000 | 0.000 |
| ultrafast | 3.052e-5 | 5.712e-9 | 0.000 | 0.000 | 0.000 |
| RFA2 legacy h264 | 32.76 | 0.281 | 0.583 | 2.261 | 41.21 |

## Known follow-ons (not chased)

- **DUSTrack UI fps on production-load video** -- the adapter probe
  documented above runs on a standalone video with default annotation
  load only, so absolute fps doesn't directly compare to dustrack
  BENCHMARKING.md's 27.8 fps (which includes DLC trace layers). To
  produce a like-for-like number: re-encode the production video
  `pia02_s001_007_LFA2.mp4` at each preset (36715-frame source on
  M:), and re-run `dustrack/tests/qt_learning/14_benchmark_fast_render.py`
  on each.
- **Network-drive TOC build cost** -- the 37 s first-touch TOC build
  on `M:` reported in [`dustrack/BENCHMARKING.md`](file:///C:/dev/dustrack/BENCHMARKING.md)
  was on a much larger 36715-frame video. Re-bench on a per-preset
  basis if a server-side TOC warming workflow is built.
- **Encode parallelism** -- pushed back on (2026-05-24) since
  libx265 is already internally multi-threaded; ultrafast picked up
  the speed slack instead. Revisit if a bulk re-encode campaign
  needs better wall-clock than the ~48 s/recording ultrafast number.
- **Probe-aperture lookup table** for the inner-image-crop question
  (cropping out the depth ruler + side margins from the panel ROI).
  Open since 2026-05-24; deferred per the user's "nothing yet"
  decision.

## Decision log

| date | decision | bench data | next-review trigger |
|---|---|---|---|
| 2026-05-23 | `telemed.crop` defaults to h265 mono / `_MONO_DEFAULT_CRF=24` | `S:/_corpus/telemed/_bench/` DLC parity bench on lossy-source crop | -- |
| 2026-05-24 | `export_video` defaults to lossless h265 mono | raw uint8 gray source has nothing to gain from CRF quantisation; tolerable size at corpus scale | a multi-modal probe with different physical_dx vs physical_dy |
| 2026-05-24 | `_VIDEO_DEFAULT_PRESET = "ultrafast"` | this doc (encode + decode + seek + TOC + DLC fps + DLC parity all neutral or favorable vs slower presets) | a corpus where +15% file size is intolerable |
| 2026-05-24 | CPU encoding only (no GPU) | NVENC h265 lossless is inconsistent across drivers + typically converts mono to yuv420p with synthetic chroma | a hardware path with reliable bit-exact h265 mono lossless on GPU |
