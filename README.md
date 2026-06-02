# telemed

Direct-read pipeline for Telemed ultrasound `.tvd` recordings — HDF5
metadata + per-panel mp4 with inner-image autocrop.

Extracts `.tvd` recordings to HDF5 sidecars via the AutoInt1 COM API,
then encodes to mp4 for downstream DLC / DUSTrack consumption. This
document is a reference for the design decisions baked into the
package; the per-function docstrings cover the surface in detail.

## Install

```
pip install telemed
```

The Windows-only `pywin32` dependency (required by the `.tvd → .tvd.h5`
extract path, which talks to EchoWave via COM) installs automatically
on Windows and is skipped on macOS / Linux. Non-Windows installs still
get the encode + analyze paths (`export_video`, `process()` over Set B
`.tvd.h5` inputs, `Log`), they just can't run COM extraction.

`ffmpeg` must be on PATH (used for the h265 encode).

### Extract-path prerequisites (`export_h5` / `process`)

Reading `.tvd` files goes through EchoWave's AutoInt1 COM server, so
beyond `pip install telemed` the extract path additionally needs, on
the Windows machine that has the device software:

1. **Echo Wave II installed** (the Telemed vendor application). The COM
   server ships inside it — there's nothing to download separately.
2. **A one-time COM registration** of the AutoInt1 ProgID — run
   `AutoInt1_regasm.bat` from an Administrator PowerShell (full command
   below).
3. **Admin EchoWave II + Admin Python per session** (COM ROT
   publication is per-elevation — a non-elevated client can't see an
   elevated server).

Full step-by-step in [Per-session setup](#per-session-setup-administrator-echowave--administrator-python)
below. The encode + analyze paths (`export_video`, `Log`,
`process()` over already-extracted `.tvd.h5` files) need none of this —
just the `pip install` + `ffmpeg`.

## Quickstart

```python
import telemed

# Start EchoWave II as Administrator first; start your Python
# session as Administrator too. (COM ROT is per-elevation.)

telemed.process(r"M:/data/pia02")
# Equivalent to:
# telemed.export_h5(r"M:/data/pia02")     # tvd  -> tvd.h5  (Admin + EchoWave)
# telemed.export_video(r"M:/data/pia02")  # tvd.h5 -> mp4(s) (offline)

# Inspect a single recording:
lf = telemed.Log("M:/data/pia02/scan.tvd.h5")
lf.view()                  # matplotlib browser (full frame), with scale bar
lf.view("right")           # just one probe of a dual-probe scan ("left"/1/2 too)
lf.to_video()              # single-recording mp4 encode
```

## Pipeline

```
       .tvd  --[COM via AutoInt1]-->  .tvd.h5  --[ffmpeg]-->  .mp4(s)
                export_h5                     export_video
                Admin EchoWave required       offline (no EchoWave)
```

`process()` chains both stages on the same source. All three accept a
file, folder, or iterable; `recursive=True` by default; idempotent
under `skip_existing=True` (default).

## HDF5 schema (v1)

Composite suffix `<stem>.tvd.h5` so downstream glob walks (`*.tvd.h5`)
catch them without picking up unrelated HDF5 data.

**Root attributes** (flat -- no nested groups):
- `n_frames`, `full_frame_width`, `full_frame_height`
- `n_b_images` -- count of active B-mode panels (1 = single probe;
  2 = dual probe; up to 4)
- `source_tvd_path`, `extracted_at_iso`, `schema_version="v1"`
- `image_dx_cm_per_px`, `image_dy_cm_per_px` -- display scale derived
  from `b_depth_mm / 10 / panel_height_px` (Telemed support's "trust
  the depth setting" calibration). Skipped if `b_depth` wasn't
  captured.
- Per active img_id N ∈ {1, 2, 3, 4} (1=B, 2=B2, 3=B3, 4=B4):
  - `roi{N}_x1`, `roi{N}_x2`, `roi{N}_y1`, `roi{N}_y2` (1-based pixel
    coords matching AutoInt1's convention)
  - `roi{N}_width`, `roi{N}_height` (inclusive pixel counts)
  - `physical_dx{N}_cm_per_px`, `physical_dy{N}_cm_per_px`
    (beamformer-native scale -- see "scale" note below)
- `param_*` -- opportunistic ParamGet sweep (~36 fields per recording
  on real EchoWave acquisitions): probe / beamformer identity, cine
  end timestamp, B-mode acquisition (depth, frequency, gain, dynamic
  range, focus, THI, frame averaging, ...), geometry / orientation
  (scan-direction-changed, rotate, view-area, scan-type, ...),
  sanity probes (file-opened, scanning-state, probe-active).

**Inner-image autocrop bounds are NOT in the schema.** The encoder
detects the inner ultrasound image (depth ruler / margins / tick row
stripped) from frame pixels at encode time -- see "Inner-image
autocrop" below. Keeping the bounds out of the sidecar means a
detector tweak only requires a re-encode (offline), not a re-extract
(Admin EchoWave).

**Datasets:**
- `/timing/frame_idx_1n` -- int32 (N,)
- `/timing/time_ms` -- float64 (N,), frame 0 anchored at 0.0 ms
- `/timing/ifi_ms` -- float64 (N,), inter-frame intervals
- `/frames/gray` -- uint8 (N, H, W), full-frame display capture
  (omit by passing `frames=False` for a fast timing-only extract)

### Schema history

| version | date | change |
|---|---|---|
| v1 | 2026-05-24 | initial public release. Consolidates the
in-development `v1a{1..5}` series (single ROI / `params` sweep /
multi-ROI / display-scale capture) into one labelled baseline. |

`Log` reads both the public `v1` label and the legacy in-development
labels (`v1a{1..5}`) transparently, so on-disk sidecars produced by
pre-graduation pipelines keep loading. Production extracts always
write `v1`. The inner-image autocrop is computed at encode time and
doesn't bump the schema -- existing on-disk sidecars get autocropped
mp4s "for free" the next time `export_video` runs over them.

## Inner-image autocrop

The panel ROI from AutoInt1's `GetUltrasoundX{1,2}`/`Y{1,2}` is the
*full* B-mode panel: depth ruler + side margins + inner ultrasound
image + bottom-tick row. The depth ruler alone eats ~25% of the
panel width on single-probe acquisitions, and the bottom-tick row
pollutes the bottom with sharp non-anatomical contrast that confuses
DLC. We don't want either in the per-panel mp4.

`export_video` runs a content-based detector against an aggregate of
16 evenly-spaced frames per panel and crops each mp4 to the inner
ultrasound image. Detection failures fall back to the outer panel
ROI with a warning. `crop="panel"` opts out explicitly.

**Why at encode time, not extract time?** Re-extraction requires
Admin EchoWave + Admin Python. The detector is offline (just needs
`/frames/gray`), and encoding is the slow stage anyway -- ~50 ms of
detection vs. minutes of h265 lossless. Putting detection here
means detector tweaks only need a re-encode, not a re-extract.

### Algorithm

* **Cols.** Estimate the EchoWave UI gray level from the panel's
  leftmost+rightmost five cols (taking the median to ignore the
  depth-ruler "0" digit's bright pixels). A column is "margin" when
  its mean is within ~12 of that gray AND its vertical std is
  low (~12); the longest contiguous run of non-margin cols is the
  inner image width. Tick rows are pre-trimmed for this pass so a
  saturated bright stripe doesn't poison the per-col std.
* **Rows.** Walk up from the last panel row, peeling off rows whose
  col-restricted mean exceeds the tick threshold
  (`max(60, 2*median + 20)`). Top edge of the inner image is the
  panel top (probed Telemed configurations place the depth-ruler "0"
  digit *above* the panel ROI's `y1`).

### Why content-based detection (not a probe-aperture lookup)

The probe-table approach (predict inner width from
`probe_name` + aperture_mm + `image_d{x,y}_cm_per_px`) was
sketched + dismissed 2026-05-24: predicted width was ~4-5% off
observed (670 vs 700 px on the LF9-5N60-A3 probe), meaning a
per-probe empirical correction table would be needed anyway. The
content-based detector self-calibrates from the actual pixels and
generalises to unfamiliar probes without a maintenance table.

For a fixed (probe, depth, view_area, panel_dims), the detected
inner ROI is deterministic across recordings of the same
acquisition config -- a useful invariant for cross-file consistency
audits.

### Failure mode: fully-black / degenerate recordings

`_detect_image_roi` returns `None` when the gray-margin step isn't
clear (no detectable inner-image bounds, or the resulting box is
< 20% of the panel area on either axis). The encoder falls back to
the panel ROI with a warning so the regression is visible.

## Multi-probe auto-detection

A dual-probe acquisition lights up two B-mode panels side-by-side
(B + B2 in AutoInt1's enum, img_id=1 + img_id=2). The authoritative
detection signal is **the ROI enumeration itself**: `_collect_b_mode_rois`
probes img_ids 1-4 and keeps every panel that returns a positive-area
rectangle. The number of populated ROIs = number of physical probes
in use.

**Why not `scanning_state` (id 200)?** It's a useful sanity-check but
has undocumented sub-states. The 2026-05-24 pia02 probe reported
state=25, which isn't in any documented `id_state_bb_*` constant.
The ROI count never ambiguates.

`export_video` follows: single-probe -> `<stem>.mp4`; dual-probe ->
`<stem>_b1.mp4` + `<stem>_b2.mp4`.

### Inactive-panel sentinel: `(0, 0, 0, 0)`

AutoInt1 returns the zero-rect sentinel `(x1, x2, y1, y2) = (0, 0, 0, 0)`
for inactive img_ids rather than raising. The 2026-05-24 metadata
probes on usl02 (single-probe) revealed B2/B3/B4 all coming back as
zero-rect. The `TelemedRoi.from_cmd` validator rejects anything that
isn't a strict positive-area rectangle (`x2 > x1` AND `y2 > y1`),
catching both the sentinel and any inverted/negative rect. Without
that fix, single-probe recordings would have been mis-classified as
quad-probe and produced four `_b{N}.mp4` files (three of them
degenerate).

## Encode pipeline: lossless h265 mono, ultrafast preset

### Why lossless

The source `/frames/gray` is uint8 grayscale straight off the
beamformer -- no upstream lossy step to reclaim quality from. A
CRF-tuned encode is buying file size at the cost of DLC accuracy that
the device gave us for free. The cropped per-panel ROI is small
enough (~700x550) that lossless h265 lands at ~3 GB per 20k-frame
recording, tolerable at corpus scale.

`export_video(..., lossless=False, crf=N)` is available as an opt-in
for users who'd rather trade a few percent of accuracy for ~50x
smaller files.

### Why `preset="ultrafast"`

For *lossless* h265 the preset only trades file size against encode
+ decode speed -- reconstructed pixels are bit-exact regardless.
`ultrafast` wins on every time axis (encode + linear decode + random
seek + TOC build) at +6% size over `fast`, ~+15% over `slow`. Full
bench table + methodology + raw numbers in
[`BENCHMARKING.md`](BENCHMARKING.md). DLC inference is GPU-bound
(~175 fps ceiling), so encoder preset is effectively neutral for
inference; the decode + seek wins are what matter for interactive
use (DUSTrack labeling, frame scrubbing).

**Accuracy invariance**: all four lossless presets produce DLC
predictions identical within float32 noise (max |delta| < 1e-4 px).
The lossy legacy h264/yuv420p path costs ~0.58 px median DLC keypoint
error and outliers up to 41 px -- the lossless h265 mono pipeline
closes that gap. Parity bench data in
[`BENCHMARKING.md`](BENCHMARKING.md).

**Dustrack UI fps**: lossless h265 mono is ~22% faster in the
dustrack UI than legacy h264 yuv420p (42 fps vs 34 fps on the
encoding-axis adapter probe). Within the lossless presets the choice
is noise (~5% spread). Stacks with the existing fast_render Tier 2
architectural 3.94x gain from `dustrack/BENCHMARKING.md`.

**Power-user overrides:**
- `preset="slow"` for the smallest lossless files (at ~7x slower
  encode + ~2x slower decode; accuracy-equivalent).
- `lossless=False, crf=22` for the smallest files at the ~0.6 px
  accuracy penalty (~50x smaller than lossless).

### CPU vs GPU

We stick with CPU encoding (libx265). GPU options were investigated
2026-05-24 and rejected:

- **NVENC h264** has `-preset lossless` / `losslesshp`, but h264
  lossless of monochrome content runs 2-3x larger than h265 lossless,
  and NVENC typically converts `gray` to `yuv420p` with synthetic
  chroma planes (defeats the chroma-free pipeline).
- **NVENC h265** lossless support is inconsistent across drivers and
  often not bit-exact.
- Other GPU paths (AMD VCN, Intel QSV) have similar limitations.

`h264_nvenc -preset slow -rc constqp -qp 0` is fast and visually
indistinguishable from lossless for spot-checks, but it's not
bit-exact and uses yuv420p -- not a production output.

### TOC creation (datanavigator + PyAV)

`datanavigator.VideoReader` (the PyAV+TOC backend used by DUSTrack /
datanavigator) builds a packet-level index the first time it opens an
mp4 and caches it as `<mp4>.dnav-toc` next to the video. TOC build
time follows the same preset curve as decode speed -- ultrafast ~2x
faster than slow. See [`BENCHMARKING.md`](BENCHMARKING.md) for raw
numbers.

## Orientation normalisation

EchoWave operators can toggle scan-direction (L/R flip) and rotation
per machine. If different machines in a cohort save with different
orientations, the same anatomy appears mirrored/rotated across
recordings -- catastrophic for cohort-wide DLC training.

The schema captures the L/R-flip state via `b_is_scan_direction_changed`
(AutoInt1 id 133, bool) and rotation via `b_rotate` (id 132, int).
`export_video` applies `-vf hflip` when the flip flag is True, so
every cohort mp4 lands in a canonical orientation regardless of
which operator toggled what on which machine. `normalize_orientation=False`
disables this for power-user inspection.

### Known limitation: U/D flip

AutoInt1's `id_b_flip_up_down` (105) is action-only -- there's no
companion getter. So U/D flip cannot be detected from the sidecar.

**Mitigation**: lock down the acquisition SOP ("never toggle U/D
flip") and visually spot-check representative frames per machine
during cohort onboarding. If a U/D mismatch is found, the operator
needs to either re-acquire or manually flag the affected recordings.

### Known limitation: rotation enum

`b_rotate` returns an int (0/1/2/3? or actual degrees?) -- the
AutoInt1 docs don't specify the mapping. Both probed cohorts
(usl02, pia02) report 0. When a non-zero `b_rotate` is encountered,
`_orientation_vf` warns + leaves the pixels untouched; the user
should investigate and update the function once the mapping is
verified against a deliberately-rotated recording.

## Spatial scale: physical (beamformer) vs image (display)

There are **two scales** to be aware of, and they differ by ~2% on
typical Telemed acquisitions:

| Attribute | Returns | Use for... |
|---|---|---|
| `Log.physical_dx_cm_per_px` | beamformer-native sample spacing (from `GetUltrasoundPhysicalDeltaX`) | hardware provenance; not measurement |
| `Log.image_dx_cm_per_px` | display scale -- `b_depth_mm / 10 / panel_height_px` | cm conversions on tracked-point coords |

The reported `physical_dx` is the beamformer's native radial sample
spacing -- a function of the ADC clock and the assumed speed of sound.
But EchoWave renders the resulting image onto a display frame whose
height is laid out to match the operator-selected depth setting (the
depth ruler IS the trusted calibration). For a 50 mm depth setting
on a 558 px panel:

- `physical_dy_cm_per_px` = 0.009166 cm/px (=> 558 × dy = 5.11 cm)
- `image_dy_cm_per_px` = 0.05 cm / 558 px = 0.00896 cm/px (=> 5.00 cm)

Per Telemed support: trust the depth setting. So `image_d{x,y}`
is what you want for spatial measurements on DLC keypoints. Both
attributes return `None` for v1 sidecars that lack `params["b_depth"]`.

The display x scale is assumed equal to the y scale (`image_dx == image_dy`)
because Telemed renders with square display pixels (1:1 aspect so
anatomy doesn't squish) and AutoInt1 reports `physical_dx == physical_dy`
on every probed acquisition. If a future probe breaks this
assumption it'll surface as anatomy rendered with a non-1:1 aspect in
`Log.view()`; revisit then.

## Timing

`/timing/time_ms` carries true per-frame timestamps at the device's
native ~100 ns precision (Telemed acquisitions are VFR -- the inter-
frame interval varies frame to frame around the mean fps; 50% of
frames land more than 1 ms off the mean). DLC and cv2 index by
frame number, so the encoded mp4 declares CFR at `mean_fps` -- but
**the .tvd.h5 is the source of truth for real time**. Downstream
analysis converting tracked points back into the OT clock should
round-trip via `Log.time_ms[frame_idx]`, not the mp4's frame rate.

There is **no timing CSV sidecar**; an earlier design considered one,
but it duplicates data already in the .tvd.h5 (and would risk
desyncing).

### Last-frame outlier

Both DICOM `FrameTimeVector` and the COM-extracted IFI show a ~0.078
ms inter-frame interval as the final entry on every recording --
recording-termination artifact (compound sub-frame). Harmless if
downstream sync work either drops the last frame or weights by IFI.

## Per-session setup (Administrator EchoWave + Administrator Python)

The h5 stage (`export_h5` / `process`) requires:

0. **One-time per machine**: install **Echo Wave II** (the Telemed
   vendor application). The AutoInt1 COM server ships inside it; the
   `Config\Plugins` folder referenced below is created by the
   installer. There is nothing to download from this package for it.

1. **One-time per machine**: register the COM ProgID. From an
   Administrator PowerShell:
   ```
   cd "C:\Program Files\Telemed\Echo Wave II Application\EchoWave II\Config\Plugins"
   .\AutoInt1_regasm.bat
   ```
   You should see "Types registered successfully". Without this,
   `GetActiveObject('EchoWave2.CmdInt1')` raises "Invalid class
   string". (The exact install path can vary by EchoWave version /
   install location — if `Config\Plugins` isn't under that path,
   search the EchoWave install dir for `AutoInt1_regasm.bat`.)

2. **Per session**: start EchoWave II as Administrator (right-click ->
   "Run as administrator").

3. **Per session**: start your Python (or terminal) as Administrator.
   COM ROT publication is per-elevation; a non-elevated client can't
   see an elevated server.

The video stage (`export_video`) is offline and runs in any Python.

### Network drives

EchoWave's `OpenFile` fails on UNC / mapped network paths in our
setup. `export_h5` auto-detects non-C: drives + UNC prefixes, copies
each `.tvd` to a local temp dir, processes there, and copies the
resulting `.h5` back to the source folder. A `ThreadPoolExecutor`
overlaps the network copies with the COM extraction
(stage N+1 + unstage N-1 run on workers while the COM thread handles
N).

## Implementation notes

### Pipe deadlock (libx265 + Popen)

libx265 prints per-frame statistics to stderr by default. Reading
frames from stdin via `subprocess.Popen(..., stderr=PIPE)` deadlocks
once the stderr buffer (~64 KB) fills -- the encoder blocks waiting
for the reader, the Python frame loop blocks waiting for the encoder.
The cmd builder mandates `-hide_banner -loglevel error` so stderr
stays small enough to never fill.

### pywin32 + .NET CCW: zero-arg methods are properties

AutoInt1's zero-argument calls (`GetFramesCount`, `GetCurrentFrameTime`,
etc.) expose as **properties** under pywin32's late binding, not
callables. Drop the parens:

```python
n = cmd.GetFramesCount     # works
n = cmd.GetFramesCount()   # TypeError
```

Methods with arguments use normal parens. This bit the initial probe
script; it's wrapped inside the module now.

## Related files

- [`BENCHMARKING.md`](BENCHMARKING.md) -- preset bench tables (encode /
  decode / seek), DLC accuracy parity data (lossless h265 mono vs
  legacy h264 yuv420p), and TOC build numbers.
- `_metadata_probe.py` -- parses `AutoInt1Client.txt` to classify all
  ~378 documented ids by extraction strategy; the source-of-truth for
  the `_PARAM_SPECS` curation.

## Acknowledgments

This package was developed as part of the ImmersionToolbox initiative
at the [MIT.nano Immersion Lab](https://immersion.mit.edu). Thanks to
NCSOFT for supporting this initiative.
