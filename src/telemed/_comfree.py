"""Fully COM-free ``.tvd`` extraction (no EchoWave, no Admin, no per-frame COM walk).

This is the parallel of :mod:`telemed._extract` (the COM path) for the case where we do
*not* want EchoWave's scan-converted display pixels but the device's **raw acoustic frames**
straight out of the container, plus all the per-recording metadata that the COM path snapshots
via AutoInt1 -- everything read directly from the ``.tvd`` bytes.

What it produces per ``.tvd`` (into ``out_dir``):

* ``<stem>.tvd.h5`` -- **metadata + timing only, NO ``/frames/gray``**. The image data is *not*
  duplicated into the h5; it lives in the mp4(s) instead. Timing is the COM-free
  ``time_ms`` (== ``time_ms_comfree``, the full *declared* frame set; see
  :func:`telemed.read_tvd_time_ms`). Root attrs carry the parsed acquisition parameters and the
  per-probe native geometry / pixel scale.
* ``<stem>.mp4`` (single probe) or ``<stem>_b1.mp4`` + ``<stem>_b2.mp4`` (dual probe) -- the raw
  acoustic frames, **native pixel grid** (``lines x active_depth``, depth vertical), no
  interpolation and no gamma. A Sample-Aspect-Ratio (SAR) tag is written so players show the true
  anatomical aspect while the stored pixels stay native; downstream point-tracking works on the
  native grid + the per-axis cm/px in the h5.

The container layout + the timing model are documented in :mod:`telemed._extract` (the ``UIFF``
walk, the ``00bb``/``01bb`` frame chunks with their 48-byte sub-header, the end-tick at off 24).
Geometry (``lines``/``samples``/``active``) varies per recording and is read from ``strf``; never
assume a line count. Image processing note: EchoWave's spatial compounding + speckle filtration +
enhancement are applied in the acoustic domain *before* the cine is stored, so the raw frames here
already carry them; only the display gray-mapping (DR window / palette) and the scan-conversion
resampling are display-side and intentionally omitted.
"""

from __future__ import annotations

import struct
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from ._extract import _log, _normalize_sources, _sidecar_h5_path, read_tvd_n_frames

# ---------- container constants (mirror _extract) ----------

_FORM_MAGIC = b"UIFF"
_CONTAINER_IDS = (b"UIFF", b"LIST")
_B_STREAM = b"00bb"   # stream 0 = primary B panel  (-> _b1)
_B2_STREAM = b"01bb"  # stream 1 = second probe     (-> _b2)
_FRAME_STREAMS = (_B_STREAM, _B2_STREAM)
_FRAME_HDR = 48          # per-frame sub-header bytes
_FRAME_DATASIZE_OFF = 0x0C
_TICK_OFF = 24
_TICKS_PER_MS = 10000.0
_READ_BUFFER = 8 * 1024 * 1024   # large buffered reads -> good sequential network throughput


# ---------- geometry from strf ----------


@dataclass
class _Geom:
    lines: int          # lateral scan-line (beam) count
    samples: int        # total depth samples per line
    active: int         # imaged (displayed) depth samples (<= samples)
    datasize: int       # bytes per frame payload = lines * samples (8-bit)


def _read_header(path: Path, n: int = 1 << 16) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def _walk_mem(buf: bytes, start: int, end: int, depth: int, out: list) -> None:
    i = start
    while i + 12 <= end:
        cid = buf[i:i + 4]
        if not all(32 <= b < 127 for b in cid):
            return
        size = int.from_bytes(buf[i + 4:i + 12], "little")
        pay = i + 12
        out.append((cid, pay, size))
        if cid in _CONTAINER_IDS and depth < 8:
            _walk_mem(buf, pay + 4, min(pay + size, end), depth + 1, out)
        i = pay + size + (size & 1)


def read_geometry(path: Union[str, Path]) -> Optional[_Geom]:
    """Per-recording acoustic geometry, read COM-free from the ``strf`` header.

    ``samples``/``lines`` come from the consecutive ``[8(bits), n_samples, total_bytes]`` triad
    (``lines = total_bytes / n_samples``); ``active`` (the imaged depth window) is the strf value in
    ``[samples/2, samples)`` that recurs. Returns ``None`` if the file has no readable strf/frame.
    """
    buf = _read_header(Path(path))
    if buf[:4] != _FORM_MAGIC:
        return None
    nodes: list = []
    _walk_mem(buf, 0, len(buf), 0, nodes)
    strf = next((n for n in nodes if n[0] == b"strf"), None)
    if strf is None:
        return None
    # Derive geometry from the strf ``[8(bits), n_samples, total_bytes]`` triad -- NOT from a frame
    # chunk: the first ``00bb`` can sit well past the header window on deeper recordings (the strf is
    # always early). ``total_bytes`` (= per-frame datasize) factors as n_samples * n_lines.
    body = buf[strf[1] + 4:strf[1] + strf[2]]
    g = struct.unpack_from("<%dI" % (len(body) // 4), body, 0)
    lines = samples = datasize = None
    for k in range(len(g) - 2):
        if g[k] != 8:
            continue
        s_, b_ = g[k + 1], g[k + 2]
        if s_ and b_ and b_ % s_ == 0:
            l_ = b_ // s_
            if 256 <= s_ <= 4096 and 16 <= l_ <= 1024 and 4096 <= b_ <= 8_000_000:
                samples, datasize, lines = s_, b_, l_
                break
    if samples is None:                          # fallback: read datasize off the first frame chunk
        f0 = next((n for n in nodes if n[0] == _B_STREAM), None)
        if f0 is None:
            return None
        datasize = struct.unpack_from("<I", buf, f0[1] + _FRAME_DATASIZE_OFF)[0]
        samples = 1024 if datasize % 1024 == 0 else 512
        lines = datasize // samples
    from collections import Counter
    cand = Counter(v for v in g if samples // 2 <= v < samples)
    active = max((v for v, c in cand.items() if c >= 2), default=samples)
    return _Geom(lines=lines, samples=samples, active=active, datasize=datasize)


# ---------- .NET BinaryFormatter mini-reader (acquisition metadata blocks) ----------
#
# The trailing UIFF blocks are .NET BinaryFormatter serializations. We only need a handful of
# primitive fields (UsgHWSettings acquisition params), two strings (probe / beamformer name) and a
# couple of System.Drawing.Rectangle structs (display image aspect). The reader below walks one
# named ClassWithMembersAndTypes record and returns {member_suffix: value}; rect members come back
# as {'x','y','width','height'} dicts. Best-effort: any field we can't reach is simply absent.

_PRIM = {1: 1, 2: 1, 6: 8, 7: 2, 8: 4, 9: 8, 10: 1, 11: 4, 13: 8, 14: 2, 15: 4, 16: 8}


def _r7(buf: bytes, i: int) -> tuple[int, int]:
    n = s = 0
    while True:
        b = buf[i]; i += 1
        n |= (b & 0x7F) << s
        if not (b & 0x80):
            break
        s += 7
    return n, i


def _rstr(buf: bytes, i: int) -> tuple[str, int]:
    n, i = _r7(buf, i)
    return buf[i:i + n].decode("utf-8", "replace"), i + n


def _rprim(buf: bytes, i: int, pt: int):
    if pt == 1:  return bool(buf[i]), i + 1
    if pt == 8:  return struct.unpack_from("<i", buf, i)[0], i + 4
    if pt == 9:  return struct.unpack_from("<q", buf, i)[0], i + 8
    if pt == 6:  return struct.unpack_from("<d", buf, i)[0], i + 8
    if pt == 11: return round(struct.unpack_from("<f", buf, i)[0], 5), i + 4
    if pt == 7:  return struct.unpack_from("<h", buf, i)[0], i + 2
    if pt == 2:  return buf[i], i + 1
    if pt == 10: return struct.unpack_from("<b", buf, i)[0], i + 1
    if pt == 14: return struct.unpack_from("<H", buf, i)[0], i + 2
    if pt == 15: return struct.unpack_from("<I", buf, i)[0], i + 4
    if pt in (16, 13): return struct.unpack_from("<Q", buf, i)[0], i + 8
    raise ValueError(f"primEnum {pt}")


def _read_class_layout(buf: bytes, i: int, has_lib: bool, layouts: dict):
    """Parse a ClassWithMembersAndTypes / SystemClassWithMembersAndTypes body starting at the
    ObjectId. Returns (objId, names, btypes, addl, value_start_i). Caches by objId in ``layouts``."""
    objid = struct.unpack_from("<i", buf, i)[0]; i += 4
    _name, i = _rstr(buf, i)
    mc = struct.unpack_from("<i", buf, i)[0]; i += 4
    names = []
    for _ in range(mc):
        s, i = _rstr(buf, i); names.append(s.split(">")[-1])
    btypes = list(buf[i:i + mc]); i += mc
    addl = []
    for bt in btypes:
        if bt in (0, 7): addl.append(buf[i]); i += 1
        elif bt == 3:    s, i = _rstr(buf, i); addl.append(s)
        elif bt == 4:    s, i = _rstr(buf, i); i += 4; addl.append(s)
        else:            addl.append(None)
    if has_lib:
        i += 4  # LibraryId
    layouts[objid] = (names, btypes, addl)
    return objid, names, btypes, addl, i


def _read_values(buf: bytes, i: int, names, btypes, addl, layouts: dict, out: dict):
    """Read the member values for one class instance; fills ``out`` with what we can decode."""
    for k in range(len(names)):
        bt = btypes[k]; nm = names[k]
        if bt == 0:                                    # primitive inline
            v, i = _rprim(buf, i, addl[k]); out[nm] = v
        elif bt == 1:                                  # string -> a record
            i, v = _read_obj(buf, i, layouts); out[nm] = v
        else:                                          # object / class / array -> a record
            i, v = _read_obj(buf, i, layouts)
            out[nm] = v
    return i


def _read_obj(buf: bytes, i: int, layouts: dict):
    """Read one value-position record (string / reference / null / inline struct / library).
    Returns (new_i, value). Inline rect-like structs come back as a {member: value} dict."""
    tag = buf[i]; i += 1
    if tag == 12:                                      # BinaryLibrary -> skip, then real record
        i += 4
        _lib, i = _rstr(buf, i)
        return _read_obj_at(buf, i, layouts)           # next tag is the real record
    return _read_obj_at(buf, i, layouts, tag)


def _read_obj_at(buf: bytes, i: int, layouts: dict, tag: Optional[int] = None):
    if tag is None:
        tag = buf[i]; i += 1
    if tag == 10:                                      # ObjectNull
        return i, None
    if tag == 6:                                       # BinaryObjectString
        i += 4
        s, i = _rstr(buf, i)
        return i, s
    if tag == 9:                                       # MemberReference
        i += 4
        return i, "<ref>"
    if tag == 5:                                       # ClassWithMembersAndTypes (inline)
        objid, names, btypes, addl, i = _read_class_layout(buf, i, True, layouts)
        sub: dict = {}
        i = _read_values(buf, i, names, btypes, addl, layouts, sub)
        return i, sub
    if tag == 4:                                       # SystemClassWithMembersAndTypes (inline)
        objid, names, btypes, addl, i = _read_class_layout(buf, i, False, layouts)
        sub = {}
        i = _read_values(buf, i, names, btypes, addl, layouts, sub)
        return i, sub
    if tag == 1:                                       # ClassWithId (reuse cached layout)
        objid = struct.unpack_from("<i", buf, i)[0]; i += 4
        metaid = struct.unpack_from("<i", buf, i)[0]; i += 4
        lay = layouts.get(metaid)
        if lay is None:
            raise ValueError(f"ClassWithId meta {metaid} not cached")
        names, btypes, addl = lay
        sub = {}
        i = _read_values(buf, i, names, btypes, addl, layouts, sub)
        return i, sub
    raise ValueError(f"unhandled record tag {tag}")


def _parse_named_class(buf: bytes, classname: bytes) -> dict:
    """Find the ``classname`` ClassWithMembersAndTypes def (rt=5) and decode its member values.
    Returns {member_suffix: value}; best-effort (stops cleanly at the first record it can't read)."""
    pat = bytes([len(classname)]) + classname
    occ, s = [], 0
    while True:
        k = buf.find(pat, s)
        if k < 0:
            break
        occ.append(k); s = k + 1
    defs = [k for k in occ if k >= 5 and buf[k - 5] == 5]
    if not defs:
        return {}
    layouts: dict = {}
    # record is [rt=5][ObjectId int32][name-len byte][name...]; defs[0] points at the name-len
    # byte, so the ObjectId starts 4 bytes earlier.
    objid, names, btypes, addl, i = _read_class_layout(buf, defs[0] - 4, True, layouts)
    out: dict = {}
    try:
        _read_values(buf, i, names, btypes, addl, layouts, out)
    except Exception:  # noqa: BLE001 -- partial decode is fine; we keep what we got
        pass
    return out


# Acquisition params we lift into the h5 (UsgHWSettings primitive members; suffix-matched).
_WANT_PARAMS = (
    "b_depth", "b_gain", "b_power", "b_dynamic_range", "b_frequency", "b_rejection",
    "b_scan_type", "b_view_area", "b_steering_angle", "b_trapezoid_angle", "b_zoom_factor",
    "b_compound_frames_number", "b_compound_angle", "b_frame_averaging", "b_multibeam",
    "b_image_enhancement", "b_image_enhancement_enabled",
    "b_speckle_filtration", "b_speckle_filtration_enabled",
    "b_speckle_filtration_before_compound", "b_speckle_filtration_enabled_before_compound",
    "b_palette_gamma", "b_palette_brightness", "b_palette_contrast", "b_palette_negative",
    "b_lines_density", "b_high_lines_density",
)


@dataclass
class _Meta:
    params: dict = field(default_factory=dict)          # b_* acquisition settings (param_* attrs)
    probe_name: Optional[str] = None
    beamformer_name: Optional[str] = None
    cine_datetime: Optional[str] = None
    image_w: Optional[int] = None                       # display image-rect dims (for aspect/SAR)
    image_h: Optional[int] = None
    scan_dir_changed: bool = False                      # b_is_scan_direction_changed -> hflip


def read_metadata(path: Union[str, Path], max_bytes: int = 16 << 20) -> _Meta:
    """Read acquisition metadata COM-free from the trailing .NET blocks. Best-effort.

    The three ``.NET``-serialized blocks (Img / Usg / Patient) sit *after* the data ``UIFF`` form,
    so we seek to its end (from the header size field) and read the whole metadata region -- the
    Img block carries an embedded PNG poster, so a fixed tail window can fall short of its record
    start. Capped at ``max_bytes`` as a guard."""
    p = Path(path)
    sz = p.stat().st_size
    with open(p, "rb") as f:
        head = f.read(12)
        meta_start = sz
        if head[:4] == _FORM_MAGIC:
            data_size = int.from_bytes(head[4:12], "little")
            meta_start = 12 + data_size + (data_size & 1)
        if not (0 < meta_start < sz):
            meta_start = max(0, sz - 600_000)
        f.seek(meta_start)
        buf = f.read(min(max_bytes, sz - meta_start))
    m = _Meta()
    hw = _parse_named_class(buf, b"EchoWave.UsgHWSettings")
    for k in _WANT_PARAMS:
        if k in hw and not isinstance(hw[k], (dict, str)):
            m.params[k] = hw[k]
    if "b_is_scan_direction_changed" in hw:
        m.scan_dir_changed = bool(hw["b_is_scan_direction_changed"])
    usg = _parse_named_class(buf, b"EchoWave.ImageFileDataUsg")
    if isinstance(usg.get("current_probe_name"), str):
        m.probe_name = usg["current_probe_name"]
    if isinstance(usg.get("current_beamformer_name"), str):
        m.beamformer_name = usg["current_beamformer_name"]
    img = _parse_named_class(buf, b"EchoWave.ImageFileDataImg")
    rect = img.get("image_rect") or img.get("original_image_rect")
    if isinstance(rect, dict) and "width" in rect and "height" in rect:
        try:
            m.image_w = int(rect["width"]); m.image_h = int(rect["height"])
        except (TypeError, ValueError):
            pass
    pat = _parse_named_class(buf, b"EchoWave.ImageFileDataPatient")
    for key in ("file_save_time", "exam_start_dt"):
        if isinstance(pat.get(key), str):
            m.cine_datetime = pat[key]; break
    return m


# ---------- streaming extract (constant memory; handles the 17 GB files) ----------


@dataclass
class _Probe:
    suffix: str          # "" (single) / "_b1" / "_b2"
    stream: bytes
    proc: Any = None
    n: int = 0


def _ffmpeg_cmd(out_path: Path, *, w: int, h: int, fps: float, lossless: bool, crf: int,
                preset: str, sar: Optional[float], hflip: bool, overwrite: bool) -> list:
    """h265-mono ffmpeg argv reading native raw-gray frames from stdin (mirrors _encode)."""
    quality = ["-x265-params", "lossless=1"] if lossless else ["-crf", str(crf)]
    vf = []
    if hflip:
        vf.append("hflip")
    if sar:
        vf.append(f"setsar={sar:.5f}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y" if overwrite else "-n",
           "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
           "-i", "-", "-c:v", "libx265", "-pix_fmt", "gray", *quality, "-preset", preset,
           "-fps_mode", "cfr", "-an"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd.append(str(out_path))
    return cmd


def extract_one(
    tvd: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    lossless: bool = True,
    crf: int = 24,
    preset: str = "ultrafast",
    overwrite: bool = False,
    progress: bool = True,
) -> dict:
    """COM-free extract of one ``.tvd`` -> per-probe mp4(s) + a metadata/timing ``.tvd.h5``.

    Streams the container once (constant memory) dispatching each ``00bb``/``01bb`` raw frame to
    its probe's ffmpeg pipe (native ``lines x active`` grid, depth-vertical, no gamma; SAR tags the
    true display aspect; ``hflip`` when the recording's scan-direction was toggled). Writes the h5
    with ``time_ms`` (COM-free, all declared frames), the parsed acquisition params, and per-probe
    geometry + pixel scale. Returns a per-file timing dict.
    """
    import h5py
    import numpy as np

    tvd = Path(tvd); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stem = tvd.name[:-4] if tvd.name.lower().endswith(".tvd") else tvd.stem
    t0 = time.perf_counter()

    geom = read_geometry(tvd)
    if geom is None:
        raise RuntimeError(f"{tvd.name}: no readable strf/frame geometry (not a B-mode .tvd?)")
    meta = read_metadata(tvd)

    # probe count from the strh stream-type records (one per B stream: "2DusB", "2DusB2", ...).
    # These live in the first few KB, unlike the first 01bb frame chunk which can sit past 64 KB.
    hdr = _read_header(tvd)
    hdr_nodes: list = []
    _walk_mem(hdr, 0, len(hdr), 0, hdr_nodes)
    n_probes = sum(1 for c, pay, _ in hdr_nodes if c == b"strh" and hdr[pay:pay + 4] == b"2Dus")
    n_probes = max(1, min(n_probes, len(_FRAME_STREAMS)))
    present = list(_FRAME_STREAMS[:n_probes])
    dual = len(present) > 1
    probes = {
        s: _Probe(suffix=(f"_b{j + 1}" if dual else ""), stream=s)
        for j, s in enumerate(present)
    }

    # display aspect -> SAR so native pixels show with true anatomy. Panel aspect = (image_w /
    # n_probes) / image_h; SAR = panel_aspect * (active / lines). Fall back to None (native shown
    # as-is) if the rect wasn't parseable or looks implausible.
    sar = None
    if meta.image_w and meta.image_h:
        panel_aspect = (meta.image_w / max(1, len(present))) / meta.image_h
        cand = panel_aspect * (geom.active / geom.lines)
        if 1.5 <= cand <= 30:
            sar = cand

    # timing (COM-free, declared frames) + mean fps
    from ._extract import read_tvd_time_ms
    time_ms = read_tvd_time_ms(tvd, cache=True)
    n_decl = int(len(time_ms)) if time_ms is not None else None
    if time_ms is not None and len(time_ms) > 1 and time_ms[-1] > 0:
        fps = (len(time_ms) - 1) / (time_ms[-1] / 1000.0)
    else:
        fps = 30.0

    # open one ffmpeg per probe (skip-existing aware)
    out_paths = {s: out_dir / f"{stem}{pr.suffix}.mp4" for s, pr in probes.items()}
    for s, pr in probes.items():
        cmd = _ffmpeg_cmd(out_paths[s], w=geom.lines, h=geom.active, fps=fps, lossless=lossless,
                          crf=crf, preset=preset, sar=sar, hflip=meta.scan_dir_changed,
                          overwrite=overwrite)
        pr.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # ---- single streaming pass over the data UIFF, dispatching frames ----
    active = geom.active
    ds = geom.datasize
    lines, samples = geom.lines, geom.samples
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=(n_decl or 0) * len(probes), desc=stem, unit="frame", leave=False)
        except ImportError:
            bar = None

    def _dispatch(stream: bytes, payload: bytes):
        pr = probes.get(stream)
        if pr is None:
            return
        fr = np.frombuffer(payload, np.uint8, count=ds).reshape(lines, samples).T[:active]
        pr.proc.stdin.write(np.ascontiguousarray(fr).tobytes())
        pr.n += 1
        if bar is not None:
            bar.update(1)

    err = None
    try:
        with open(tvd, "rb", buffering=_READ_BUFFER) as f:   # big buffer -> good network throughput
            if f.read(4) != _FORM_MAGIC:
                raise RuntimeError("bad magic")
            data_size = int.from_bytes(f.read(8), "little")  # outer UIFF (the data form)
            f.read(4)                                        # form type ("UDI ")
            _stream_frames(f, f.tell(), 12 + data_size, _dispatch)
    except Exception as e:  # noqa: BLE001
        err = e
    finally:
        if bar is not None:
            bar.close()
        for pr in probes.values():
            if pr.proc and pr.proc.stdin:
                pr.proc.stdin.close()
        for pr in probes.values():
            if pr.proc:
                pr.proc.wait()
    if err is not None:
        raise err
    for s, pr in probes.items():
        if pr.proc.returncode != 0:
            se = (pr.proc.stderr.read() or b"").decode("utf-8", "replace")
            raise RuntimeError(f"ffmpeg failed for {out_paths[s].name}: {se}")

    # ---- write the metadata/timing h5 (NO frames) ----
    axial_cm = ((meta.params.get("b_depth", 0) / 10.0) / active) if meta.params.get("b_depth") else None
    h5_path = _sidecar_h5_path(out_dir / tvd.name)   # <stem>.tvd.h5 in out_dir
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["schema_version"] = "comfree-v1"
        h5.attrs["backend"] = "comfree"
        h5.attrs["source_tvd_path"] = str(tvd)
        h5.attrs["extracted_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        h5.attrs["n_frames"] = int(n_decl or min((p.n for p in probes.values()), default=0))
        h5.attrs["n_b_images"] = len(probes)
        h5.attrs["src_fps"] = float(fps)
        declared = read_tvd_n_frames(tvd)
        if declared is not None:
            h5.attrs["tvd_declared_n_frames"] = int(declared)
        # Log-compat nominal full-frame dims (no real display frame; native of probe 1)
        h5.attrs["full_frame_width"] = int(geom.lines)
        h5.attrs["full_frame_height"] = int(geom.active)
        if meta.probe_name:      h5.attrs["param_probe_name"] = meta.probe_name
        if meta.beamformer_name: h5.attrs["param_beamformer_name"] = meta.beamformer_name
        if meta.cine_datetime:   h5.attrs["param_cine_end_datetime_str"] = meta.cine_datetime
        for k, v in meta.params.items():
            h5.attrs[f"param_{k}"] = v
        # per-probe native geometry + the mp4 it lives in + pixel scale
        for j, (s, pr) in enumerate(probes.items(), start=1):
            h5.attrs[f"probe{j}_stream"] = s.decode()
            h5.attrs[f"probe{j}_video"] = out_paths[s].name
            h5.attrs[f"probe{j}_lines"] = int(lines)
            h5.attrs[f"probe{j}_samples"] = int(samples)
            h5.attrs[f"probe{j}_active"] = int(active)
            h5.attrs[f"probe{j}_n_frames"] = int(pr.n)
            if axial_cm:
                h5.attrs[f"probe{j}_axial_cm_per_px"] = float(axial_cm)
                if sar:
                    h5.attrs[f"probe{j}_lateral_cm_per_px"] = float(axial_cm * sar)
            if sar:
                h5.attrs[f"probe{j}_sar"] = float(sar)
        tg = h5.create_group("timing")
        if time_ms is not None:
            tg.create_dataset("frame_idx_1n", data=np.arange(1, len(time_ms) + 1, dtype=np.int32))
            tg.create_dataset("time_ms", data=time_ms)
            ifi = np.zeros(len(time_ms), dtype=np.float64); ifi[1:] = np.diff(time_ms)
            tg.create_dataset("ifi_ms", data=ifi)

    dt = time.perf_counter() - t0
    counts = {pr.suffix or "single": pr.n for pr in probes.values()}
    sz_gb = tvd.stat().st_size / 1e9
    if progress:
        _log(f"comfree {stem}: {counts} frames, {geom.lines}x{geom.active} native, "
             f"{sz_gb:.2f} GB in {dt:.1f} s ({(sum(p.n for p in probes.values()))/dt:.0f} fps)",
             progress=progress)
    return {"h5": str(h5_path), "videos": [str(out_paths[s]) for s in probes],
            "counts": counts, "seconds": dt, "bytes": tvd.stat().st_size,
            "geometry": (geom.lines, geom.active), "sar": sar}


def _stream_frames(f, start: int, end: int, dispatch) -> None:
    """Sequentially walk chunks in ``[start,end)``, dispatching frame payloads. Constant memory.

    Descends into ``UIFF``/``LIST`` containers; for a frame chunk reads its 48-byte sub-header +
    payload and hands ``(stream, payload)`` to ``dispatch``; skips everything else by seeking.
    """
    f.seek(start)
    i = start
    while i + 12 <= end:
        hdr = f.read(12)
        if len(hdr) < 12:
            return
        cid = hdr[:4]
        if not all(32 <= b < 127 for b in cid):
            return
        size = int.from_bytes(hdr[4:12], "little")
        pay = i + 12
        if cid in _FRAME_STREAMS:
            sub = f.read(_FRAME_HDR)
            datasize = struct.unpack_from("<I", sub, _FRAME_DATASIZE_OFF)[0]
            payload = f.read(datasize)
            dispatch(cid, payload)
        elif cid in _CONTAINER_IDS:
            f.seek(pay + 4)
            _stream_frames(f, pay + 4, min(pay + size, end), dispatch)
        i = pay + size + (size & 1)
        f.seek(i)


# ---------- batch entry ----------


def extract_comfree(
    source: Union[str, Path, Iterable[Union[str, Path]]],
    out_dir: Union[str, Path],
    *,
    recursive: bool = True,
    pattern: str = "*.tvd",
    skip_existing: bool = True,
    lossless: bool = True,
    crf: int = 24,
    preset: str = "ultrafast",
    overwrite: bool = False,
    progress: bool = True,
) -> dict:
    """Fully COM-free batch extract: ``.tvd`` -> per-probe mp4(s) + metadata/timing ``.tvd.h5``.

    No EchoWave, no Admin, no COM. ``out_dir`` receives all outputs (kept off the network drive on
    purpose -- pass a local path). ``skip_existing`` skips a recording whose ``.tvd.h5`` already
    exists in ``out_dir``. Returns ``{tvd_path: status}`` plus a ``"_timing"`` summary entry.
    """
    out_dir = Path(out_dir)
    files = _normalize_sources(source, recursive=recursive, pattern=pattern)
    results: dict = {}
    t_all = time.perf_counter()
    total_bytes = 0
    total_frames = 0
    for idx, tvd in enumerate(files):
        h5_out = _sidecar_h5_path(out_dir / tvd.name)
        if skip_existing and h5_out.exists():
            results[str(tvd)] = "hit"
            if progress:
                print(f"[{idx + 1}/{len(files)}] {tvd.name}  (hit, skip)", flush=True)
            continue
        if progress:
            print(f"[{idx + 1}/{len(files)}] {tvd.name}", flush=True)
        try:
            r = extract_one(tvd, out_dir, lossless=lossless, crf=crf, preset=preset,
                            overwrite=overwrite, progress=progress)
            results[str(tvd)] = "built"
            total_bytes += r["bytes"]
            total_frames += sum(r["counts"].values())
        except Exception as e:  # noqa: BLE001
            results[str(tvd)] = f"error: {e}"
            if progress:
                print(f"    error: {e}", flush=True)
    dt = time.perf_counter() - t_all
    results["_timing"] = {
        "seconds": dt, "n_files": len(files), "total_GB": round(total_bytes / 1e9, 2),
        "total_frames": total_frames, "fps": round(total_frames / dt, 1) if dt else 0,
    }
    if progress:
        print(f"\ncomfree batch: {len(files)} file(s), {total_bytes/1e9:.1f} GB, "
              f"{total_frames} frames in {dt:.1f} s ({results['_timing']['fps']} fps).", flush=True)
    return results
