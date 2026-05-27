"""Completeness QC for extracted Telemed recordings.

EchoWave silently truncates a ``.tvd`` load to the number of frames
that fit in available memory. The extracted ``.tvd.h5`` then carries
fewer frames than the device recorded, with no error raised. This
module's :func:`verify_complete` catches that by comparing the
extracted frame count against two independent references:

* **the recorded count in the source ``.tvd`` header**
  (:func:`telemed.read_tvd_n_frames`) -- present for every recording,
  and pristine because the extract pipeline opens a *copy* of the
  ``.tvd``, never the source. This is the universal signal.
* **the native EchoWave ``<stem>.mp4`` export's ``nb_frames``** when one
  sits beside the sidecar. For dual-probe recordings that side-by-side
  mp4 is the operator's own export (a genuinely independent count,
  validated to match the extracted ``n_frames`` exactly on every
  complete pia02 recording). For single-probe recordings ``<stem>.mp4``
  is this package's own output, so it's a consistency check rather than
  an independent one.

Already-extracted sidecars don't need re-processing to be auditable:
:func:`verify_complete` reads the sibling ``.tvd`` header directly when
a sidecar predates the stored ``tvd_declared_n_frames`` attr. Use
:func:`backfill_tvd_n_frames` to write that attr into old sidecars
(milliseconds each -- no COM, no re-extraction) if you want them
consistent with future extractions.

Re-exported as ``telemed.verify_complete`` / ``telemed.backfill_tvd_n_frames``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Union

import h5py

from ._encode import _resolve_h5_sources, _stem_from_h5
from ._extract import _TVD_TRUNCATION_TOLERANCE, _samples_look_lut_inverted, read_tvd_n_frames


def _probe_mp4_nb_frames(mp4_path: Path) -> Optional[int]:
    """``nb_frames`` from an mp4 container via ffprobe; None on failure.

    Reads the container's stream metadata (instant -- no packet scan).
    Returns ``None`` when ffprobe isn't on PATH, the file is unreadable,
    or the field is absent / non-numeric.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        out = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "json",
                str(mp4_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        nb = json.loads(out.stdout)["streams"][0]["nb_frames"]
        return int(nb)
    except Exception:  # noqa: BLE001
        return None


def looks_lut_inverted(source, *, n_samples: int = 3) -> bool:
    """True if an extracted sidecar's frames look LUT-inverted.

    EchoWave builds before 4.4.0 return ``GetLoadedFrameGray`` with the
    grayscale LUT inverted (``255 - x``), so the normally-dark B-mode
    background comes back bright. ``export_h5`` guards against this at
    extract time; this function runs the same pixel-statistics test on
    an *already-extracted* ``.tvd.h5`` so an existing cohort can be
    audited.

    Args:
        source: A ``.tvd.h5`` path or a :class:`telemed.Log` instance.
        n_samples: How many frames to sample (first / middle / last).

    Returns:
        ``True`` when the sampled frames look inverted. ``False`` when
        they look normal **or** the sidecar can't be tested (extracted
        with ``frames=False``, or no img_id=1 panel) -- a frames-less
        sidecar carries no pixels to judge.
    """
    from .log import Log

    lf = source if isinstance(source, Log) else Log(source)
    if not lf.has_frames or 1 not in lf.b_mode_rois:
        return False
    n = lf.n_frames
    idxs = sorted({0, n // 2, n - 1})[: max(1, n_samples)]
    frames = [lf.frame(i, crop=False) for i in idxs]
    return _samples_look_lut_inverted(frames, lf.b_mode_rois[1])


def _sibling_tvd_for(h5_path: Path) -> Path:
    """``<stem>.tvd.h5`` -> sibling ``<stem>.tvd``."""
    name = h5_path.name
    if name.endswith(".tvd.h5"):
        return h5_path.with_name(name[: -len(".h5")])
    return h5_path.with_suffix("")


def _native_mp4_for(h5_path: Path) -> Path:
    """The ``<stem>.mp4`` that sits beside the sidecar (native export for
    dual-probe recordings; this package's own output for single-probe)."""
    return h5_path.parent / f"{_stem_from_h5(h5_path)}.mp4"


def _verify_one(h5_path: Path, *, tol: int) -> dict:
    """Build the completeness record for one ``.tvd.h5``.

    Returns a dict with ``extracted`` / ``declared`` / ``native_mp4``
    frame counts (any may be ``None``) and a ``status`` of
    ``"complete"`` / ``"truncated"`` / ``"unknown"`` (no reference
    available) / ``f"error: {msg}"``.
    """
    try:
        with h5py.File(h5_path, "r") as h5:
            extracted = int(h5.attrs["n_frames"])
            declared = h5.attrs.get("tvd_declared_n_frames")
            declared = int(declared) if declared is not None else None
    except Exception as e:  # noqa: BLE001
        return {"extracted": None, "declared": None, "native_mp4": None, "status": f"error: {e}"}

    # Back-compat: sidecars extracted before the attr existed -- parse
    # the sibling .tvd header on the fly so old files are still auditable.
    if declared is None:
        tvd = _sibling_tvd_for(h5_path)
        if tvd.is_file():
            declared = read_tvd_n_frames(tvd)

    native = None
    native_mp4 = _native_mp4_for(h5_path)
    if native_mp4.is_file():
        native = _probe_mp4_nb_frames(native_mp4)

    issues = []
    if declared is not None and declared - extracted > tol:
        issues.append(f".tvd header declares {declared} (~{declared - extracted} missing)")
    if native is not None and native - extracted > tol:
        issues.append(f"native mp4 has {native} (~{native - extracted} missing)")

    if declared is None and native is None:
        status = "unknown"
    elif issues:
        status = "truncated"
    else:
        status = "complete"

    return {
        "extracted": extracted,
        "declared": declared,
        "native_mp4": native,
        "status": status,
        "issues": issues,
    }


def _format_line(h5_path: Path, info: dict) -> str:
    """One human-readable report line for a verify result."""
    status = info["status"]
    if status.startswith("error"):
        return f"  [error]      {h5_path.name}: {status[len('error: '):]}"
    extracted = info["extracted"]
    ref_bits = []
    if info["declared"] is not None:
        ref_bits.append(f".tvd={info['declared']}")
    if info["native_mp4"] is not None:
        ref_bits.append(f"native_mp4={info['native_mp4']}")
    refs = (" [" + ", ".join(ref_bits) + "]") if ref_bits else " [no reference]"
    tag = {"complete": "[complete]", "truncated": "[TRUNCATED]", "unknown": "[unknown] "}[status]
    line = f"  {tag}  {h5_path.name}: extracted={extracted}{refs}"
    if info.get("issues"):
        line += " -- " + "; ".join(info["issues"])
    return line


def verify_complete(
    source: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    recursive: bool = True,
    tol: int = _TVD_TRUNCATION_TOLERANCE,
    progress: bool = True,
) -> dict:
    """Check extracted ``.tvd.h5`` sidecar(s) for memory-truncated loads.

    For each recording, compares the extracted ``n_frames`` against the
    ``.tvd`` header's recorded count (the universal reference) and the
    native ``<stem>.mp4`` ``nb_frames`` when present (see the module
    docstring for which is independent when). A recording is flagged
    ``"truncated"`` when a reference exceeds the extracted count by more
    than ``tol`` frames -- the benign header overcount is ~2 frames, a
    memory truncation drops orders of magnitude more, so the default
    ``tol`` cleanly separates the two.

    Works on already-extracted sidecars without re-processing: when a
    sidecar predates the stored ``tvd_declared_n_frames`` attr, the
    sibling ``.tvd`` header is parsed directly.

    Args:
        source: A ``.tvd.h5`` file, a ``.tvd`` file (its sibling sidecar
            is checked), a directory (walked for ``*.tvd.h5``), or an
            iterable of any combination.
        recursive: Recurse into subdirectories when walking directories.
        tol: Frame-count slack below which a recording counts as
            complete (default 16).
        progress: Print a per-recording report line plus a summary.

    Returns:
        ``{h5_path_str: info}`` where ``info`` is the dict from
        :func:`_verify_one` (``extracted`` / ``declared`` /
        ``native_mp4`` / ``status`` / ``issues``).

    Example::

        results = telemed.verify_complete("M:/data/060")
        bad = [p for p, i in results.items() if i["status"] == "truncated"]
    """
    h5_files = _resolve_h5_sources(source, recursive=recursive, pattern="*.tvd.h5")
    results: dict = {}
    for h5_path in h5_files:
        info = _verify_one(h5_path, tol=tol)
        results[str(h5_path)] = info
        if progress:
            print(_format_line(h5_path, info), flush=True)
    if progress:
        n_trunc = sum(1 for v in results.values() if v["status"] == "truncated")
        n_unk = sum(1 for v in results.values() if v["status"] == "unknown")
        print(
            f"telemed.verify_complete: {len(results)} checked, "
            f"{n_trunc} truncated, {n_unk} with no reference.",
            flush=True,
        )
    return results


def backfill_tvd_n_frames(
    source: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    recursive: bool = True,
    progress: bool = True,
) -> dict:
    """Write ``tvd_declared_n_frames`` into already-extracted sidecars.

    Parses each sidecar's sibling ``.tvd`` header and stores the
    recorded frame count as a root attr, so old sidecars carry the same
    completeness metadata as future extractions. Fast (header-only read
    + one attr write per file) -- no COM, no re-extraction. Re-running
    the full :func:`telemed.process` would only be worth it for a
    sidecar that is *actually* truncated (a fresh extract with more free
    RAM recovers the missing frames); to merely record the metadata,
    use this.

    Args:
        source: Same shapes as :func:`verify_complete`.
        recursive: Recurse into subdirectories when walking directories.
        progress: Print a per-file status line.

    Returns:
        ``{h5_path_str: status}`` where status is ``f"added ({n})"`` /
        ``f"updated ({n})"`` / ``"skipped: no sibling .tvd"`` /
        ``"skipped: unparseable .tvd header"`` / ``f"error: {msg}"``.
    """
    h5_files = _resolve_h5_sources(source, recursive=recursive, pattern="*.tvd.h5")
    results: dict = {}
    for h5_path in h5_files:
        tvd = _sibling_tvd_for(h5_path)
        if not tvd.is_file():
            results[str(h5_path)] = "skipped: no sibling .tvd"
        else:
            declared = read_tvd_n_frames(tvd)
            if declared is None:
                results[str(h5_path)] = "skipped: unparseable .tvd header"
            else:
                try:
                    with h5py.File(h5_path, "r+") as h5:
                        existed = "tvd_declared_n_frames" in h5.attrs
                        h5.attrs["tvd_declared_n_frames"] = int(declared)
                    results[str(h5_path)] = (
                        f"updated ({declared})" if existed else f"added ({declared})"
                    )
                except Exception as e:  # noqa: BLE001
                    results[str(h5_path)] = f"error: {e}"
        if progress:
            print(f"  {h5_path.name}: {results[str(h5_path)]}", flush=True)
    return results
