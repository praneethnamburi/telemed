"""Pipeline orchestrator for ``telemed.process()``.

Folds the three pipeline stages (``.tvd -> .tvd.h5 -> .mp4 + .dnav-toc``)
into a single per-file end-to-end pipeline so the encode + TOC + upload
cost for file N hides inside the COM extract window of file N+1.

Bottleneck shape on the typical pia02 workload (20k-frame dual-probe
recording on a network drive):

* Extract (COM, single-threaded): ~33 min (timing-only) / ~67 min (pixels)
* Encode mp4(s) (libx265 ultrafast): ~100 s (dual panel)
* TOC build (PyAV demux): ~80 s (dual panel)
* Upload (h5 ~20 GB + mp4s ~3 GB + TOCs): ~3-5 min

So the full post-process for one recording (~6-9 min) comfortably
hides inside the next recording's extract (~33-67 min). The user's
wall clock is bounded by extract time + a small drain at the end.

Triage shape:

* Set A: ``.tvd`` files with no sibling ``.tvd.h5`` (need extraction)
* Set B: ``.tvd.h5`` files already on disk (just need encode + TOC)

The dispatcher routes to one pipeline, the other, or both. Scenario 4
(both non-empty) runs the two pipelines concurrently on a top-level
2-thread executor; the bottlenecks are orthogonal (Set A is COM-bound,
Set B is CPU/disk-bound for libx265 + PyAV) so they don't compete on
the critical path.
"""
from __future__ import annotations

import concurrent.futures
import inspect
import shutil
import time
from pathlib import Path
from typing import Iterable, Optional, Union

from . import _encode  # dotted access so monkeypatch.setattr propagates to bg-thread postprocess
from ._encode import export_video
from ._extract import (
    _StagedFile,
    _log,
    _normalize_sources,
    _sidecar_h5_path,
    _size_human,
    _unstage_one,
    export_h5,
)


# ---------- Triage ----------


def _triage(
    source: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    recursive: bool = True,
) -> tuple[list[Path], list[Path]]:
    """Split sources into ``(set_a, set_b)``.

    * ``set_a``: ``.tvd`` files with no sibling ``.tvd.h5`` -- need
      extraction (Pipeline A).
    * ``set_b``: ``.tvd.h5`` files on disk -- need encode + TOC
      only (Pipeline B). Includes both .tvd.h5 with a matching .tvd
      and orphaned .tvd.h5 (where the source .tvd was deleted post-
      extraction).

    Raises:
        FileNotFoundError: A named source path doesn't exist. The
            most common cause on Windows is running elevated Python
            against a mapped network drive: elevation tokens don't
            inherit the user's mapped drives, so ``M:\\...`` literally
            doesn't exist from an Admin process. Use the UNC path
            (``\\\\server\\share\\...``) instead, or enable
            ``EnableLinkedConnections`` in HKLM to share mappings
            across elevation. ``_normalize_sources`` silently skips
            non-existent entries (for the recursive-walk case where
            some entries are legitimately absent), so we re-validate
            here and surface the misconfig early.
    """
    if isinstance(source, (str, Path)):
        entries: list[Path] = [Path(source)]
    else:
        entries = [Path(s) for s in source]
    missing = [e for e in entries if not e.exists()]
    if missing:
        raise FileNotFoundError(
            "telemed.process: source path(s) not found: "
            + ", ".join(repr(str(m)) for m in missing)
            + ". On Windows, mapped network drives (M:, S:, ...) do "
            "NOT inherit across UAC elevation -- if you're running "
            "elevated Python (required for the COM extract step), "
            "use the UNC path (\\\\server\\share\\...) or enable "
            "EnableLinkedConnections in HKLM."
        )
    tvds = _normalize_sources(source, recursive=recursive, pattern="*.tvd")
    h5s = _normalize_sources(source, recursive=recursive, pattern="*.tvd.h5")
    set_a = [t for t in tvds if not _sidecar_h5_path(t).exists()]
    set_b = list(h5s)
    return set_a, set_b


# ---------- Pipeline A postprocess: encode + upload + TOC ----------


def _make_postprocess(
    *,
    video_kwargs: dict,
    video_results: dict,
    toc_results: dict,
    progress: bool = False,
):
    """Build the postprocess closure for ``export_h5(set_a, ...)``.

    The closure runs on a background worker after each COM extract.
    On success it:

    1. Encodes mp4(s) into the same dir as ``staged.local_h5`` (local
       temp when copied, source folder otherwise; ``build_toc=False``
       deferred until after the upload).
    2. Copies the .h5 + mp4(s) to ``staged.dst_h5.parent`` when the
       file was staged via local copy. No-op when the source was
       already local (``staged.stage_dir is None``).
    3. Cleans up the local staging dir (when present).
    4. Builds the dnav TOC sidecar **against the final-destination
       mp4** -- if we built it locally then uploaded, the sidecar's
       cache-key (size+mtime+SHA of head/tail) would invalidate on
       first read of the network copy.

    Each phase is bracketed with timestamped log lines when
    ``progress=True`` so the user can see what the bg worker is
    doing during the (long) extract of the next file.

    The closure writes per-mp4 status into the shared ``video_results``
    and ``toc_results`` dicts. Python's GIL keeps dict assignment
    atomic per statement, which is enough for status strings; no
    explicit lock needed.
    """

    def _postprocess(staged: _StagedFile, success: bool) -> None:
        if not success:
            _unstage_one(staged, upload=False, progress=progress)
            return

        dst_dir = staged.dst_h5.parent
        staged_locally = staged.stage_dir is not None
        rec_name = staged.src_tvd.name

        # Phase 1: encode mp4(s) locally (or in-place if not staged).
        # build_toc=False here -- we build TOC against the network
        # mp4 after upload (see Phase 4).
        _log(
            f"encoding {rec_name} -> mp4(s)...",
            tag="encode", progress=progress,
        )
        t0 = time.perf_counter()
        try:
            local_video_results = export_video(
                staged.local_h5,
                out_dir=staged.local_h5.parent,
                build_toc=False,
                progress=False,
                **video_kwargs,
            )
            _log(
                f"encoded {rec_name} in {time.perf_counter() - t0:.1f} s "
                f"({len(local_video_results)} panel(s))",
                tag="encode", progress=progress,
            )
        except Exception as e:  # noqa: BLE001
            video_results[str(dst_dir / f"{staged.src_tvd.stem}.mp4")] = (
                f"error: encode: {e}"
            )
            local_video_results = {}
            _log(
                f"encode failed for {rec_name}: {e}",
                tag="encode", progress=progress,
            )

        # Phase 2: upload h5 + mp4s (staged only) and resolve each
        # mp4's final destination path. For local sources, the mp4
        # already lives at the destination; record it in place.
        network_mp4s: list[Path] = []
        try:
            if staged_locally and staged.local_h5.exists():
                try:
                    h5_size = staged.local_h5.stat().st_size
                except OSError:
                    h5_size = 0
                _log(
                    f"uploading {staged.local_h5.name} "
                    f"({_size_human(h5_size)}) -> {dst_dir}...",
                    tag="upload", progress=progress,
                )
                t_up = time.perf_counter()
                shutil.copy2(staged.local_h5, staged.dst_h5)
                _log(
                    f"uploaded {staged.local_h5.name} in "
                    f"{time.perf_counter() - t_up:.1f} s",
                    tag="upload", progress=progress,
                )

            for local_mp4_str, status in local_video_results.items():
                local_mp4 = Path(local_mp4_str)
                if staged_locally:
                    network_mp4 = dst_dir / local_mp4.name
                    if status in ("built", "hit") and local_mp4.exists():
                        try:
                            mp4_size = local_mp4.stat().st_size
                        except OSError:
                            mp4_size = 0
                        _log(
                            f"uploading {local_mp4.name} "
                            f"({_size_human(mp4_size)}) -> {dst_dir}...",
                            tag="upload", progress=progress,
                        )
                        t_up = time.perf_counter()
                        shutil.copy2(local_mp4, network_mp4)
                        _log(
                            f"uploaded {local_mp4.name} in "
                            f"{time.perf_counter() - t_up:.1f} s",
                            tag="upload", progress=progress,
                        )
                        network_mp4s.append(network_mp4)
                else:
                    network_mp4 = local_mp4  # already at destination
                    if status in ("built", "hit") and network_mp4.exists():
                        network_mp4s.append(network_mp4)
                video_results[str(network_mp4)] = status
        finally:
            # Phase 3: cleanup local temp (no-op when not staged).
            if staged_locally:
                _log(
                    f"cleaning up {staged.stage_dir}",
                    tag="cleanup", progress=progress,
                )
                shutil.rmtree(staged.stage_dir, ignore_errors=True)

        # Phase 4: TOC against final-destination mp4 (so the cache
        # key matches what subsequent dnav opens will see). Dotted
        # access via _encode so monkeypatch.setattr in tests reaches
        # this background-thread call site.
        for mp4 in network_mp4s:
            _log(
                f"building TOC for {mp4.name}...",
                tag="toc", progress=progress,
            )
            t_toc = time.perf_counter()
            status = _encode._ensure_toc_sidecar(mp4)
            toc_results[str(mp4)] = status
            _log(
                f"TOC {mp4.name}: {status} "
                f"({time.perf_counter() - t_toc:.1f} s)",
                tag="toc", progress=progress,
            )

    return _postprocess


# ---------- Pipeline B: encode + TOC for files that already have .h5 ----------


def _run_pipeline_b(
    set_b: list[Path],
    *,
    video_kwargs: dict,
    video_results: dict,
    toc_results: dict,
    progress: bool,
) -> None:
    """Sequential encode + TOC per .tvd.h5 in ``set_b``.

    No background workers -- there's no extract step to hide work
    behind. ``export_video`` itself does the encode + TOC inline
    (``build_toc=True``).
    """
    for h5 in set_b:
        try:
            local = export_video(
                h5,
                build_toc=True,
                progress=progress,
                **video_kwargs,
            )
        except Exception as e:  # noqa: BLE001
            video_results[str(h5)] = f"error: {e}"
            continue
        for mp4_str, status in local.items():
            video_results[mp4_str] = status
            if status in ("built", "hit"):
                # export_video already built/checked the TOC sidecar;
                # reflect status here for transparency. A re-check is
                # cheap (one Path.exists()) -- avoids re-running the
                # full dnav probe.
                sidecar = Path(mp4_str + ".dnav-toc")
                toc_results[mp4_str] = "hit" if sidecar.exists() else "missing"


# ---------- Public entry point ----------


def process(
    source: Union[str, Path, Iterable[Union[str, Path]]],
    **kwargs,
) -> dict:
    """Run the full pipeline -- ``.tvd -> .tvd.h5 -> .mp4(s) + .dnav-toc``.

    The end-to-end orchestrator for the canonical Telemed workflow.
    Triages sources into two sets and runs the appropriate pipeline
    for each; when both sets are non-empty, runs them concurrently
    on a 2-thread executor.

    * Set A (``.tvd`` without sibling ``.tvd.h5``): full pipeline.
      COM extract on the main thread; for each completed extract, a
      background worker encodes mp4(s), uploads everything to the
      source folder (when the source is on a network drive), and
      builds the dnav TOC sidecar. The post-process cost hides
      inside the next file's extract window.
    * Set B (``.tvd.h5`` already on disk): sequential encode + TOC.
      No COM, no upload. Idempotent on cohorts where mp4 + TOC
      already exist (``skip_existing=True``).

    Per-stage idempotency: re-running on a partly-processed corpus
    picks up where the previous run left off. Existing ``.tvd.h5``
    skips extract; existing ``.mp4`` skips encode; existing
    ``.dnav-toc`` skips TOC build. Force a full rebuild with
    ``skip_existing=False, overwrite=True``.

    Kwargs are signature-routed:

    * h5-only kwargs (``frames``, ``compression``, ``compression_opts``,
      ``copy_to_local``, ``local_temp_root``) -> Pipeline A's
      ``export_h5``.
    * video-only kwargs (``out_dir``, ``codec``, ``lossless``,
      ``crf``, ``preset``, ``fps``, ``normalize_orientation``,
      ``crop``, ``build_toc``, ``overwrite``) -> ``export_video``
      in both pipelines.
    * Common kwargs (``recursive``, ``skip_existing``, ``progress``,
      ``progress_callback``, ``cancel_check``) -> both pipelines.

    Returns:
        ``{"h5": {...}, "video": {...}, "toc": {...}}`` -- per-stage
        result dicts. Each maps path-string -> status (``"built"`` /
        ``"hit"`` / ``"missing"`` / ``"skipped: no dnav"`` /
        ``f"error: {msg}"``).

    Example::

        telemed.process(r"M:/data/pia02")
        # Walks for .tvd + .tvd.h5; triages into needs-extract vs
        # has-h5; runs both pipelines concurrently if both sets
        # are non-empty; returns when every file has its mp4 + TOC.
    """
    h5_params = set(inspect.signature(export_h5).parameters) - {"source"}
    video_params = set(inspect.signature(export_video).parameters) - {"source"}
    # ``postprocess`` is set by the dispatcher, never by the caller.
    h5_params.discard("postprocess")
    unknown = set(kwargs) - h5_params - video_params
    if unknown:
        raise TypeError(
            f"process(): unknown kwargs {sorted(unknown)}; accepted: "
            f"h5={sorted(h5_params - video_params)}, "
            f"video={sorted(video_params - h5_params)}, "
            f"common={sorted(h5_params & video_params)}"
        )
    h5_kw = {k: v for k, v in kwargs.items() if k in h5_params}
    video_kw = {k: v for k, v in kwargs.items() if k in video_params}

    recursive = kwargs.get("recursive", True)
    progress = kwargs.get("progress", True)
    set_a, set_b = _triage(source, recursive=recursive)

    if progress:
        # Always say what triage saw -- silent no-ops on empty
        # folders / typo'd paths are the #1 confusion source.
        if not set_a and not set_b:
            print(
                f"telemed.process: no .tvd or .tvd.h5 files found under "
                f"{source!r} (recursive={recursive}). Nothing to do.",
                flush=True,
            )
        else:
            print(
                f"telemed.process: triage -> {len(set_a)} .tvd needing "
                f"extract, {len(set_b)} .tvd.h5 needing encode/TOC.",
                flush=True,
            )

    h5_results: dict = {}
    video_results: dict = {}
    toc_results: dict = {}

    def _run_pipeline_a() -> None:
        if not set_a:
            return
        postprocess = _make_postprocess(
            video_kwargs=video_kw,
            video_results=video_results,
            toc_results=toc_results,
            progress=progress,
        )
        local_h5_results = export_h5(
            set_a, postprocess=postprocess, **h5_kw,
        )
        h5_results.update(local_h5_results)

    def _run_pipeline_b_local() -> None:
        if not set_b:
            return
        _run_pipeline_b(
            set_b,
            video_kwargs=video_kw,
            video_results=video_results,
            toc_results=toc_results,
            progress=progress,
        )

    if set_a and set_b:
        # Scenario 4: bottlenecks are orthogonal (COM vs CPU/disk),
        # run concurrently.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="telemed-pipeline",
        ) as super_pool:
            fa = super_pool.submit(_run_pipeline_a)
            fb = super_pool.submit(_run_pipeline_b_local)
            fa.result()
            fb.result()
    else:
        # Single pipeline; run on the calling thread for simplicity.
        _run_pipeline_a()
        _run_pipeline_b_local()

    return {"h5": h5_results, "video": video_results, "toc": toc_results}
