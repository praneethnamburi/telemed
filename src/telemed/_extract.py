"""COM-backed reader for Telemed `.tvd` (Telemed Video Data) files.

Wraps the AutoInt1 automation interface that ships with EchoWave II
(see ``C:/Program Files/Telemed/Echo Wave II Application/EchoWave II/
Config/Plugins/AutoInt1Client.txt`` for the underlying API docs).

This is the **only** chroma-free + native-VFR-timing path for Telemed
device data: bypasses the lossy mp4 export entirely, gives true uint8
grayscale arrays and per-frame timestamps at the device's native ~100 ns
precision. (DICOM exports also carry the timing in FrameTimeVector, but
truncate above ~10k frames -- unusable for typical pia02 recordings.)

One-time setup (per machine):

1. **Register the COM ProgID** -- open an Administrator PowerShell::

    cd "C:\\Program Files\\Telemed\\Echo Wave II Application\\EchoWave II\\Config\\Plugins"
    .\\AutoInt1_regasm.bat

   You should see "Types registered successfully".

Per-session setup:

2. **Start Echo Wave II as administrator** (right-click -> "Run as
   administrator"). Get it to its normal main window.
3. **Run Python from an Administrator shell** -- the COM connection
   only binds when both processes share elevation.

Network-drive note: EchoWave's OpenFile fails on UNC / mapped network
paths in our setup. :func:`extract_recording_folder` handles this
transparently by copying each source file to a local temp directory,
processing locally, and writing results back to the source folder.

Example::

    from immersionlab import telemed

    # Single file -- writes a sibling .tvd.h5
    telemed.export("C:/data/some.tvd")

    # Timing-only (much faster; skip pixel extraction)
    telemed.export("C:/data/some.tvd", frames=False)

    # Batch a folder, even when on a network drive
    telemed.export("M:/data/pia02")

    # Mix folders and individual files
    telemed.export(["M:/data/pia02", "M:/data/pia03", "C:/scratch/x.tvd"])

Known win32com gotcha (wrapped inside this module): zero-argument COM
methods on the .NET CCW are exposed as **properties**, not callables.
Attribute access invokes the call. See
``feedback_win32com_dotnet_ccw_zero_arg_property`` in auto-memory.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Union


# ---------- Metadata structures ----------


@dataclass
class TelemedRoi:
    """B-mode region-of-interest within the full Telemed display frame.

    Coordinates are reported by the COM API for ``img_id=1`` (B-mode).
    Units are pixels in the full-frame display coordinate system. Note
    the COM API uses 1-based pixel indexing (so x1=73 means the ROI's
    leftmost column is the 73rd pixel from the left edge); the
    ``width`` and ``height`` here are inclusive pixel counts
    (``x2 - x1 + 1``, ``y2 - y1 + 1``).
    """

    x1: int
    x2: int
    y1: int
    y2: int
    width: int
    height: int

    @classmethod
    def from_cmd(cls, cmd, img_id: int = 1) -> "TelemedRoi":
        x1 = int(cmd.GetUltrasoundX1(img_id))
        x2 = int(cmd.GetUltrasoundX2(img_id))
        y1 = int(cmd.GetUltrasoundY1(img_id))
        y2 = int(cmd.GetUltrasoundY2(img_id))
        return cls(
            x1=x1, x2=x2, y1=y1, y2=y2,
            width=x2 - x1 + 1, height=y2 - y1 + 1,
        )


@dataclass
class TelemedRecordingMeta:
    """Per-recording metadata captured alongside per-frame timing.

    Persisted into the HDF5 sidecar's root attributes so downstream
    code can reproduce crops + scale physical measurements without
    re-opening the .tvd through EchoWave.
    """

    n_frames: int
    full_frame_width: int
    full_frame_height: int
    b_mode_roi: TelemedRoi
    physical_dx_cm_per_px: float
    physical_dy_cm_per_px: float
    source_tvd_path: str
    extracted_at_iso: str
    schema_version: int = 1

    def to_flat_attrs(self) -> dict:
        """Flatten for HDF5 root-attribute persistence (no nested dicts)."""
        d = {
            k: v for k, v in asdict(self).items() if k != "b_mode_roi"
        }
        for k, v in asdict(self.b_mode_roi).items():
            d[f"roi_{k}"] = v
        return d

    @classmethod
    def from_cmd(cls, cmd, source_tvd_path: Union[str, Path]) -> "TelemedRecordingMeta":
        # Need to load frame 1 once to populate width/height.
        cmd.GoToFrame1n(1, True)
        return cls(
            n_frames=int(cmd.GetFramesCount),
            full_frame_width=int(cmd.GetLoadedFrameWidth),
            full_frame_height=int(cmd.GetLoadedFrameHeight),
            b_mode_roi=TelemedRoi.from_cmd(cmd, img_id=1),
            physical_dx_cm_per_px=float(cmd.GetUltrasoundPhysicalDeltaX(1)),
            physical_dy_cm_per_px=float(cmd.GetUltrasoundPhysicalDeltaY(1)),
            source_tvd_path=str(source_tvd_path),
            extracted_at_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )


# ---------- Reader ----------


_PROGID = "EchoWave2.CmdInt1"


class TelemedTvdReader:
    """Wraps the EchoWave II AutoInt1 COM interface.

    A single reader instance is fine for the lifetime of the EchoWave
    process; ``open()`` can be called repeatedly to switch files
    (Echo Wave is single-document, so the previously-open file is
    closed implicitly).

    See module docstring for setup prereqs.
    """

    def __init__(self):
        self._cmd = None
        self._opened: Optional[Path] = None

    def connect(self) -> "TelemedTvdReader":
        """Attach to the running Echo Wave II instance via the COM ROT.

        Raises:
            RuntimeError: If GetActiveObject fails (Echo Wave not
                running, COM ProgID not registered, or elevation
                mismatch).
        """
        try:
            import win32com.client
        except ImportError as e:
            raise RuntimeError(
                "pywin32 is required for the COM .tvd path. "
                "Install with: conda install -c conda-forge pywin32"
            ) from e
        try:
            self._cmd = win32com.client.GetActiveObject(_PROGID)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"GetActiveObject('{_PROGID}') failed: {e}\n"
                "Causes:\n"
                "  - Echo Wave II is not running -- start it.\n"
                "  - Echo Wave II is not 'Run as administrator'.\n"
                "  - This Python process is not 'Run as administrator'.\n"
                "  - AutoInt1.dll is not registered (one-time fix:\n"
                "    run AutoInt1_regasm.bat as Admin)."
            ) from e
        return self

    def _require_cmd(self):
        if self._cmd is None:
            raise RuntimeError("Not connected. Call .connect() first.")
        return self._cmd

    def _require_open(self):
        cmd = self._require_cmd()
        if self._opened is None:
            raise RuntimeError("No file open. Call .open(path) first.")
        return cmd

    def open(self, tvd_path: Union[str, Path]) -> "TelemedTvdReader":
        """Open a .tvd file in Echo Wave (idempotent freeze/stop).

        Stops any running scan, recording, or cine playback before
        opening (per AutoInt1Client.txt those states prevent
        GoToFrame1n navigation).

        Raises:
            FileNotFoundError: If ``tvd_path`` doesn't exist.
            RuntimeError: If EchoWave's OpenFile returns -1 (common
                cause: the file is on a network drive -- EchoWave's
                OpenFile fails on UNC / mapped network paths in our
                setup, so callers should copy to local first).
        """
        cmd = self._require_cmd()
        p = Path(tvd_path)
        if not p.is_file():
            raise FileNotFoundError(p)

        # Property-style access fires the call. See module docstring.
        if cmd.IsRecordingState == 1:
            _ = cmd.RecordStop
        if cmd.IsRunState == 1:
            _ = cmd.FreezeRun
        if cmd.IsPlayState == 1:
            _ = cmd.PlayPause

        if cmd.OpenFile(str(p)) == -1:
            raise RuntimeError(
                f"OpenFile failed for {p}. If the file is on a network "
                f"drive, copy it to a local path first -- EchoWave's "
                f"OpenFile fails on UNC / mapped network paths."
            )
        self._opened = p
        return self

    @property
    def opened_path(self) -> Optional[Path]:
        return self._opened

    @property
    def n_frames(self) -> int:
        return int(self._require_open().GetFramesCount)

    @property
    def b_mode_roi(self) -> TelemedRoi:
        return TelemedRoi.from_cmd(self._require_open(), img_id=1)

    def get_frame_time_ms(self, frame_idx_0n: int) -> float:
        """Time of frame ``frame_idx_0n`` in ms, with frame 0 -> 0.0.

        ``frame_idx_0n`` is 0-indexed for Python convention; the
        underlying COM API is 1-indexed and the conversion happens
        here.
        """
        cmd = self._require_open()
        if not (0 <= frame_idx_0n < self.n_frames):
            raise IndexError(
                f"frame_idx_0n {frame_idx_0n} out of range "
                f"[0, {self.n_frames})"
            )
        cmd.GoToFrame1n(frame_idx_0n + 1, False)
        return float(cmd.GetCurrentFrameTime)

    def get_frame_gray(self, frame_idx_0n: int):
        """Get uint8 grayscale pixel array for the given frame.

        Returns a numpy array of shape (H, W) covering the FULL Echo
        Wave display, not the B-mode ROI. Crop yourself via
        :attr:`b_mode_roi`.
        """
        import numpy as np

        cmd = self._require_open()
        if not (0 <= frame_idx_0n < self.n_frames):
            raise IndexError(
                f"frame_idx_0n {frame_idx_0n} out of range "
                f"[0, {self.n_frames})"
            )
        cmd.GoToFrame1n(frame_idx_0n + 1, True)
        return np.asarray(cmd.GetLoadedFrameGray, dtype=np.uint8)

    def extract_metadata(self) -> TelemedRecordingMeta:
        """Snapshot per-recording metadata for sidecar persistence."""
        return TelemedRecordingMeta.from_cmd(
            self._require_open(), source_tvd_path=self._opened
        )


# ---------- Module-level conveniences ----------


def connect() -> TelemedTvdReader:
    """Build + connect a :class:`TelemedTvdReader` in one call."""
    r = TelemedTvdReader()
    r.connect()
    return r


def _sidecar_h5_path(tvd_path: Path) -> Path:
    """Composite-suffix sidecar name: ``<stem>.tvd.h5``.

    Composite suffix (matches the ``.dnav-toc`` convention) so
    downstream tools walking ``*.h5`` don't accidentally pick these
    up as unrelated HDF5 data.
    """
    return tvd_path.with_suffix(tvd_path.suffix + ".h5")


def _is_network_path(p: Path) -> bool:
    """Detect Windows UNC paths and mapped network drives.

    UNC: starts with ``\\\\``. Mapped drives: we'd need ``net use`` to
    distinguish from local letters, which is heavier than warranted --
    so we conservatively only flag UNC here, plus drive letters not
    in ``{C}`` by default. Caller can override via the explicit
    ``copy_to_local`` flag.
    """
    s = str(p)
    if s.startswith("\\\\") or s.startswith("//"):
        return True
    # Heuristic: anything not on C: is treated as potentially-network
    # for the copy-to-local path. Praneeth's M:, S:, etc. are mapped
    # network shares in his setup.
    if len(s) >= 2 and s[1] == ":" and s[0].upper() != "C":
        return True
    return False


def _extract_one(
    tvd_path: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    *,
    reader: Optional[TelemedTvdReader] = None,
    frames: bool = True,
    compression: str = "gzip",
    compression_opts: int = 4,
    progress: bool = True,
) -> Path:
    """Extract one .tvd's timing + metadata + (optionally) frames to HDF5.

    Internal single-file primitive. Public callers should use
    :func:`export`, which accepts the same kwargs plus handles
    folders / lists and the network-drive copy-to-local dance.

    Output HDF5 schema (v1):

    * Root attributes (flat): ``n_frames``, ``full_frame_width``,
      ``full_frame_height``, ``roi_x1`` / ``roi_x2`` / ``roi_y1`` /
      ``roi_y2`` / ``roi_width`` / ``roi_height``,
      ``physical_dx_cm_per_px``, ``physical_dy_cm_per_px``,
      ``source_tvd_path``, ``extracted_at_iso``, ``schema_version``.
    * ``/timing/frame_idx_1n`` -- int32 (N,)
    * ``/timing/time_ms`` -- float64 (N,)
    * ``/timing/ifi_ms`` -- float64 (N,)
    * ``/frames/gray`` -- uint8 (N, H, W) [only when ``frames=True``]

    Args:
        tvd_path: Source .tvd file. Must be on a local drive --
            EchoWave's OpenFile fails on UNC / mapped-network paths
            (the :func:`export` wrapper handles this).
        out_path: Output HDF5 path. Defaults to ``<stem>.tvd.h5``
            next to the source.
        reader: Optional already-connected reader (amortise the COM
            connect step across many files).
        frames, compression, compression_opts, progress: As for
            :func:`export`.

    Returns:
        Path to the written HDF5 file.
    """
    import h5py
    import numpy as np

    p = Path(tvd_path)
    out = Path(out_path) if out_path is not None else _sidecar_h5_path(p)

    r = reader if reader is not None else connect()
    r.open(p)
    cmd = r._cmd
    meta = r.extract_metadata()
    n = meta.n_frames

    # Pre-allocate timing arrays (cheaper than building a DataFrame
    # per frame inside the hot loop).
    times = np.zeros(n, dtype=np.float64)
    frames_arr = None
    if frames:
        # Full-frame stack; consumer crops via root attrs.
        h = meta.full_frame_height
        w = meta.full_frame_width
        frames_arr = np.empty((n, h, w), dtype=np.uint8)

    # tqdm if available + requested; gracefully degrade to a silent
    # loop otherwise. The bar's ``desc`` carries the file stem so
    # batch logs are readable.
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(
                total=n,
                desc=p.stem,
                unit="frame",
                unit_scale=False,
                leave=False,
            )
        except ImportError:
            bar = None

    t0 = time.perf_counter()
    try:
        for i in range(1, n + 1):
            # load_frame_data only matters when we'll pull pixels.
            cmd.GoToFrame1n(i, frames)
            times[i - 1] = float(cmd.GetCurrentFrameTime)
            if frames:
                frames_arr[i - 1] = np.asarray(cmd.GetLoadedFrameGray, dtype=np.uint8)
            if bar is not None:
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()
    walk_s = time.perf_counter() - t0
    if progress:
        print(f"  walk: {walk_s:.1f} s  ({n / walk_s:.1f} fps)", flush=True)

    ifi = np.zeros(n, dtype=np.float64)
    ifi[1:] = np.diff(times)

    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as h5:
        for k, v in meta.to_flat_attrs().items():
            h5.attrs[k] = v
        tg = h5.create_group("timing")
        tg.create_dataset(
            "frame_idx_1n", data=np.arange(1, n + 1, dtype=np.int32)
        )
        tg.create_dataset("time_ms", data=times)
        tg.create_dataset("ifi_ms", data=ifi)
        if frames:
            fg = h5.create_group("frames")
            kwargs = {}
            if compression is not None:
                kwargs["compression"] = compression
                if compression == "gzip":
                    kwargs["compression_opts"] = compression_opts
            fg.create_dataset(
                "gray",
                data=frames_arr,
                chunks=(1, meta.full_frame_height, meta.full_frame_width),
                **kwargs,
            )
    return out


def _normalize_sources(
    source: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    recursive: bool,
    pattern: str,
) -> list[Path]:
    """Resolve ``source`` to a de-duplicated list of .tvd file paths.

    ``source`` may be: a single file path, a single directory, or an
    iterable mixing both. Directories are walked for ``pattern`` files
    (recursively when ``recursive=True``); files matching ``pattern``
    are taken as-is even if they wouldn't match the glob (caller
    explicitly named them). De-duplication is by ``Path.resolve()`` so
    overlapping roots / repeated entries don't double-process.
    """
    if isinstance(source, (str, Path)):
        entries = [Path(source)]
    else:
        entries = [Path(s) for s in source]

    seen: set = set()
    files: list[Path] = []
    for entry in entries:
        if entry.is_file():
            candidates = [entry]
        elif entry.is_dir():
            candidates = sorted(
                entry.rglob(pattern) if recursive else entry.glob(pattern)
            )
        else:
            # Non-existent or special: skip silently. The caller's
            # results dict will simply lack the entry.
            continue
        for fp in candidates:
            key = fp.resolve()
            if key in seen:
                continue
            seen.add(key)
            files.append(fp)
    return files


def export(
    source: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    recursive: bool = True,
    pattern: str = "*.tvd",
    skip_existing: bool = True,
    frames: bool = True,
    compression: str = "gzip",
    compression_opts: int = 4,
    copy_to_local: Optional[bool] = None,
    local_temp_root: Optional[Union[str, Path]] = None,
    progress: bool = True,
    progress_callback: Optional[Callable[[int, int, Path, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict:
    """Extract Telemed ``.tvd`` recording(s) to HDF5 sidecar(s).

    Single unified entry point. ``source`` may be:

    * A path to a single ``.tvd`` file.
    * A directory (walked for ``pattern`` files; ``recursive=True`` by
      default).
    * An iterable of any combination of the above.

    For each .tvd, a sibling ``<stem>.tvd.h5`` is written. With
    ``skip_existing=True`` (default), files whose sidecar already
    exists are skipped, so re-running on a partly-processed corpus
    picks up where the previous run left off.

    Opens **one** Echo Wave COM connection for the entire job (cheap
    per-file overhead).

    **Network-drive handling.** EchoWave's ``OpenFile`` fails on UNC /
    mapped-network paths. When ``copy_to_local`` is True (or ``None``
    -- the default -- and the source looks like a network path: non-C:
    drive letter, or starts with ``\\\\``), each source is copied to a
    unique subdir of ``local_temp_root`` (default: system temp dir),
    processed there, and the resulting HDF5 is copied back next to the
    original ``.tvd``. The local staging dir is cleaned up after each
    file (even on error).

    Args:
        source: File path, directory, or iterable of either. See
            above for shape semantics.
        recursive: If True (default), recurse into subdirectories
            when walking directories. Ignored for individual file
            entries.
        pattern: Glob for file selection inside walked directories
            (default ``"*.tvd"``). Ignored for individual file
            entries.
        skip_existing: If True (default), skip files whose ``.tvd.h5``
            sidecar already exists at the destination.
        frames: If True (default), include raw grayscale frames in
            the HDF5 sidecar. Pass False for a fast timing-only
            extraction (~3x faster, much smaller output).
        compression: HDF5 compression for the frames dataset.
            ``"gzip"`` (default) is lossless and ~5x smaller for
            typical ultrasound; ``"lzf"`` is faster but ~30% larger;
            ``None`` skips compression entirely.
        compression_opts: gzip level [0-9]; default 4.
        copy_to_local: Force-on/off the network-aware copy. ``None``
            (default) auto-detects per source path.
        local_temp_root: Where to stage local copies. ``None`` uses
            the system temp directory.
        progress: If True (default), print ``[i/N] <filename>`` before
            each file and let the per-file tqdm bar render. False
            suppresses both.
        progress_callback: Optional ``fn(idx, total, path, status)``
            for machine-readable progress -- matches the
            ``dustrack.batch`` convention.
        cancel_check: Optional zero-arg callable polled between
            files. If truthy, the loop exits early; the partial
            results dict is returned; any in-flight local staging
            dir is cleaned up.

    Returns:
        ``{path: status}`` where status is ``"built"`` (just
        extracted), ``"hit"`` (skipped existing), or ``f"error: {msg}"``.

    Examples::

        # One file
        telemed.export("C:/data/scan.tvd")

        # One folder (recursive walk for *.tvd)
        telemed.export("M:/data/pia02")

        # Mix of folders and individual files
        telemed.export([
            "M:/data/pia02",
            "M:/data/pia03",
            "C:/scratch/single.tvd",
        ])

        # Timing only -- fast pass for bulk metadata extraction
        telemed.export("M:/data/pia02", frames=False)
    """
    files = _normalize_sources(source, recursive=recursive, pattern=pattern)
    if not files:
        return {}

    temp_root = Path(local_temp_root) if local_temp_root else Path(tempfile.gettempdir())
    temp_root.mkdir(parents=True, exist_ok=True)

    reader = connect()
    results: dict[str, str] = {}
    total = len(files)
    for idx, src_tvd in enumerate(files):
        if cancel_check is not None and cancel_check():
            break
        dst_h5 = _sidecar_h5_path(src_tvd)
        if skip_existing and dst_h5.exists():
            results[str(src_tvd)] = "hit"
            if progress:
                print(f"[{idx + 1}/{total}] {src_tvd.name}  (hit, skip)", flush=True)
            if progress_callback is not None:
                progress_callback(idx, total, src_tvd, "hit")
            continue

        if progress:
            print(f"[{idx + 1}/{total}] {src_tvd.name}", flush=True)

        use_copy = copy_to_local
        if use_copy is None:
            use_copy = _is_network_path(src_tvd)

        local_tvd: Optional[Path] = None
        local_h5: Optional[Path] = None
        try:
            if use_copy:
                # Stage into a unique subdir of temp_root so concurrent
                # runs don't collide on basenames.
                stage = Path(tempfile.mkdtemp(prefix="telemed_tvd_", dir=temp_root))
                local_tvd = stage / src_tvd.name
                shutil.copy2(src_tvd, local_tvd)
                local_h5 = _sidecar_h5_path(local_tvd)
                _extract_one(
                    local_tvd,
                    out_path=local_h5,
                    reader=reader,
                    frames=frames,
                    compression=compression,
                    compression_opts=compression_opts,
                    progress=progress,
                )
                # Copy result back next to the original .tvd.
                shutil.copy2(local_h5, dst_h5)
            else:
                _extract_one(
                    src_tvd,
                    out_path=dst_h5,
                    reader=reader,
                    frames=frames,
                    compression=compression,
                    compression_opts=compression_opts,
                    progress=progress,
                )
            results[str(src_tvd)] = "built"
        except Exception as e:  # noqa: BLE001
            results[str(src_tvd)] = f"error: {e}"
        finally:
            # Always clean up the temp staging dir, even on error.
            if local_tvd is not None:
                try:
                    shutil.rmtree(local_tvd.parent, ignore_errors=True)
                except Exception:  # noqa: BLE001
                    pass

        if progress_callback is not None:
            progress_callback(idx, total, src_tvd, results[str(src_tvd)])
    return results
