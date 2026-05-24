"""Tests for ``immersionlab.telemed.export`` input-shape normalisation.

The COM-based extraction path itself is not unit-testable without a
running Echo Wave II instance, but the source-normalisation that
underlies ``telemed.export(source)`` (file / folder / list / mix) is
pure Python and worth pinning so the dispatch doesn't silently
regress.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _make_synthetic_tvd_files(root: Path, names: list[str]) -> list[Path]:
    """Create empty placeholder .tvd files so glob walking sees them."""
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for n in names:
        p = root / n
        p.write_bytes(b"")  # empty -- never opened by these tests
        paths.append(p)
    return paths


def test_normalize_single_file(tmp_path):
    from immersionlab.telemed._extract import _normalize_sources

    files = _make_synthetic_tvd_files(tmp_path, ["a.tvd"])
    out = _normalize_sources(files[0], recursive=True, pattern="*.tvd")
    assert out == files


def test_normalize_single_folder_recursive(tmp_path):
    from immersionlab.telemed._extract import _normalize_sources

    _make_synthetic_tvd_files(tmp_path, ["a.tvd"])
    _make_synthetic_tvd_files(tmp_path / "sub", ["b.tvd"])
    out = _normalize_sources(tmp_path, recursive=True, pattern="*.tvd")
    names = sorted(p.name for p in out)
    assert names == ["a.tvd", "b.tvd"]


def test_normalize_single_folder_non_recursive(tmp_path):
    from immersionlab.telemed._extract import _normalize_sources

    _make_synthetic_tvd_files(tmp_path, ["a.tvd"])
    _make_synthetic_tvd_files(tmp_path / "sub", ["b.tvd"])
    out = _normalize_sources(tmp_path, recursive=False, pattern="*.tvd")
    names = [p.name for p in out]
    assert names == ["a.tvd"]


def test_normalize_list_of_files(tmp_path):
    from immersionlab.telemed._extract import _normalize_sources

    files = _make_synthetic_tvd_files(tmp_path, ["a.tvd", "b.tvd", "c.tvd"])
    out = _normalize_sources(files, recursive=True, pattern="*.tvd")
    assert sorted(p.name for p in out) == ["a.tvd", "b.tvd", "c.tvd"]


def test_normalize_list_of_folders(tmp_path):
    from immersionlab.telemed._extract import _normalize_sources

    fa = tmp_path / "fa"
    fb = tmp_path / "fb"
    _make_synthetic_tvd_files(fa, ["a1.tvd", "a2.tvd"])
    _make_synthetic_tvd_files(fb, ["b1.tvd"])
    out = _normalize_sources([fa, fb], recursive=True, pattern="*.tvd")
    assert sorted(p.name for p in out) == ["a1.tvd", "a2.tvd", "b1.tvd"]


def test_normalize_mixed_files_and_folders(tmp_path):
    from immersionlab.telemed._extract import _normalize_sources

    folder = tmp_path / "data"
    standalone = tmp_path / "loose.tvd"
    _make_synthetic_tvd_files(folder, ["x.tvd"])
    standalone.write_bytes(b"")
    out = _normalize_sources([folder, standalone], recursive=True, pattern="*.tvd")
    assert sorted(p.name for p in out) == ["loose.tvd", "x.tvd"]


def test_normalize_dedupes_overlapping_roots(tmp_path):
    """Passing the same folder twice -- or nested parents -- should
    yield each file once."""
    from immersionlab.telemed._extract import _normalize_sources

    sub = tmp_path / "sub"
    _make_synthetic_tvd_files(sub, ["a.tvd"])
    # parent + child + the file itself, plus the file again
    out = _normalize_sources(
        [tmp_path, sub, sub / "a.tvd", sub / "a.tvd"],
        recursive=True,
        pattern="*.tvd",
    )
    assert len(out) == 1
    assert out[0].name == "a.tvd"


def test_normalize_skips_nonexistent_entries(tmp_path):
    """Non-existent paths should be silently skipped (not raise) --
    they just won't show up in the results dict."""
    from immersionlab.telemed._extract import _normalize_sources

    real = _make_synthetic_tvd_files(tmp_path, ["real.tvd"])[0]
    out = _normalize_sources(
        [real, tmp_path / "ghost.tvd", tmp_path / "no_such_folder"],
        recursive=True,
        pattern="*.tvd",
    )
    assert out == [real]


def test_normalize_pattern_kwarg(tmp_path):
    from immersionlab.telemed._extract import _normalize_sources

    _make_synthetic_tvd_files(tmp_path, ["a.tvd", "b.dat", "c.tvd"])
    out = _normalize_sources(tmp_path, recursive=True, pattern="*.dat")
    assert [p.name for p in out] == ["b.dat"]


def test_is_network_path_heuristic():
    from immersionlab.telemed._extract import _is_network_path

    assert _is_network_path(Path(r"\\server\share\file.tvd"))
    assert _is_network_path(Path("//server/share/file.tvd"))
    assert _is_network_path(Path("M:/data/file.tvd"))
    assert _is_network_path(Path("s:/data/file.tvd"))
    assert not _is_network_path(Path("C:/data/file.tvd"))
    assert not _is_network_path(Path("c:/data/file.tvd"))


def test_export_h5_with_no_matches_returns_empty(tmp_path):
    """``export_h5()`` on an empty folder is a no-op (no COM connection
    is attempted, so this test runs without EchoWave)."""
    from immersionlab import telemed

    empty = tmp_path / "empty_dir"
    empty.mkdir()
    out = telemed.export_h5(empty)
    assert out == {}


# ---------- Background-prefetch staging primitives ----------


def test_stage_one_no_copy_passthrough(tmp_path):
    """``use_copy=False`` returns a _StagedFile whose local paths are
    just the originals (no file copy made, no stage_dir to clean up)."""
    from immersionlab.telemed._extract import _stage_one

    src = tmp_path / "x.tvd"
    src.write_bytes(b"abc")
    dst = tmp_path / "x.tvd.h5"
    staged = _stage_one(src, dst, use_copy=False, temp_root=tmp_path)
    assert staged.src_tvd == src
    assert staged.dst_h5 == dst
    assert staged.local_tvd == src
    assert staged.local_h5 == dst
    assert staged.stage_dir is None


def test_stage_one_with_copy_makes_local(tmp_path):
    """``use_copy=True`` copies the source into a fresh temp dir and
    points local_tvd / local_h5 there."""
    from immersionlab.telemed._extract import _stage_one

    src = tmp_path / "src.tvd"
    src.write_bytes(b"hello")
    dst = tmp_path / "out" / "src.tvd.h5"
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    staged = _stage_one(src, dst, use_copy=True, temp_root=temp_root)
    assert staged.stage_dir is not None
    assert staged.stage_dir.is_dir()
    assert staged.stage_dir.parent == temp_root
    assert staged.local_tvd.is_file()
    assert staged.local_tvd.read_bytes() == b"hello"
    assert staged.local_h5 == staged.local_tvd.with_suffix(".tvd.h5")


def test_unstage_one_upload_true_copies_and_cleans(tmp_path):
    """``upload=True`` copies the local .h5 to dst_h5 and removes the
    staging directory."""
    from immersionlab.telemed._extract import _StagedFile, _unstage_one

    stage = tmp_path / "stage"
    stage.mkdir()
    local_tvd = stage / "x.tvd"
    local_h5 = stage / "x.tvd.h5"
    local_tvd.write_bytes(b"")
    local_h5.write_bytes(b"result-bytes")
    dst = tmp_path / "dst" / "x.tvd.h5"
    dst.parent.mkdir()
    staged = _StagedFile(
        src_tvd=Path("M:/fake/x.tvd"), dst_h5=dst,
        local_tvd=local_tvd, local_h5=local_h5, stage_dir=stage,
    )
    _unstage_one(staged, upload=True)
    assert dst.read_bytes() == b"result-bytes"
    assert not stage.exists()


def test_unstage_one_upload_false_skips_copy(tmp_path):
    """``upload=False`` cleans up local temp but does NOT copy back --
    used when extract failed and there's nothing valid to upload."""
    from immersionlab.telemed._extract import _StagedFile, _unstage_one

    stage = tmp_path / "stage"
    stage.mkdir()
    local_h5 = stage / "x.tvd.h5"
    local_h5.write_bytes(b"partial")
    dst = tmp_path / "dst" / "x.tvd.h5"
    dst.parent.mkdir()
    staged = _StagedFile(
        src_tvd=Path("M:/fake/x.tvd"), dst_h5=dst,
        local_tvd=stage / "x.tvd", local_h5=local_h5, stage_dir=stage,
    )
    _unstage_one(staged, upload=False)
    assert not dst.exists()
    assert not stage.exists()


# ---------- Schema v2 ParamGet sweep ----------


class _FakeCmd:
    """Minimal stand-in for the EchoWave COM object exposing only the
    ParamGet* methods the v2 sweep touches.

    Constructed with two dicts: ``values`` (param_id -> value to return
    by kind) and ``fail`` (param_id -> exception class to raise).
    Unknown IDs raise AttributeError-shaped failure so absent-vs-
    silently-fail is distinguishable in the test.
    """

    def __init__(self, values=None, fail=None):
        self._values = values or {}
        self._fail = fail or {}

    def _resolve(self, kind, param_id):
        if param_id in self._fail:
            raise self._fail[param_id]("simulated COM failure")
        if (kind, param_id) in self._values:
            return self._values[(kind, param_id)]
        if param_id in self._values:
            return self._values[param_id]
        # Mimic .NET's COMException-ish behaviour for unknown IDs.
        raise RuntimeError(f"no value configured for ({kind}, {param_id})")

    def ParamGetInt(self, pid):
        return self._resolve("int", pid)

    def ParamGetBool(self, pid):
        return self._resolve("bool", pid)

    def ParamGetString(self, pid):
        return self._resolve("string", pid)


def test_param_specs_have_unique_ids_and_names():
    from immersionlab.telemed._extract import _PARAM_SPECS

    ids = [s.param_id for s in _PARAM_SPECS]
    names = [s.name for s in _PARAM_SPECS]
    assert len(set(ids)) == len(ids), "duplicate param_id in _PARAM_SPECS"
    assert len(set(names)) == len(names), "duplicate name in _PARAM_SPECS"
    assert all(s.kind in {"int", "bool", "string"} for s in _PARAM_SPECS)


def test_safe_param_get_success_by_kind():
    from immersionlab.telemed._extract import _ParamSpec, _safe_param_get

    cmd = _FakeCmd(values={
        ("int", 305): 60,
        ("bool", 177): True,
        ("string", 918): "L18-10",
    })
    assert _safe_param_get(cmd, _ParamSpec("b_depth", 305, "int")) == 60
    assert _safe_param_get(cmd, _ParamSpec("b_thi", 177, "bool")) is True
    assert _safe_param_get(cmd, _ParamSpec("probe_name", 918, "string")) == "L18-10"


def test_safe_param_get_failure_returns_none():
    """Any exception from the COM call -> None (don't propagate)."""
    from immersionlab.telemed._extract import _ParamSpec, _safe_param_get

    cmd = _FakeCmd(fail={305: RuntimeError})
    assert _safe_param_get(cmd, _ParamSpec("b_depth", 305, "int")) is None


def test_collect_params_skips_failures(monkeypatch):
    """The full sweep records each successful probe under
    ``param_<name>`` and silently omits failed ones."""
    from immersionlab.telemed import _extract

    monkeypatch.setattr(_extract, "_PARAM_SPECS", (
        _extract._ParamSpec("probe_name", 918, "string"),
        _extract._ParamSpec("b_depth",    305, "int"),
        _extract._ParamSpec("b_thi",      177, "bool"),
    ))
    cmd = _FakeCmd(
        values={("string", 918): "L18-10", ("bool", 177): True},
        fail={305: RuntimeError},
    )
    out = _extract._collect_params(cmd)
    assert out == {"param_probe_name": "L18-10", "param_b_thi": True}


def test_collect_params_all_fail_returns_empty():
    from immersionlab.telemed import _extract

    # Fail every id in the production sweep.
    cmd = _FakeCmd(fail={s.param_id: RuntimeError for s in _extract._PARAM_SPECS})
    assert _extract._collect_params(cmd) == {}


# ---------- Multi-image ROI capture (schema v4) ----------


class _FakeRoiCmd:
    """COM stand-in for the ROI/PhysicalDelta calls used by
    ``TelemedRoi.from_cmd``. ``rois`` maps img_id ->
    {x1,x2,y1,y2,dx,dy}; absent img_ids raise."""

    def __init__(self, rois):
        self._rois = rois

    def _get(self, img_id, field):
        if img_id not in self._rois:
            raise RuntimeError(f"img_id {img_id} not active")
        return self._rois[img_id][field]

    def GetUltrasoundX1(self, img_id): return self._get(img_id, "x1")
    def GetUltrasoundX2(self, img_id): return self._get(img_id, "x2")
    def GetUltrasoundY1(self, img_id): return self._get(img_id, "y1")
    def GetUltrasoundY2(self, img_id): return self._get(img_id, "y2")
    def GetUltrasoundPhysicalDeltaX(self, img_id): return self._get(img_id, "dx")
    def GetUltrasoundPhysicalDeltaY(self, img_id): return self._get(img_id, "dy")


def test_telemed_roi_from_cmd_returns_none_when_panel_absent():
    """img_ids the device isn't currently rendering raise from the COM;
    ``TelemedRoi.from_cmd`` must return ``None`` rather than propagating."""
    from immersionlab.telemed._extract import TelemedRoi

    cmd = _FakeRoiCmd({1: dict(x1=10, x2=50, y1=5, y2=45, dx=0.01, dy=0.02)})
    assert TelemedRoi.from_cmd(cmd, img_id=1) is not None
    assert TelemedRoi.from_cmd(cmd, img_id=2) is None


def test_telemed_roi_from_cmd_rejects_degenerate_rect():
    """A panel-absent sentinel sometimes comes back as an inverted or
    negative rect rather than a raise. Treat that as 'not present' too."""
    from immersionlab.telemed._extract import TelemedRoi

    cmd = _FakeRoiCmd({1: dict(x1=10, x2=5, y1=5, y2=45, dx=0.01, dy=0.02)})
    assert TelemedRoi.from_cmd(cmd, img_id=1) is None


def test_telemed_roi_from_cmd_rejects_zero_rect_sentinel():
    """**Regression guard**: AutoInt1 returns ``(0,0,0,0)`` for inactive
    panels rather than raising (observed 2026-05-24 on the usl02
    single-probe ``_metadata_probe`` output -- img_ids 2/3/4 all came
    back as ``(0,0,0,0)``). An earlier ``x2 < x1`` validator wrongly
    accepted this sentinel as a 1x1 ROI, making single-probe
    recordings look like quad-probe."""
    from immersionlab.telemed._extract import TelemedRoi

    cmd = _FakeRoiCmd({
        1: dict(x1=0, x2=0, y1=0, y2=0, dx=0.0, dy=0.0),
    })
    assert TelemedRoi.from_cmd(cmd, img_id=1) is None


def test_telemed_roi_from_cmd_rejects_single_pixel_rect():
    """``x1==x2`` (and ``y1==y2``) gives a 1x1 inclusive rect -- not
    plausible B-mode geometry, so reject."""
    from immersionlab.telemed._extract import TelemedRoi

    cmd = _FakeRoiCmd({1: dict(x1=10, x2=10, y1=5, y2=5, dx=0.01, dy=0.02)})
    assert TelemedRoi.from_cmd(cmd, img_id=1) is None


def test_collect_b_mode_rois_single_probe():
    from immersionlab.telemed._extract import _collect_b_mode_rois

    cmd = _FakeRoiCmd({1: dict(x1=10, x2=50, y1=5, y2=45, dx=0.01, dy=0.02)})
    out = _collect_b_mode_rois(cmd)
    assert set(out) == {1}
    assert out[1].img_id == 1
    assert out[1].physical_dx_cm_per_px == 0.01


def test_collect_b_mode_rois_usl02_shape():
    """**Cohort guard**: matches the exact AutoInt1 output recorded for
    ``usl02_s005_02_003.tvd`` on 2026-05-24 -- B at (73, 1481, 43, 600)
    full-width, and B2/B3/B4 all return the ``(0,0,0,0)`` sentinel. Must
    classify as single-probe (n=1), not quad-probe."""
    from immersionlab.telemed._extract import _collect_b_mode_rois

    cmd = _FakeRoiCmd({
        1: dict(x1=73, x2=1481, y1=43, y2=600, dx=0.009166, dy=0.009166),
        2: dict(x1=0,  x2=0,    y1=0,  y2=0,   dx=0.0,      dy=0.0),
        3: dict(x1=0,  x2=0,    y1=0,  y2=0,   dx=0.0,      dy=0.0),
        4: dict(x1=0,  x2=0,    y1=0,  y2=0,   dx=0.0,      dy=0.0),
    })
    out = _collect_b_mode_rois(cmd)
    assert set(out) == {1}
    assert out[1].width == 1481 - 73 + 1
    assert out[1].height == 600 - 43 + 1


def test_collect_b_mode_rois_pia02_dual_shape():
    """**Cohort guard**: matches the exact AutoInt1 output recorded for
    ``pia02_s018_003.tvd`` (dual-probe) on 2026-05-24 -- B at
    (73, 777, 43, 600), B2 at (777, 1481, 43, 600), B3/B4 zero-rect.
    Must classify as dual-probe (n=2) with both halves 705 wide."""
    from immersionlab.telemed._extract import _collect_b_mode_rois

    cmd = _FakeRoiCmd({
        1: dict(x1=73,  x2=777,  y1=43, y2=600, dx=0.009166, dy=0.009166),
        2: dict(x1=777, x2=1481, y1=43, y2=600, dx=0.009166, dy=0.009166),
        3: dict(x1=0,   x2=0,    y1=0,  y2=0,   dx=0.0,      dy=0.0),
        4: dict(x1=0,   x2=0,    y1=0,  y2=0,   dx=0.0,      dy=0.0),
    })
    out = _collect_b_mode_rois(cmd)
    assert set(out) == {1, 2}
    assert out[1].width == 705 and out[2].width == 705


def test_collect_b_mode_rois_dual_probe():
    """Active img_ids 1 + 2 -> both ROIs returned with per-panel
    physical resolutions preserved."""
    from immersionlab.telemed._extract import _collect_b_mode_rois

    cmd = _FakeRoiCmd({
        1: dict(x1=10, x2=50, y1=5, y2=45, dx=0.012, dy=0.013),
        2: dict(x1=60, x2=100, y1=5, y2=45, dx=0.011, dy=0.014),
    })
    out = _collect_b_mode_rois(cmd)
    assert set(out) == {1, 2}
    assert out[1].physical_dx_cm_per_px == 0.012
    assert out[2].physical_dx_cm_per_px == 0.011


def test_recording_meta_to_flat_attrs_single_probe():
    """v1a5 ``to_flat_attrs`` flattens img_id=1 to ``roi1_*`` /
    ``physical_d{x,y}1_cm_per_px``, emits ``n_b_images`` +
    ``image_d{x,y}_cm_per_px``, and merges ``param_*`` in as-is."""
    from immersionlab.telemed._extract import TelemedRecordingMeta, TelemedRoi

    meta = TelemedRecordingMeta(
        n_frames=100,
        full_frame_width=1554,
        full_frame_height=601,
        b_mode_rois={1: TelemedRoi(
            img_id=1, x1=73, x2=777, y1=43, y2=600, width=705, height=558,
            physical_dx_cm_per_px=0.012, physical_dy_cm_per_px=0.013,
        )},
        image_dx_cm_per_px=0.00896,
        image_dy_cm_per_px=0.00896,
        source_tvd_path="C:/x.tvd",
        extracted_at_iso="2026-05-23T00:00:00",
        params={"param_probe_name": "L18-10", "param_b_depth": 60},
    )
    attrs = meta.to_flat_attrs()
    assert attrs["schema_version"] == "v1a5"
    assert attrs["n_b_images"] == 1
    assert attrs["roi1_x1"] == 73
    assert attrs["roi1_width"] == 705
    assert attrs["physical_dx1_cm_per_px"] == 0.012
    assert attrs["physical_dy1_cm_per_px"] == 0.013
    assert attrs["image_dx_cm_per_px"] == 0.00896
    assert attrs["image_dy_cm_per_px"] == 0.00896
    assert attrs["param_probe_name"] == "L18-10"
    assert attrs["param_b_depth"] == 60
    # Nested dicts shouldn't leak through.
    assert "b_mode_rois" not in attrs and "params" not in attrs
    # The legacy unprefixed roi_* / physical_d*_cm_per_px keys are
    # gone in v1a4+ (clean break -- no production sidecars on disk).
    assert "roi_x1" not in attrs
    assert "physical_dx_cm_per_px" not in attrs
    # Inner-image autocrop bounds are NOT in the schema -- detection
    # runs at encode time. No image_roi* attrs should appear here.
    assert "image_roi1_x1" not in attrs
    assert "image_roi1_width" not in attrs


def test_recording_meta_to_flat_attrs_image_d_omitted_when_none():
    """If b_depth wasn't captured, image_d{x,y} is None and the attrs
    are skipped entirely (rather than written as a sentinel)."""
    from immersionlab.telemed._extract import TelemedRecordingMeta, TelemedRoi

    meta = TelemedRecordingMeta(
        n_frames=100,
        full_frame_width=1554,
        full_frame_height=601,
        b_mode_rois={1: TelemedRoi(
            img_id=1, x1=73, x2=777, y1=43, y2=600, width=705, height=558,
            physical_dx_cm_per_px=0.012, physical_dy_cm_per_px=0.013,
        )},
        image_dx_cm_per_px=None,
        image_dy_cm_per_px=None,
        source_tvd_path="C:/x.tvd",
        extracted_at_iso="2026-05-23T00:00:00",
    )
    attrs = meta.to_flat_attrs()
    assert "image_dx_cm_per_px" not in attrs
    assert "image_dy_cm_per_px" not in attrs


def test_recording_meta_to_flat_attrs_dual_probe():
    """Dual-probe -> two ``roi{N}_*`` blocks + two
    ``physical_d{x,y}{N}_cm_per_px`` pairs + ``n_b_images=2``."""
    from immersionlab.telemed._extract import TelemedRecordingMeta, TelemedRoi

    meta = TelemedRecordingMeta(
        n_frames=100,
        full_frame_width=1554,
        full_frame_height=601,
        b_mode_rois={
            1: TelemedRoi(1, 73, 425, 43, 600, 353, 558, 0.012, 0.013),
            2: TelemedRoi(2, 429, 777, 43, 600, 349, 558, 0.011, 0.014),
        },
        image_dx_cm_per_px=0.00896,
        image_dy_cm_per_px=0.00896,
        source_tvd_path="C:/x.tvd",
        extracted_at_iso="2026-05-23T00:00:00",
    )
    attrs = meta.to_flat_attrs()
    assert attrs["n_b_images"] == 2
    assert attrs["roi1_x1"] == 73 and attrs["roi2_x1"] == 429
    assert attrs["physical_dx1_cm_per_px"] == 0.012
    assert attrs["physical_dx2_cm_per_px"] == 0.011
    assert attrs["physical_dy1_cm_per_px"] == 0.013
    assert attrs["physical_dy2_cm_per_px"] == 0.014


def test_unstage_one_no_copy_is_noop(tmp_path):
    """``stage_dir=None`` means no local copy was made -- unstage
    should be a no-op (don't try to delete the source!)."""
    from immersionlab.telemed._extract import _StagedFile, _unstage_one

    src = tmp_path / "x.tvd"
    src.write_bytes(b"original")
    dst = tmp_path / "x.tvd.h5"
    dst.write_bytes(b"sidecar")
    staged = _StagedFile(
        src_tvd=src, dst_h5=dst,
        local_tvd=src, local_h5=dst, stage_dir=None,
    )
    _unstage_one(staged, upload=True)
    # Both files survive unchanged.
    assert src.read_bytes() == b"original"
    assert dst.read_bytes() == b"sidecar"


# ---------- Postprocess hook ----------


class TestPostprocessHook:
    """``export_h5(postprocess=...)`` lets the dispatcher attach a
    richer per-file tail (encode + TOC + upload) without forking the
    extract loop. Tests stub the COM-bound ``_extract_one`` so we
    don't need EchoWave running."""

    def _patch_extract(self, monkeypatch, *, raise_for=None):
        """Replace ``_extract_one`` with a stub that touches the
        local_h5 output and (optionally) raises for named files."""
        from immersionlab.telemed import _extract

        raise_for = raise_for or set()

        def _fake(tvd_path, out_path=None, **kwargs):
            from pathlib import Path

            if Path(tvd_path).name in raise_for:
                raise RuntimeError(f"forced failure for {Path(tvd_path).name}")
            out = Path(out_path) if out_path is not None else (
                Path(str(tvd_path) + ".h5")
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"fake h5")
            return out

        monkeypatch.setattr(_extract, "_extract_one", _fake)
        # Also stub connect() so we don't try to attach to EchoWave.
        monkeypatch.setattr(_extract, "connect", lambda: None)

    def test_default_postprocess_uploads_and_cleans_up(self, tmp_path, monkeypatch):
        """No ``postprocess`` arg -> legacy ``_unstage_one`` behaviour:
        the local .h5 lands at ``dst_h5`` (sidecar of src .tvd)."""
        from immersionlab.telemed import export_h5

        self._patch_extract(monkeypatch)
        src = tmp_path / "rec.tvd"
        src.write_bytes(b"")
        # Force the copy-to-local path so unstage actually has work
        # to do (otherwise local_h5 IS dst_h5 and the test is vacuous).
        results = export_h5(src, copy_to_local=True, progress=False)
        assert results[str(src)] == "built"
        assert (tmp_path / "rec.tvd.h5").exists()

    def test_custom_postprocess_called_with_success_true(self, tmp_path, monkeypatch):
        from immersionlab.telemed import export_h5

        self._patch_extract(monkeypatch)
        src = tmp_path / "rec.tvd"
        src.write_bytes(b"")
        seen: list = []

        def _hook(staged, success):
            seen.append((staged.src_tvd, success))

        export_h5(
            src, copy_to_local=False, progress=False, postprocess=_hook,
        )
        # Pool exits at end of export_h5; wait for any in-flight
        # bg submits to drain via the context-manager exit.
        assert len(seen) == 1
        assert seen[0] == (src, True)

    def test_custom_postprocess_called_with_success_false_on_extract_failure(
        self, tmp_path, monkeypatch,
    ):
        from immersionlab.telemed import export_h5

        self._patch_extract(monkeypatch, raise_for={"bad.tvd"})
        bad = tmp_path / "bad.tvd"
        bad.write_bytes(b"")
        seen: list = []

        def _hook(staged, success):
            seen.append((staged.src_tvd, success))

        results = export_h5(
            bad, copy_to_local=False, progress=False, postprocess=_hook,
        )
        assert results[str(bad)].startswith("error:")
        assert seen == [(bad, False)]

    def test_postprocess_runs_across_multiple_files(self, tmp_path, monkeypatch):
        from immersionlab.telemed import export_h5

        self._patch_extract(monkeypatch)
        a = tmp_path / "a.tvd"
        b = tmp_path / "b.tvd"
        a.write_bytes(b"")
        b.write_bytes(b"")
        seen: list = []
        import threading
        lock = threading.Lock()

        def _hook(staged, success):
            with lock:
                seen.append((staged.src_tvd.name, success))

        export_h5(
            [a, b], copy_to_local=False, progress=False, postprocess=_hook,
        )
        assert sorted(seen) == [("a.tvd", True), ("b.tvd", True)]
