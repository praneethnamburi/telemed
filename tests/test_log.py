"""Tests for ``telemed.Log``.

The Log class loads the HDF5 sidecar produced by ``telemed.export``.
Most tests run against a synthetic HDF5 built inside the test (so
they work without the C:/data/temp2 fixture); one test exercises
the real fixture when present.
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

# Force Agg before any matplotlib import in this module (the view()
# test needs a non-interactive backend).
os.environ.setdefault("MPLBACKEND", "Agg")


def _make_synthetic_h5(path: Path, n_frames: int = 5, h: int = 64, w: int = 96,
                       include_frames: bool = True,
                       schema_version: int | str = "v1",
                       params: dict | None = None,
                       rois: dict[int, dict] | None = None,
                       image_dx: float | None = None,
                       image_dy: float | None = None,
                       frames_data=None) -> Path:
    """Write a minimal Telemed-shape HDF5 sidecar with synthetic data.

    Mirrors the schema written by ``export_h5``. ``schema_version``
    accepts either a string ("v1a5", "v1a4", ...) for the alpha track
    or an int (1-4 = v1a1-v1a4) for legacy. ``params`` keys are
    stored under ``param_*`` root attrs (v1a2+).

    ``rois``: optional ``{img_id: {x1,x2,y1,y2,width,height,dx,dy}}``
    map. v1a4+ writes per-img_id ``roi{N}_*`` /
    ``physical_d{x,y}{N}_cm_per_px`` blocks; v1a1-v1a3 use the legacy
    unprefixed block (img_id=1 only). Defaults to a single img_id=1
    ROI of (10..50, 5..45) at 0.01 cm/px.

    ``image_dx`` / ``image_dy``: optional v1a5+ display-scale root
    attrs (cm/px). Omitted from the synthetic sidecar when None.

    ``frames_data``: optional explicit ``(n_frames, h, w)`` uint8
    array to use instead of the synthetic gradient (handy for
    autocrop tests that need a margin/inner contrast).
    """
    if rois is None:
        rois = {1: dict(x1=10, x2=50, y1=5, y2=45, width=41, height=41,
                        dx=0.01, dy=0.01)}
    # Normalise legacy int versions for the per-img_id-block predicate.
    is_per_img_id = isinstance(schema_version, str) or (
        isinstance(schema_version, int) and schema_version >= 4
    )
    # Display-scale image_d{x,y} attrs land in the sidecar for the
    # public "v1" schema and for the late-alpha "v1a5" track only.
    writes_image_d = schema_version == "v1" or (
        isinstance(schema_version, str) and schema_version >= "v1a5"
    )
    times = np.cumsum([0.0] + [14.5 + 0.5 * (i % 3) for i in range(n_frames - 1)])
    ifi = np.zeros(n_frames)
    ifi[1:] = np.diff(times)
    with h5py.File(path, "w") as h5:
        h5.attrs["n_frames"] = n_frames
        h5.attrs["full_frame_width"] = w
        h5.attrs["full_frame_height"] = h
        if is_per_img_id:
            h5.attrs["n_b_images"] = len(rois)
            for img_id, r in rois.items():
                h5.attrs[f"roi{img_id}_x1"] = r["x1"]
                h5.attrs[f"roi{img_id}_x2"] = r["x2"]
                h5.attrs[f"roi{img_id}_y1"] = r["y1"]
                h5.attrs[f"roi{img_id}_y2"] = r["y2"]
                h5.attrs[f"roi{img_id}_width"] = r["width"]
                h5.attrs[f"roi{img_id}_height"] = r["height"]
                h5.attrs[f"physical_dx{img_id}_cm_per_px"] = r["dx"]
                h5.attrs[f"physical_dy{img_id}_cm_per_px"] = r["dy"]
        else:
            r = rois[1]
            h5.attrs["roi_x1"] = r["x1"]
            h5.attrs["roi_x2"] = r["x2"]
            h5.attrs["roi_y1"] = r["y1"]
            h5.attrs["roi_y2"] = r["y2"]
            h5.attrs["roi_width"] = r["width"]
            h5.attrs["roi_height"] = r["height"]
            h5.attrs["physical_dx_cm_per_px"] = r["dx"]
            h5.attrs["physical_dy_cm_per_px"] = r["dy"]
        if writes_image_d and image_dx is not None:
            h5.attrs["image_dx_cm_per_px"] = image_dx
        if writes_image_d and image_dy is not None:
            h5.attrs["image_dy_cm_per_px"] = image_dy
        h5.attrs["source_tvd_path"] = "C:/synthetic/test.tvd"
        h5.attrs["extracted_at_iso"] = "2026-05-23T00:00:00"
        h5.attrs["schema_version"] = schema_version
        if params:
            for k, v in params.items():
                h5.attrs[f"param_{k}"] = v
        tg = h5.create_group("timing")
        tg.create_dataset("frame_idx_1n", data=np.arange(1, n_frames + 1, dtype=np.int32))
        tg.create_dataset("time_ms", data=times)
        tg.create_dataset("ifi_ms", data=ifi)
        if include_frames:
            if frames_data is not None:
                arr = np.asarray(frames_data, dtype=np.uint8)
                assert arr.shape == (n_frames, h, w), (
                    f"frames_data shape {arr.shape} != ({n_frames}, {h}, {w})"
                )
            else:
                # gradient-fill frames so each one is distinguishable
                arr = np.zeros((n_frames, h, w), dtype=np.uint8)
                for i in range(n_frames):
                    arr[i] = (np.linspace(0, 255, w, dtype=np.float32)[None, :]
                              .repeat(h, axis=0) * (1.0 - i / max(n_frames - 1, 1))).astype(np.uint8)
            h5.create_group("frames").create_dataset(
                "gray", data=arr, chunks=(1, h, w),
            )
    return path


def _make_telemed_shaped_frames(n_frames=5, full_h=64, full_w=96,
                                roi=(10, 50, 5, 45),
                                margin_w=5, gray=56, seed=0):
    """Build a synthetic frame stack with Telemed-shaped UI margins.

    The panel ROI (defaults x1=10..x2=50, y1=5..y2=45 = 41x41) is
    filled with: solid gray margins (``margin_w`` cols on each side),
    a random low-contrast inner image in the middle, and a single
    saturated tick row at the panel's bottom. Outside the panel, the
    frame stays zero. This is enough for the detector to find the
    inner box.
    """
    rng = np.random.default_rng(seed)
    x1, x2, y1, y2 = roi
    arr = np.zeros((n_frames, full_h, full_w), dtype=np.uint8)
    for i in range(n_frames):
        # Fill the whole panel with gray
        arr[i, y1-1:y2, x1-1:x2] = gray
        # Inner image (excluding margins on cols + bottom tick row)
        ix1 = x1 - 1 + margin_w
        ix2 = x2 - margin_w  # exclusive
        # leave the last panel row (y2-1) for the tick band
        iy1 = y1 - 1
        iy2 = y2 - 1  # tick row at y2-1
        inner = rng.integers(8, 41, size=(iy2 - iy1, ix2 - ix1), dtype=np.uint8)
        arr[i, iy1:iy2, ix1:ix2] = inner
        # Tick row (saturated white across panel width)
        arr[i, y2 - 1, x1 - 1:x2] = 255
    return arr


# ---------- Synthetic-fixture tests ----------


def test_load_basic_attrs(tmp_path):
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "syn.tvd.h5", n_frames=5)
    lf = Log(f)
    assert lf.name == "syn"
    assert lf.n_frames == 5
    assert lf.full_frame_width == 96
    assert lf.full_frame_height == 64
    assert lf.b_mode_roi.width == 41
    assert lf.b_mode_roi.height == 41
    assert lf.schema_version == "v1"
    assert lf.n_b_images == 1
    assert lf.has_frames is True
    assert lf.duration_s > 0
    assert 60 < lf.mean_fps < 80  # synthetic data lands near 67-70 fps


def test_v1a1_legacy_int_schema_loads(tmp_path):
    """Sidecars from pre-v1a2 extracts had ``schema_version=1`` (int)
    + a single unprefixed ``roi_*`` block + no params. Log must read
    them, normalising the version to the ``"v1aN"`` string form."""
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "v1a1.tvd.h5", n_frames=3,
                           schema_version=1, params=None)
    lf = Log(f)
    assert lf.schema_version == "v1"
    assert lf.params == {}
    assert lf.n_b_images == 1
    assert lf.b_mode_rois[1].img_id == 1


def test_v1a3_legacy_roi_collapses_to_img_id_1(tmp_path):
    """v1a3 sidecars (the int-schema_version=3 production format up
    through 2026-05-24) wrote a single unprefixed ``roi_*`` block +
    flat ``physical_d{x,y}_cm_per_px``. Current Log must read these
    as ``b_mode_rois[1]`` and normalise the version label to ``"v1"``."""
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "v1a3.tvd.h5", n_frames=3,
                           schema_version=3, params={"probe_name": "L18-10"})
    lf = Log(f)
    assert lf.schema_version == "v1"
    assert lf.n_b_images == 1
    assert lf.b_mode_rois[1].x1 == 10
    assert lf.b_mode_rois[1].physical_dx_cm_per_px == 0.01
    # Back-compat aliases still work.
    assert lf.physical_dx_cm_per_px == 0.01
    assert lf.b_mode_roi.width == 41


def test_v1a5_image_d_round_trip(tmp_path):
    """v1a5+ sidecars store the display scale as root attrs; Log reads
    them from storage rather than deriving on the fly."""
    from telemed import Log

    f = _make_synthetic_h5(
        tmp_path / "v1a5.tvd.h5", n_frames=3,
        image_dx=0.00896, image_dy=0.00896,
        params={"b_depth": 50},  # would derive 0.05/41 = 0.00122 if not stored
    )
    lf = Log(f)
    # Stored value wins over the derivation fallback.
    assert lf.image_dx_cm_per_px == 0.00896
    assert lf.image_dy_cm_per_px == 0.00896


def test_image_d_falls_back_to_derivation_for_legacy(tmp_path):
    """Legacy v1a4 sidecars (no stored image_d) must still produce a
    value via the b_depth + panel_height fallback."""
    from telemed import Log

    f = _make_synthetic_h5(
        tmp_path / "v1a4.tvd.h5", n_frames=3,
        schema_version=4,           # legacy int form (= v1a4)
        params={"b_depth": 50},     # 5 cm depth, panel_height=41 px (synth)
    )
    lf = Log(f)
    assert lf.schema_version == "v1"
    # No stored value -> derive from depth/height. b_depth in mm
    # (=50 here = 5 cm); panel_height = 41 (synthetic fixture default).
    # image_dy_cm_per_px = (50 / 10) / 41 = 5.0 / 41 ~= 0.122
    expected = 5.0 / 41
    assert lf.image_dy_cm_per_px == pytest.approx(expected, rel=1e-9)
    assert lf.image_dx_cm_per_px == pytest.approx(expected, rel=1e-9)


def test_image_d_none_when_no_depth(tmp_path):
    """v1a2+ sidecars with no ``b_depth`` param + no stored image_d
    -> property returns ``None``."""
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "no_depth.tvd.h5", n_frames=3)
    lf = Log(f)
    assert lf.image_dx_cm_per_px is None
    assert lf.image_dy_cm_per_px is None


def test_frame_crop_image_runs_inline_detection(tmp_path):
    """``lf.frame(i, crop="image")`` aggregates the sidecar's frames,
    runs the inner-image detector, and returns the cropped slice.
    With Telemed-shaped synthetic frames (gray margins + tick row),
    the inner box is smaller than the outer panel."""
    from telemed import Log

    panel_roi = (10, 50, 5, 45)  # full-frame x1..x2, y1..y2
    frames = _make_telemed_shaped_frames(
        n_frames=5, full_h=64, full_w=96, roi=panel_roi, margin_w=5,
    )
    f = _make_synthetic_h5(
        tmp_path / "shaped.tvd.h5", n_frames=5, h=64, w=96,
        frames_data=frames,
    )
    lf = Log(f)
    inner = lf.frame(0, crop="image")
    panel = lf.frame(0, crop="panel")
    full = lf.frame(0, crop=False)
    # Panel is 41x41 (x2-x1+1, y2-y1+1). Inner is strictly smaller on
    # both axes thanks to the gray margins + tick row.
    assert panel.shape == (41, 41)
    assert full.shape == (64, 96)
    assert inner.shape[0] < panel.shape[0]
    assert inner.shape[1] < panel.shape[1]
    # crop=True is an alias for crop="image" -- same shape.
    assert lf.frame(0, crop=True).shape == inner.shape


def test_image_slice_caches_per_panel(tmp_path):
    """``Log.image_slice(panel)`` caches its result so subsequent
    calls don't re-aggregate the frames + re-run the detector."""
    from telemed import Log

    panel_roi = (10, 50, 5, 45)
    frames = _make_telemed_shaped_frames(
        n_frames=5, full_h=64, full_w=96, roi=panel_roi, margin_w=5,
    )
    f = _make_synthetic_h5(
        tmp_path / "cached.tvd.h5", n_frames=5, h=64, w=96,
        frames_data=frames,
    )
    lf = Log(f)
    s1 = lf.image_slice(1)
    s2 = lf.image_slice(1)
    assert s1 == s2
    # Result is cached on the instance (read it directly).
    assert 1 in lf._image_slice_cache


def test_image_slice_falls_back_to_panel_on_flat_frames(tmp_path):
    """Default synthetic fixture (gradient, no UI margins) -> detector
    returns None -> Log.image_slice warns + returns the panel slice."""
    import pytest as _pt
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "flat.tvd.h5", n_frames=3)
    lf = Log(f)
    with _pt.warns(UserWarning, match="couldn't identify inner ultrasound image"):
        ys, xs = lf.image_slice(1)
    panel_ys, panel_xs = lf.b_mode_roi.as_slice()
    assert (ys, xs) == (panel_ys, panel_xs)


def test_frame_crop_invalid_raises(tmp_path):
    import pytest as _pt
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "syn.tvd.h5", n_frames=3)
    lf = Log(f)
    with _pt.raises(ValueError, match="crop="):
        lf.frame(0, crop="bogus")


def test_v4_dual_probe_rois(tmp_path):
    """v4 sidecars from dual-probe recordings carry two ROI blocks
    (img_id=1 and 2) with potentially different physical resolutions."""
    from telemed import Log

    rois = {
        1: dict(x1=73, x2=425, y1=43, y2=600, width=353, height=558,
                dx=0.012, dy=0.013),
        2: dict(x1=429, x2=777, y1=43, y2=600, width=349, height=558,
                dx=0.011, dy=0.014),
    }
    f = _make_synthetic_h5(
        tmp_path / "dual.tvd.h5", n_frames=3, h=601, w=1554,
        rois=rois,
    )
    lf = Log(f)
    assert lf.n_b_images == 2
    assert set(lf.b_mode_rois) == {1, 2}
    assert lf.b_mode_rois[1].physical_dx_cm_per_px == 0.012
    assert lf.b_mode_rois[2].physical_dx_cm_per_px == 0.011
    # Back-compat alias still points at img_id=1.
    assert lf.b_mode_roi.img_id == 1


def test_v2_params_round_trip(tmp_path):
    """``param_*`` HDF5 attrs surface on ``Log.params`` with the prefix
    stripped and HDF5-native types coerced to plain Python."""
    from telemed import Log

    f = _make_synthetic_h5(
        tmp_path / "v2.tvd.h5", n_frames=3,
        params={
            "probe_name": "L18-10",
            "probe_code": 4209,
            "beamformer_name": "ArtUS",
            "cine_end_datetime_str": "2025.10.06 15:50:44.234567",
            "b_depth": 60,
            "b_frequency": 12,
            "b_gain": 50,
            "b_thi": True,
            "b_image_enhancement": False,
        },
    )
    lf = Log(f)
    p = lf.params
    assert p["probe_name"] == "L18-10"
    assert isinstance(p["probe_code"], int) and p["probe_code"] == 4209
    assert p["cine_end_datetime_str"].startswith("2025.10.06")
    assert p["b_depth"] == 60
    assert p["b_thi"] is True or p["b_thi"] == np.True_  # h5py bool variants
    assert isinstance(p["b_image_enhancement"], (bool, np.bool_))


def test_v2_partial_params_silently_ok(tmp_path):
    """Failed ParamGet probes don't crash the load -- they're just
    absent from ``Log.params``. ``.get(name)`` returns None."""
    from telemed import Log

    f = _make_synthetic_h5(
        tmp_path / "partial.tvd.h5", n_frames=3,
        params={"probe_name": "L18-10"},
    )
    lf = Log(f)
    assert lf.params.get("probe_name") == "L18-10"
    assert lf.params.get("b_depth") is None
    assert "b_depth" not in lf.params


def test_timing_arrays_shape_and_anchor(tmp_path):
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "syn.tvd.h5", n_frames=5)
    lf = Log(f)
    assert lf.time_ms.shape == (5,)
    assert lf.ifi_ms.shape == (5,)
    # Frame 1 is the anchor: time=0, ifi=0.
    assert lf.time_ms[0] == 0.0
    assert lf.ifi_ms[0] == 0.0


def test_frame_full_vs_crop(tmp_path):
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "syn.tvd.h5", n_frames=3, h=64, w=96)
    lf = Log(f)
    full = lf.frame(0)
    cropped = lf.frame(0, crop=True)
    assert full.shape == (64, 96)
    assert full.dtype == np.uint8
    assert cropped.shape == (41, 41)  # roi_height x roi_width
    # crop should be a slice of the full frame
    np.testing.assert_array_equal(
        cropped,
        full[lf.b_mode_roi.y1 - 1:lf.b_mode_roi.y2,
             lf.b_mode_roi.x1 - 1:lf.b_mode_roi.x2],
    )


def test_frame_out_of_range_raises(tmp_path):
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "syn.tvd.h5", n_frames=3)
    lf = Log(f)
    with pytest.raises(IndexError):
        lf.frame(5)


def test_no_frames_raises_useful_message(tmp_path):
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "syn.tvd.h5", n_frames=3, include_frames=False)
    lf = Log(f)
    assert lf.has_frames is False
    with pytest.raises(RuntimeError, match="no frame data"):
        lf.frame(0)
    with pytest.raises(RuntimeError, match="no frame data"):
        lf.view()


def test_missing_file_raises():
    from telemed import Log

    with pytest.raises(FileNotFoundError):
        Log("C:/does/not/exist.tvd.h5")


def test_view_returns_figure_with_widgets(tmp_path):
    """``view()`` should construct a Figure with the slider attached.

    Runs under MPLBACKEND=Agg (set at module top); we don't actually
    drive any interaction -- just verify the wiring is intact.
    """
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "syn.tvd.h5", n_frames=5)
    lf = Log(f)
    fig = lf.view()
    try:
        assert hasattr(fig, "_telemed_view_slider")
        assert fig._telemed_view_slider.valmax == 4  # n_frames - 1
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_repr_includes_name_and_shape(tmp_path):
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "syn.tvd.h5", n_frames=3)
    r = repr(Log(f))
    assert "telemed.Log" in r
    assert "syn" in r
    assert "n_frames=3" in r


# ---------- Per-panel frame access ----------


def _dual_probe_h5(tmp_path: Path) -> Path:
    """Two side-by-side B-mode panels at distinct ROIs + dx/dy."""
    rois = {
        1: dict(x1=11, x2=50, y1=6, y2=45, width=40, height=40,
                dx=0.012, dy=0.013),
        2: dict(x1=61, x2=100, y1=6, y2=45, width=40, height=40,
                dx=0.011, dy=0.014),
    }
    return _make_synthetic_h5(
        tmp_path / "dual.tvd.h5", n_frames=3, h=64, w=110, rois=rois,
    )


def test_frame_panel_dual_probe_crop(tmp_path):
    """Each panel's crop is a slice of the shared full frame at the
    panel's own ROI."""
    from telemed import Log

    lf = Log(_dual_probe_h5(tmp_path))
    full = lf.frame(0)
    c1 = lf.frame(0, crop=True, panel=1)
    c2 = lf.frame(0, crop=True, panel=2)
    assert c1.shape == (40, 40)
    assert c2.shape == (40, 40)
    np.testing.assert_array_equal(
        c1, full[lf.b_mode_rois[1].y1 - 1:lf.b_mode_rois[1].y2,
                 lf.b_mode_rois[1].x1 - 1:lf.b_mode_rois[1].x2],
    )
    np.testing.assert_array_equal(
        c2, full[lf.b_mode_rois[2].y1 - 1:lf.b_mode_rois[2].y2,
                 lf.b_mode_rois[2].x1 - 1:lf.b_mode_rois[2].x2],
    )
    # The two panels look at different columns of the same source,
    # so they must not be identical.
    assert not np.array_equal(c1, c2)


def test_frame_default_panel_unchanged(tmp_path):
    """Existing call sites without ``panel=`` keep getting img_id=1."""
    from telemed import Log

    lf = Log(_dual_probe_h5(tmp_path))
    c_default = lf.frame(0, crop=True)
    c_explicit = lf.frame(0, crop=True, panel=1)
    np.testing.assert_array_equal(c_default, c_explicit)


def test_frame_panel_validated_even_when_not_cropping(tmp_path):
    from telemed import Log

    lf = Log(_dual_probe_h5(tmp_path))
    with pytest.raises(KeyError, match="panel=99"):
        lf.frame(0, crop=False, panel=99)


# ---------- mp4_path planning ----------


def test_mp4_path_single_probe(tmp_path):
    """Single-probe recordings use the bare ``<stem>.mp4`` form."""
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "scan.tvd.h5", n_frames=3)
    lf = Log(f)
    assert lf.n_b_images == 1
    assert lf.mp4_path() == tmp_path / "scan.mp4"
    assert lf.mp4_path(panel=1) == tmp_path / "scan.mp4"


def test_mp4_path_multi_probe(tmp_path):
    """Multi-probe recordings get one ``<stem>_b{N}.mp4`` per active panel."""
    from telemed import Log

    lf = Log(_dual_probe_h5(tmp_path))
    assert lf.mp4_path(panel=1) == tmp_path / "dual_b1.mp4"
    assert lf.mp4_path(panel=2) == tmp_path / "dual_b2.mp4"


def test_mp4_path_honors_out_dir(tmp_path):
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "scan.tvd.h5", n_frames=3)
    lf = Log(f)
    elsewhere = tmp_path / "outputs"
    assert lf.mp4_path(out_dir=elsewhere) == elsewhere / "scan.mp4"


def test_mp4_path_rejects_inactive_panel(tmp_path):
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "scan.tvd.h5", n_frames=3)
    lf = Log(f)
    with pytest.raises(KeyError, match="panel=2"):
        lf.mp4_path(panel=2)


def test_mp4_path_strips_only_composite_suffix(tmp_path):
    """Sidecars not named ``*.tvd.h5`` fall back to ``Path.stem``."""
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "scan.h5", n_frames=3)
    lf = Log(f)
    assert lf.mp4_path() == tmp_path / "scan.mp4"


# ---------- ensure_mp4 ----------


def test_ensure_mp4_skips_encode_when_mp4_exists(tmp_path, monkeypatch):
    """If the target mp4 is already on disk, ensure_mp4 must NOT call
    out to the encoder."""
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "scan.tvd.h5", n_frames=3)
    lf = Log(f)
    expected = lf.mp4_path()
    expected.write_bytes(b"")  # placeholder; existence is all ensure checks

    calls: list = []
    monkeypatch.setattr(
        lf, "to_video",
        lambda *a, **kw: calls.append((a, kw)) or {},
    )
    got = lf.ensure_mp4()
    assert got == expected
    assert calls == []


def test_ensure_mp4_invokes_encode_when_missing(tmp_path, monkeypatch):
    """Encode side-effect simulated; verifies (a) to_video is called
    with the same ``out_dir`` ensure_mp4 used, (b) encode_kwargs are
    forwarded, and (c) the returned path is the planned mp4."""
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "scan.tvd.h5", n_frames=3)
    lf = Log(f)
    expected = lf.mp4_path()

    calls: list = []

    def _fake_to_video(*a, **kw):
        calls.append((a, kw))
        # Simulate the encode landing the file at the planned path.
        Path(expected).write_bytes(b"")
        return {str(expected): "built"}

    monkeypatch.setattr(lf, "to_video", _fake_to_video)
    got = lf.ensure_mp4(lossless=False, crf=22)
    assert got == expected
    assert got.exists()
    assert len(calls) == 1
    _, kw = calls[0]
    assert kw["out_dir"] is None
    assert kw["lossless"] is False
    assert kw["crf"] == 22


def test_ensure_mp4_multi_probe_one_call_builds_all_panels(tmp_path, monkeypatch):
    """Asking for one panel triggers the shared encode pass; the
    sibling panel's mp4 lands as a side effect (export_video iterates
    every panel of the recording)."""
    from telemed import Log

    lf = Log(_dual_probe_h5(tmp_path))
    p1 = lf.mp4_path(panel=1)
    p2 = lf.mp4_path(panel=2)

    calls: list = []

    def _fake_to_video(*a, **kw):
        calls.append((a, kw))
        Path(p1).write_bytes(b"")
        Path(p2).write_bytes(b"")
        return {str(p1): "built", str(p2): "built"}

    monkeypatch.setattr(lf, "to_video", _fake_to_video)
    got = lf.ensure_mp4(panel=2)
    assert got == p2
    assert p1.exists() and p2.exists()
    assert len(calls) == 1

    # Second call for the other panel is now a no-op (file exists).
    got2 = lf.ensure_mp4(panel=1)
    assert got2 == p1
    assert len(calls) == 1


def test_ensure_mp4_honors_out_dir(tmp_path, monkeypatch):
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "scan.tvd.h5", n_frames=3)
    lf = Log(f)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    expected = lf.mp4_path(out_dir=elsewhere)

    captured: dict = {}

    def _fake_to_video(*a, **kw):
        captured.update(kw)
        Path(expected).write_bytes(b"")
        return {str(expected): "built"}

    monkeypatch.setattr(lf, "to_video", _fake_to_video)
    got = lf.ensure_mp4(out_dir=elsewhere)
    assert got == expected
    # ensure_mp4 must forward the same out_dir to to_video so the
    # existence check and the encode agree on the target folder.
    assert captured["out_dir"] == elsewhere


def test_ensure_mp4_rejects_inactive_panel(tmp_path):
    from telemed import Log

    f = _make_synthetic_h5(tmp_path / "scan.tvd.h5", n_frames=3)
    lf = Log(f)
    with pytest.raises(KeyError, match="panel=2"):
        lf.ensure_mp4(panel=2)


# ---------- Real-fixture test (skipped if absent) ----------


REAL_FIXTURE = Path("C:/data/temp2/1_20251006_155042.tvd.h5")


@pytest.mark.skipif(
    not REAL_FIXTURE.is_file(),
    reason=f"Real Telemed fixture {REAL_FIXTURE} not present",
)
def test_real_fixture_consistent_with_export():
    """Pin the schema + numbers Praneeth saw on the 2026-05-23 export.

    149 frames; B-mode ROI 705x558 at (73..777, 43..600); the known
    recording-end IFI outlier (~0.078 ms) is in the last position.
    """
    from telemed import Log

    lf = Log(REAL_FIXTURE)
    assert lf.n_frames == 149
    assert lf.full_frame_width == 1554
    assert lf.full_frame_height == 601
    assert lf.b_mode_roi.width == 705
    assert lf.b_mode_roi.height == 558
    assert lf.b_mode_roi.x1 == 73 and lf.b_mode_roi.x2 == 777
    # The known last-frame outlier (recording-end artefact).
    assert lf.ifi_ms[-1] < 1.0
