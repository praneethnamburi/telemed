"""Tests for ``immersionlab.telemed.Log``.

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
                       schema_version: int | str = "v1a5",
                       params: dict | None = None,
                       rois: dict[int, dict] | None = None,
                       image_dx: float | None = None,
                       image_dy: float | None = None) -> Path:
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
    """
    if rois is None:
        rois = {1: dict(x1=10, x2=50, y1=5, y2=45, width=41, height=41,
                        dx=0.01, dy=0.01)}
    # Normalise legacy int versions for the per-img_id-block predicate.
    is_per_img_id = isinstance(schema_version, str) or (
        isinstance(schema_version, int) and schema_version >= 4
    )
    is_v1a5_plus = isinstance(schema_version, str) and schema_version >= "v1a5"
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
        if is_v1a5_plus and image_dx is not None:
            h5.attrs["image_dx_cm_per_px"] = image_dx
        if is_v1a5_plus and image_dy is not None:
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
            # gradient-fill frames so each one is distinguishable
            arr = np.zeros((n_frames, h, w), dtype=np.uint8)
            for i in range(n_frames):
                arr[i] = (np.linspace(0, 255, w, dtype=np.float32)[None, :]
                          .repeat(h, axis=0) * (1.0 - i / max(n_frames - 1, 1))).astype(np.uint8)
            h5.create_group("frames").create_dataset(
                "gray", data=arr, chunks=(1, h, w),
            )
    return path


# ---------- Synthetic-fixture tests ----------


def test_load_basic_attrs(tmp_path):
    from immersionlab.telemed import Log

    f = _make_synthetic_h5(tmp_path / "syn.tvd.h5", n_frames=5)
    lf = Log(f)
    assert lf.name == "syn"
    assert lf.n_frames == 5
    assert lf.full_frame_width == 96
    assert lf.full_frame_height == 64
    assert lf.b_mode_roi.width == 41
    assert lf.b_mode_roi.height == 41
    assert lf.schema_version == "v1a5"
    assert lf.n_b_images == 1
    assert lf.has_frames is True
    assert lf.duration_s > 0
    assert 60 < lf.mean_fps < 80  # synthetic data lands near 67-70 fps


def test_v1a1_legacy_int_schema_loads(tmp_path):
    """Sidecars from pre-v1a2 extracts had ``schema_version=1`` (int)
    + a single unprefixed ``roi_*`` block + no params. Log must read
    them, normalising the version to the ``"v1aN"`` string form."""
    from immersionlab.telemed import Log

    f = _make_synthetic_h5(tmp_path / "v1a1.tvd.h5", n_frames=3,
                           schema_version=1, params=None)
    lf = Log(f)
    assert lf.schema_version == "v1a1"
    assert lf.params == {}
    assert lf.n_b_images == 1
    assert lf.b_mode_rois[1].img_id == 1


def test_v1a3_legacy_roi_collapses_to_img_id_1(tmp_path):
    """v1a3 sidecars (the int-schema_version=3 production format up
    through 2026-05-24) wrote a single unprefixed ``roi_*`` block +
    flat ``physical_d{x,y}_cm_per_px``. Current Log must read these
    as ``b_mode_rois[1]`` and normalise the version to ``"v1a3"``."""
    from immersionlab.telemed import Log

    f = _make_synthetic_h5(tmp_path / "v1a3.tvd.h5", n_frames=3,
                           schema_version=3, params={"probe_name": "L18-10"})
    lf = Log(f)
    assert lf.schema_version == "v1a3"
    assert lf.n_b_images == 1
    assert lf.b_mode_rois[1].x1 == 10
    assert lf.b_mode_rois[1].physical_dx_cm_per_px == 0.01
    # Back-compat aliases still work.
    assert lf.physical_dx_cm_per_px == 0.01
    assert lf.b_mode_roi.width == 41


def test_v1a5_image_d_round_trip(tmp_path):
    """v1a5+ sidecars store the display scale as root attrs; Log reads
    them from storage rather than deriving on the fly."""
    from immersionlab.telemed import Log

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
    from immersionlab.telemed import Log

    f = _make_synthetic_h5(
        tmp_path / "v1a4.tvd.h5", n_frames=3,
        schema_version=4,           # legacy int form (= v1a4)
        params={"b_depth": 50},     # 5 cm depth, panel_height=41 px (synth)
    )
    lf = Log(f)
    assert lf.schema_version == "v1a4"
    # No stored value -> derive from depth/height. b_depth in mm
    # (=50 here = 5 cm); panel_height = 41 (synthetic fixture default).
    # image_dy_cm_per_px = (50 / 10) / 41 = 5.0 / 41 ~= 0.122
    expected = 5.0 / 41
    assert lf.image_dy_cm_per_px == pytest.approx(expected, rel=1e-9)
    assert lf.image_dx_cm_per_px == pytest.approx(expected, rel=1e-9)


def test_image_d_none_when_no_depth(tmp_path):
    """v1a2+ sidecars with no ``b_depth`` param + no stored image_d
    -> property returns ``None``."""
    from immersionlab.telemed import Log

    f = _make_synthetic_h5(tmp_path / "no_depth.tvd.h5", n_frames=3)
    lf = Log(f)
    assert lf.image_dx_cm_per_px is None
    assert lf.image_dy_cm_per_px is None


def test_v4_dual_probe_rois(tmp_path):
    """v4 sidecars from dual-probe recordings carry two ROI blocks
    (img_id=1 and 2) with potentially different physical resolutions."""
    from immersionlab.telemed import Log

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
    from immersionlab.telemed import Log

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
    from immersionlab.telemed import Log

    f = _make_synthetic_h5(
        tmp_path / "partial.tvd.h5", n_frames=3,
        params={"probe_name": "L18-10"},
    )
    lf = Log(f)
    assert lf.params.get("probe_name") == "L18-10"
    assert lf.params.get("b_depth") is None
    assert "b_depth" not in lf.params


def test_timing_arrays_shape_and_anchor(tmp_path):
    from immersionlab.telemed import Log

    f = _make_synthetic_h5(tmp_path / "syn.tvd.h5", n_frames=5)
    lf = Log(f)
    assert lf.time_ms.shape == (5,)
    assert lf.ifi_ms.shape == (5,)
    # Frame 1 is the anchor: time=0, ifi=0.
    assert lf.time_ms[0] == 0.0
    assert lf.ifi_ms[0] == 0.0


def test_frame_full_vs_crop(tmp_path):
    from immersionlab.telemed import Log

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
    from immersionlab.telemed import Log

    f = _make_synthetic_h5(tmp_path / "syn.tvd.h5", n_frames=3)
    lf = Log(f)
    with pytest.raises(IndexError):
        lf.frame(5)


def test_no_frames_raises_useful_message(tmp_path):
    from immersionlab.telemed import Log

    f = _make_synthetic_h5(tmp_path / "syn.tvd.h5", n_frames=3, include_frames=False)
    lf = Log(f)
    assert lf.has_frames is False
    with pytest.raises(RuntimeError, match="no frame data"):
        lf.frame(0)
    with pytest.raises(RuntimeError, match="no frame data"):
        lf.view()


def test_missing_file_raises():
    from immersionlab.telemed import Log

    with pytest.raises(FileNotFoundError):
        Log("C:/does/not/exist.tvd.h5")


def test_view_returns_figure_with_widgets(tmp_path):
    """``view()`` should construct a Figure with the slider attached.

    Runs under MPLBACKEND=Agg (set at module top); we don't actually
    drive any interaction -- just verify the wiring is intact.
    """
    from immersionlab.telemed import Log

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
    from immersionlab.telemed import Log

    f = _make_synthetic_h5(tmp_path / "syn.tvd.h5", n_frames=3)
    r = repr(Log(f))
    assert "telemed.Log" in r
    assert "syn" in r
    assert "n_frames=3" in r


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
    from immersionlab.telemed import Log

    lf = Log(REAL_FIXTURE)
    assert lf.n_frames == 149
    assert lf.full_frame_width == 1554
    assert lf.full_frame_height == 601
    assert lf.b_mode_roi.width == 705
    assert lf.b_mode_roi.height == 558
    assert lf.b_mode_roi.x1 == 73 and lf.b_mode_roi.x2 == 777
    # The known last-frame outlier (recording-end artefact).
    assert lf.ifi_ms[-1] < 1.0
