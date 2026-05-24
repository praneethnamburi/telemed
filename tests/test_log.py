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
                       include_frames: bool = True) -> Path:
    """Write a minimal Telemed-shape HDF5 sidecar with synthetic data.

    Mirrors the schema written by
    :func:`immersionlab.telemed.export.extract_recording` (v1).
    """
    times = np.cumsum([0.0] + [14.5 + 0.5 * (i % 3) for i in range(n_frames - 1)])
    ifi = np.zeros(n_frames)
    ifi[1:] = np.diff(times)
    with h5py.File(path, "w") as h5:
        h5.attrs["n_frames"] = n_frames
        h5.attrs["full_frame_width"] = w
        h5.attrs["full_frame_height"] = h
        h5.attrs["roi_x1"] = 10
        h5.attrs["roi_x2"] = 50
        h5.attrs["roi_y1"] = 5
        h5.attrs["roi_y2"] = 45
        h5.attrs["roi_width"] = 41
        h5.attrs["roi_height"] = 41
        h5.attrs["physical_dx_cm_per_px"] = 0.01
        h5.attrs["physical_dy_cm_per_px"] = 0.01
        h5.attrs["source_tvd_path"] = "C:/synthetic/test.tvd"
        h5.attrs["extracted_at_iso"] = "2026-05-23T00:00:00"
        h5.attrs["schema_version"] = 1
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
    assert lf.schema_version == 1
    assert lf.has_frames is True
    assert lf.duration_s > 0
    assert 60 < lf.mean_fps < 80  # synthetic data lands near 67-70 fps


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
