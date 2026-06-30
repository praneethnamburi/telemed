"""Shared pytest fixtures / warnings filters for the telemed test suite."""

# Force the non-interactive Agg backend for the whole suite, before
# matplotlib is imported anywhere. ``Log.view()`` builds a Figure via
# ``plt.subplots`` -- if the backend is left to auto-resolve it picks the
# default *GUI* backend (e.g. TkAgg), which then needs a working display /
# Tcl install to even create the window. On headless or mis-provisioned CI
# runners (the hosted Python 3.10 image ships a broken Tcl) that raises
# ``TclError`` at figure creation. conftest.py is imported before any test
# module, so doing this here (rather than a per-module
# ``os.environ.setdefault`` that loses the race if MPLBACKEND is already
# set or matplotlib is imported first) makes the choice deterministic.
import os

os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg", force=True)

import warnings

import pytest


@pytest.fixture(autouse=True)
def _silence_crop_deprecation():
    """Silence ``telemed.crop_video`` / ``telemed.crop_folder`` deprecation
    noise within tests that exercise the legacy crop pipeline.

    The crop module emits a ``DeprecationWarning`` on every call (it's
    scheduled for removal in v0.2.0). The warning is correct user-facing
    behaviour, but during the test suite we know the module is being
    invoked deliberately.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"telemed\.crop_(video|folder) is deprecated",
            category=DeprecationWarning,
        )
        yield
