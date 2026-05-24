"""Shared pytest fixtures / warnings filters for the telemed test suite."""

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
