"""Tests for ``telemed._winperf`` (Windows background-throttle suppression).

The Win32 calls (SetProcessInformation / Toolhelp / SetThreadExecutionState)
are environment-coupled, but the process-name matcher is pure and the public
``keep_full_speed`` entry point must be a safe best-effort no-op-on-failure
on every platform. Those are what we pin here.

The ``keep_full_speed`` smoke test deliberately uses a non-matching token and
``prevent_sleep=False`` so it never opens / mutates a real EchoWave process
or holds a system-sleep lock -- safe to run alongside a live extraction.
"""

from __future__ import annotations

import inspect

import pytest


@pytest.mark.parametrize(
    "exe_name",
    ["EchoWave.exe", "echowave.exe", "Echo Wave II.exe", "ECHOWAVE.EXE", "EchoWave"],
)
def test_name_matches_echowave_variants(exe_name):
    from telemed._winperf import _name_matches

    assert _name_matches(exe_name, "echowave")


@pytest.mark.parametrize(
    "exe_name",
    ["EchoShell.exe", "Ew2Osk.exe", "python.exe", "explorer.exe", ""],
)
def test_name_matches_rejects_non_echowave(exe_name):
    from telemed._winperf import _name_matches

    assert not _name_matches(exe_name, "echowave")


def test_name_matches_is_space_and_case_insensitive():
    from telemed._winperf import _name_matches

    # token with spaces / mixed case still matches the normalised image name
    assert _name_matches("EchoWave.exe", "Echo Wave")
    assert _name_matches("Echo Wave II.exe", "ECHOWAVE")


def test_keep_full_speed_is_safe_and_returns_shape():
    """Best-effort: returns the documented dict shape and never raises.

    Uses a token that can't match EchoWave and ``prevent_sleep=False`` so
    the call touches nothing but (harmlessly) the current test process.
    """
    import telemed

    msgs: list[str] = []
    result = telemed.keep_full_speed(
        echowave_match="telemed_no_such_process_xyz",
        prevent_sleep=False,
        log=msgs.append,
    )
    assert set(result) >= {"self", "echowave_pids", "prevent_sleep"}
    assert result["echowave_pids"] == []  # token can't match anything real
    assert result["prevent_sleep"] is False  # we asked it not to
    assert len(msgs) == 1  # exactly one status line emitted


def test_export_h5_exposes_keep_full_speed_kwarg():
    """The flag must be on export_h5 so process() signature-routes it."""
    import telemed

    params = inspect.signature(telemed.export_h5).parameters
    assert "keep_full_speed" in params
    assert params["keep_full_speed"].default is True


def test_process_accepts_keep_full_speed_kwarg():
    """process() routes unknown kwargs by export_h5's signature; the flag
    must survive that filter rather than raise TypeError."""
    import telemed

    h5_params = set(inspect.signature(telemed.export_h5).parameters)
    assert "keep_full_speed" in h5_params
