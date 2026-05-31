"""Suppress Windows background-throttling of the COM extract loop.

The ``.tvd`` extract drives EchoWave one frame at a time over synchronous
cross-process COM round-trips (see :func:`telemed._extract._extract_one`).
Throughput (~5 fps with pixels) holds only while the session has an
*attended, foreground, connected* desktop. The moment the driving console
loses foreground focus -- **or the whole RDP session is disconnected** --
Windows starts treating both processes as background work and the rate
collapses to ~1-2 fps after a short grace period.

That grace period is why an early ~1-minute "disconnected runs full speed"
measurement looked fine while a longer disconnected run does not: the
throttle hadn't kicked in yet.

Two background-deprioritization levers are in play, and both are
opt-out-able **without touching EchoWave's (undocumented) COM threading
model**:

1. **EcoQoS / power throttling** (Win10 1709+). Windows parks background
   processes on efficiency cores and lowers their clock. We opt *both* the
   Python client and the ``EchoWave.exe`` server out via
   ``SetProcessInformation(ProcessPowerThrottling, StateMask=0)``. Immediate,
   no reboot.

2. **System sleep** during long unattended runs -- guarded with
   ``SetThreadExecutionState`` so a multi-hour batch on a disconnected
   session doesn't stall on the box going to sleep.

A *third* lever -- system timer resolution -- can't be fixed at runtime
from a background process (post-2004 ``timeBeginPeriod`` is per-process and
ignored while backgrounded). The escape hatch is a reboot-time registry
flag, documented in :func:`keep_full_speed` rather than applied here.

Everything is best-effort: any failure (missing privilege, EchoWave not
running, older Windows) is swallowed so the extract still runs -- just
possibly throttled. Non-Windows is a clean no-op.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Callable, Optional

_IS_WINDOWS = sys.platform == "win32"

# ---- Win32 constants ----

# PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
_PROCESS_POWER_THROTTLING = 4
_PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
_PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
_PROCESS_SET_INFORMATION = 0x0200

# SetThreadExecutionState flags
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001

# CreateToolhelp32Snapshot
_TH32CS_SNAPPROCESS = 0x00000002


class _PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
    _fields_ = [
        ("Version", wintypes.DWORD),
        ("ControlMask", wintypes.DWORD),
        ("StateMask", wintypes.DWORD),
    ]


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


def _kernel32():
    """kernel32 with explicit arg/return types.

    Critical on 64-bit: HANDLEs are pointer-sized, so leaving ``restype``
    at the default ``c_int`` truncates them. Configure once at import.
    """
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.GetCurrentProcess.argtypes = []
    k.OpenProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.CloseHandle.restype = wintypes.BOOL
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.SetProcessInformation.restype = wintypes.BOOL
    k.SetProcessInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    k.SetThreadExecutionState.restype = wintypes.DWORD
    k.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k.Process32FirstW.restype = wintypes.BOOL
    k.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    k.Process32NextW.restype = wintypes.BOOL
    k.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    return k


_K32 = _kernel32() if _IS_WINDOWS else None
_INVALID_HANDLE = ctypes.c_void_p(-1).value


# ---- pure helper (unit-testable without Win32) ----


def _name_matches(exe_name: str, match: str) -> bool:
    """True if process image name ``exe_name`` matches the ``match`` token.

    Normalises by lower-casing, stripping spaces, and dropping a trailing
    ``.exe`` so ``"EchoWave.exe"`` and ``"Echo Wave II.exe"`` both match the
    default ``"echowave"`` token, while siblings like ``"EchoShell.exe"`` do
    not.
    """
    norm = exe_name.lower().replace(" ", "")
    if norm.endswith(".exe"):
        norm = norm[:-4]
    return match.lower().replace(" ", "") in norm


# ---- Win32-backed primitives (best-effort) ----


def _disable_power_throttling_handle(handle) -> bool:
    """Opt the process behind ``handle`` out of EcoQoS execution-speed
    throttling. ``ControlMask=EXECUTION_SPEED, StateMask=0`` means
    "I control this, and it's OFF" -> always full speed."""
    state = _PROCESS_POWER_THROTTLING_STATE(
        Version=_PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask=_PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        StateMask=0,
    )
    return bool(
        _K32.SetProcessInformation(
            handle,
            _PROCESS_POWER_THROTTLING,
            ctypes.byref(state),
            ctypes.sizeof(state),
        )
    )


def _disable_power_throttling_pid(pid: int) -> bool:
    """Open ``pid`` for SET_INFORMATION and opt it out of EcoQoS."""
    h = _K32.OpenProcess(_PROCESS_SET_INFORMATION, False, pid)
    if not h:
        return False
    try:
        return _disable_power_throttling_handle(h)
    finally:
        _K32.CloseHandle(h)


def _find_pids(match: str) -> list[int]:
    """PIDs of running processes whose image name matches ``match``."""
    snap = _K32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if not snap or snap == _INVALID_HANDLE:
        return []
    pids: list[int] = []
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
        ok = _K32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            if _name_matches(entry.szExeFile, match):
                pids.append(int(entry.th32ProcessID))
            ok = _K32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        _K32.CloseHandle(snap)
    return pids


# ---- public entry point ----


def keep_full_speed(
    *,
    echowave_match: str = "echowave",
    prevent_sleep: bool = True,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    """Stop Windows from background-throttling the .tvd extract loop.

    Opts the **current Python process** and every running **EchoWave**
    process out of EcoQoS execution-speed throttling, so the COM extract
    keeps its full ~5 fps even when the driving console is backgrounded or
    the RDP session is disconnected. Optionally inhibits system sleep for
    the duration of the calling thread (long unattended batches).

    Call it once before a manual extraction loop; :func:`telemed.export_h5`
    (and therefore :func:`telemed.process`) calls it automatically unless
    you pass ``keep_full_speed=False``.

    Args:
        echowave_match: Substring (space/case-insensitive, ``.exe``
            stripped) matched against running process image names. Default
            ``"echowave"`` matches ``EchoWave.exe``. Override only if your
            EchoWave build ships under a different name.
        prevent_sleep: If True (default), hold ``ES_SYSTEM_REQUIRED`` so a
            disconnected box doesn't sleep mid-batch. The hold lasts as long
            as the calling thread lives.
        log: Optional ``fn(msg)`` for a one-line status report.

    Returns:
        ``{"self": bool, "echowave_pids": [int, ...], "prevent_sleep":
        bool}`` -- what actually took. ``echowave_pids == []`` means
        EchoWave wasn't running yet (start it first) or is named something
        other than ``echowave_match``. Non-Windows returns the same dict
        shape with everything falsey plus ``"platform": "non-windows"``.

    Note:
        A third throttle -- **system timer resolution** -- can't be fixed
        from here: on Win10 2004+, ``timeBeginPeriod`` is per-process and
        ignored while a process is backgrounded, so the foreground trick
        the earlier debugging tried could never have propagated to
        ``EchoWave.exe``. If EcoQoS opt-out alone doesn't fully recover the
        disconnected rate, restore global timer behaviour with a reboot::

            # HKLM\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\kernel
            #   GlobalTimerResolutionRequests = 1   (DWORD), then reboot

        which makes Windows honour any process's 1 ms request system-wide.
    """
    if not _IS_WINDOWS:
        return {
            "platform": "non-windows",
            "self": False,
            "echowave_pids": [],
            "prevent_sleep": False,
        }

    result: dict = {"self": False, "echowave_pids": [], "prevent_sleep": False}

    try:
        result["self"] = _disable_power_throttling_handle(_K32.GetCurrentProcess())
    except Exception:  # noqa: BLE001 -- best-effort guard, never fatal
        pass

    try:
        for pid in _find_pids(echowave_match):
            if _disable_power_throttling_pid(pid):
                result["echowave_pids"].append(pid)
    except Exception:  # noqa: BLE001
        pass

    if prevent_sleep:
        try:
            rc = _K32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
            result["prevent_sleep"] = rc != 0
        except Exception:  # noqa: BLE001
            pass

    if log is not None:
        ew = result["echowave_pids"]
        ew_str = (
            f"EchoWave pid(s) {ew}"
            if ew
            else f"EchoWave NOT found (start it first, or it isn't named "
            f"'{echowave_match}*.exe')"
        )
        self_str = "Python" if result["self"] else "Python(failed)"
        sleep_str = "; sleep inhibited" if result["prevent_sleep"] else ""
        try:
            log(
                f"keep-full-speed: opted {self_str} + {ew_str} out of "
                f"background power throttling{sleep_str}"
            )
        except Exception:  # noqa: BLE001
            pass

    return result
