"""One-shot discovery probe for AutoInt1 file-level metadata.

Run this once on a representative saved ``.tvd`` to find every
ParamGet-able / direct-getter datum the COM interface will surrender
for a file-mode (probe-detached) recording. Output is a dict plus an
optional JSON + markdown report.

Intended audience: a human curating which fields go into the production
:data:`_PARAM_SPECS` and the HDF5 sidecar schema. Not part of the
batch-export hot path; not exported from the package ``__init__``.

Safety:

* ``AutoInt1Client.txt`` warns (lines 261-263) that calling the *wrong*
  ParamGet variant for an id can crash EchoWave, not merely raise an
  exception. To stay off the crash-prone paths this probe only sweeps
  ids whose description text in the doc names a specific ``ParamGet*``
  variant, plus ``_shift``-suffixed ids (handled by ParamGetInt per the
  production convention proven in :data:`_PARAM_SPECS`). Action-only
  commands (``val = 0;``) and untagged ids are recorded in the report
  but not probed -- visible for human review without touching COM.
* The probe opens ``tvd_path`` in the currently-running EchoWave II
  instance, which closes whatever file EchoWave is currently showing.
  Do not run while another export is using the same EchoWave.

Example::

    from telemed import _metadata_probe as mp

    result = mp.probe("C:/data/temp2/one_recording.tvd")
    md = mp.write_report(result, "C:/scratch/telemed_probe.json")
    print(f"Wrote {md}")
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

# Default doc location on the lab Windows machines (override per call
# if the SDK was installed somewhere non-standard).
_DOC_PATH = Path(
    "C:/Program Files/Telemed/Echo Wave II Application/"
    "EchoWave II/Config/Plugins/AutoInt1Client.txt"
)

_DEFINE_RE = re.compile(r"^#define\s+(id_\w+)\s+(\d+)")
_PARAMGET_HINT_RE = re.compile(r"ParamGet(Int|Bool|Double|Float|String)\b")
_ACTION_RE = re.compile(r"\bval\s*=\s*0\s*;")

# img_ids enumerated by the AutoInt1 doc: 1-4 = B / B2 / B3 / B4;
# 7 = M; 8 = PW; 9 = CW. The probe sweeps all of them so a
# non-B-mode recording would surface in the report.
_PROBE_IMG_IDS: tuple[tuple[int, str], ...] = (
    (1, "B"), (2, "B2"), (3, "B3"), (4, "B4"),
    (7, "M"), (8, "PW"), (9, "CW"),
)


@dataclass(frozen=True)
class _IdEntry:
    """One parsed ``#define id_* NUMBER`` line + its probing strategy."""

    name: str
    param_id: int
    strategy: str   # "documented_get" | "shift_inferred" | "action_only" | "unknown"
    variant: Optional[str]  # "int" | "bool" | "double" | "string" | None
    description: str


def parse_doc(doc_path: Union[str, Path] = _DOC_PATH) -> list[_IdEntry]:
    """Extract every ``#define id_* NUMBER`` and classify how to probe it.

    Strategies:

    * ``documented_get`` -- description text contains an explicit
      ``ParamGet{Int|Bool|Double|Float|String}(...)`` reference. Probed
      with that variant. (``Float`` is normalised to ``Double``: both
      return numeric and Double is higher precision.)
    * ``shift_inferred`` -- name contains ``_shift`` and has no explicit
      ParamGet hint. Probed with ParamGetInt per the production
      convention used by :data:`_PARAM_SPECS` (e.g. ``id_b_depth_shift``
      = 305 reads via ParamGetInt). Reported as ``shift_inferred`` so
      "doc says so" stays distinguishable from "convention says so".
    * ``action_only`` -- description contains ``val = 0;`` and no
      ParamGet hint. Set-only command ids; not probed.
    * ``unknown`` -- no hint at all. Not probed (crash-warning policy).

    Duplicate ``id_`` names keep the first definition (a handful are
    re-defined later inside different sections of the doc).
    """
    entries: list[_IdEntry] = []
    seen: set[str] = set()
    text = Path(doc_path).read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        m = _DEFINE_RE.match(line)
        if not m:
            continue
        name, pid = m.group(1), int(m.group(2))
        if name in seen:
            continue
        seen.add(name)
        desc = line.split("//", 1)[1].strip() if "//" in line else ""
        hint = _PARAMGET_HINT_RE.search(desc)
        if hint:
            variant = hint.group(1).lower()
            if variant == "float":
                variant = "double"
            entries.append(_IdEntry(name, pid, "documented_get", variant, desc))
        elif "_shift" in name:
            entries.append(_IdEntry(name, pid, "shift_inferred", "int", desc))
        elif _ACTION_RE.search(desc):
            entries.append(_IdEntry(name, pid, "action_only", None, desc))
        else:
            entries.append(_IdEntry(name, pid, "unknown", None, desc))
    return entries


def _safe_get(cmd, pid: int, variant: str) -> tuple[bool, Any]:
    """One ParamGet call wrapped in try/except.

    Returns ``(ok, value_or_error_repr)``. Doesn't shield against COM
    crashes, but the doc-driven variant restriction keeps us off the
    crash-prone paths.
    """
    try:
        if variant == "int":
            return True, int(cmd.ParamGetInt(pid))
        if variant == "bool":
            return True, bool(cmd.ParamGetBool(pid))
        if variant == "double":
            return True, float(cmd.ParamGetDouble(pid))
        if variant == "string":
            return True, str(cmd.ParamGetString(pid))
    except Exception as e:  # noqa: BLE001
        return False, repr(e)
    return False, f"unknown variant {variant!r}"


def _probe_region(cmd, img_id: int) -> Optional[dict]:
    """All Get* per-region calls for one img_id; ``None`` if inactive.

    Inactive panels return ``-1`` for X1 (per doc) or raise; we treat
    both as 'not present' so the report only carries populated regions.
    """
    try:
        x1 = int(cmd.GetUltrasoundX1(img_id))
    except Exception:  # noqa: BLE001
        return None
    if x1 == -1:
        return None
    out: dict[str, Any] = {"x1": x1}
    for getter, key, conv in (
        ("GetUltrasoundX2",             "x2",                    int),
        ("GetUltrasoundY1",             "y1",                    int),
        ("GetUltrasoundY2",             "y2",                    int),
        ("GetUltrasoundPhysicalDeltaX", "physical_dx_cm_per_px", float),
        ("GetUltrasoundPhysicalDeltaY", "physical_dy_cm_per_px", float),
    ):
        try:
            out[key] = conv(getattr(cmd, getter)(img_id))
        except Exception as e:  # noqa: BLE001
            out[key] = f"err: {e!r}"
    return out


def probe(
    tvd_path: Union[str, Path],
    *,
    doc_path: Union[str, Path] = _DOC_PATH,
) -> dict:
    """Sweep every file-level metadatum readable from one saved .tvd.

    Args:
        tvd_path: Local .tvd to open. Must already be staged off any
            network drive (EchoWave can't open UNC paths).
        doc_path: Override the AutoInt1Client.txt location if the SDK
            is installed somewhere non-standard.

    Returns:
        Dict with sections:

        * ``source_tvd`` / ``probed_at_iso`` -- provenance.
        * ``direct`` -- non-ParamGet getters (frame count, frame dims,
          current frame idx + time after seeking to frame 1).
        * ``ultrasound_regions`` -- ``{label: {img_id, x1, x2, y1, y2,
          physical_d{x,y}_cm_per_px}}`` for every img_id that returns
          valid geometry.
        * ``params`` -- ``{id_name: {param_id, variant, strategy,
          description, value}}`` for every documented-get +
          shift-inferred id that returned a value.
        * ``failed`` -- same shape as ``params`` plus an ``err`` field
          for ids that the probe tried and errored on (most likely
          'only valid during a live scan').
        * ``skipped`` -- ids that the doc-parser flagged as not safe to
          probe (action-only + unknown). For human review only.
        * ``dim_consistency`` -- frame 1 vs frame N dims + B-mode ROI;
          ``constant`` should be True for normal B-mode recordings.
          False is a finding worth investigating (mid-recording depth /
          mode / probe change).
    """
    # Local import: keeps `import telemed._metadata_probe`
    # cheap on machines without COM/pywin32 (e.g. CI Linux).
    from . import _extract

    reader = _extract.connect()
    reader.open(tvd_path)
    cmd = reader._cmd

    out: dict[str, Any] = {
        "source_tvd": str(tvd_path),
        "probed_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    cmd.GoToFrame1n(1, True)
    n = int(cmd.GetFramesCount)
    frame1_w = int(cmd.GetLoadedFrameWidth)
    frame1_h = int(cmd.GetLoadedFrameHeight)
    out["direct"] = {
        "n_frames": n,
        "loaded_frame_width_at_frame1": frame1_w,
        "loaded_frame_height_at_frame1": frame1_h,
        "current_frame_idx_1n_after_seek": int(cmd.GetCurrentFrameIdx1n),
        "current_frame_time_ms_at_frame1": float(cmd.GetCurrentFrameTime),
    }

    regions: dict[str, dict] = {}
    for img_id, label in _PROBE_IMG_IDS:
        region = _probe_region(cmd, img_id)
        if region is not None:
            regions[label] = {"img_id": img_id, **region}
    out["ultrasound_regions"] = regions

    entries = parse_doc(doc_path)
    params: dict[str, dict] = {}
    failed: dict[str, dict] = {}
    skipped: dict[str, dict] = {}
    for e in entries:
        record = {
            "param_id": e.param_id,
            "variant": e.variant,
            "strategy": e.strategy,
            "description": e.description,
        }
        if e.strategy in ("action_only", "unknown"):
            skipped[e.name] = record
            continue
        ok, value = _safe_get(cmd, e.param_id, e.variant)
        if ok:
            params[e.name] = {**record, "value": value}
        else:
            failed[e.name] = {**record, "err": value}
    out["params"] = params
    out["failed"] = failed
    out["skipped"] = skipped

    cmd.GoToFrame1n(n, True)
    frame_n = {
        "loaded_frame_width": int(cmd.GetLoadedFrameWidth),
        "loaded_frame_height": int(cmd.GetLoadedFrameHeight),
        "b_x1": int(cmd.GetUltrasoundX1(1)),
        "b_x2": int(cmd.GetUltrasoundX2(1)),
        "b_y1": int(cmd.GetUltrasoundY1(1)),
        "b_y2": int(cmd.GetUltrasoundY2(1)),
    }
    b_region = regions.get("B", {})
    frame_1 = {
        "loaded_frame_width": frame1_w,
        "loaded_frame_height": frame1_h,
        "b_x1": b_region.get("x1"),
        "b_x2": b_region.get("x2"),
        "b_y1": b_region.get("y1"),
        "b_y2": b_region.get("y2"),
    }
    out["dim_consistency"] = {
        "frame_1": frame_1,
        "frame_n": frame_n,
        "constant": frame_1 == frame_n,
    }
    return out


def write_report(result: dict, out_path: Union[str, Path]) -> Path:
    """Write ``result`` as JSON plus a sibling markdown summary.

    JSON is the machine-readable canonical form; markdown is for human
    eyeballing. Returns the markdown path (the JSON path is just
    ``out_path``).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))

    md_path = out_path.with_suffix(".md")
    src = Path(result.get("source_tvd", "?")).name
    lines: list[str] = [
        f"# Telemed metadata probe -- `{src}`",
        f"_Probed at {result.get('probed_at_iso', '?')}._",
        "",
        "## Direct getters",
        "",
    ]
    for k, v in result.get("direct", {}).items():
        lines.append(f"- `{k}` = `{v}`")

    lines += ["", "## Ultrasound regions", ""]
    regions = result.get("ultrasound_regions", {})
    if not regions:
        lines.append("_(none populated)_")
    else:
        for label, region in regions.items():
            kv = ", ".join(f"`{k}={v}`" for k, v in region.items())
            lines.append(f"- **{label}** -- {kv}")

    params = result.get("params", {})
    failed = result.get("failed", {})
    skipped = result.get("skipped", {})
    lines += [
        "",
        "## ParamGet sweep (summary)",
        "",
        f"- Populated: **{len(params)}**",
        f"- Failed (tried, errored): **{len(failed)}**",
        f"- Skipped (no safe variant in doc): **{len(skipped)}**",
        "",
        "### Populated",
        "",
        "| name | id | variant | strategy | value |",
        "|---|---|---|---|---|",
    ]
    for name, info in sorted(params.items(), key=lambda kv: kv[1]["param_id"]):
        v = info["value"]
        v_repr = json.dumps(v) if isinstance(v, str) else str(v)
        lines.append(
            f"| `{name}` | {info['param_id']} | {info['variant']} | "
            f"{info['strategy']} | `{v_repr}` |"
        )

    if failed:
        lines += [
            "",
            "### Failed (likely 'only valid in live scan')",
            "",
            "| name | id | variant | err |",
            "|---|---|---|---|",
        ]
        for name, info in sorted(failed.items(), key=lambda kv: kv[1]["param_id"]):
            lines.append(
                f"| `{name}` | {info['param_id']} | {info['variant']} | "
                f"`{info['err']}` |"
            )

    if skipped:
        lines += [
            "",
            "### Skipped (no documented ParamGet variant)",
            "",
            "| name | id | strategy | description |",
            "|---|---|---|---|",
        ]
        for name, info in sorted(skipped.items(), key=lambda kv: kv[1]["param_id"]):
            lines.append(
                f"| `{name}` | {info['param_id']} | {info['strategy']} | "
                f"{info['description'][:80]} |"
            )

    dc = result.get("dim_consistency", {})
    lines += [
        "",
        "## Dimension consistency (frame 1 vs frame N)",
        "",
        f"- Constant: **{dc.get('constant')}**",
    ]
    if dc.get("constant") is False:
        lines.append(f"  - frame 1: `{dc.get('frame_1')}`")
        lines.append(f"  - frame N: `{dc.get('frame_n')}`")

    md_path.write_text("\n".join(lines))
    return md_path
