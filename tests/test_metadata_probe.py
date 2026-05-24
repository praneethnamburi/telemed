"""Tests for the AutoInt1Client.txt doc-parser used by the metadata
probe. The COM probe itself can't be unit-tested without a running
EchoWave II instance, but the parser is pure text processing and
worth pinning so the strategy classification doesn't silently drift.
"""
from __future__ import annotations

import pytest

from telemed import _metadata_probe as mp


_DOC_AVAILABLE = mp._DOC_PATH.is_file()
pytestmark = pytest.mark.skipif(
    not _DOC_AVAILABLE,
    reason=f"AutoInt1Client.txt not installed at {mp._DOC_PATH}",
)


def test_parse_doc_classifies_known_ids():
    """Spot-check one id from each strategy bucket against the live doc."""
    by_name = {e.name: e for e in mp.parse_doc()}

    # documented_get / bool -- description has `ParamGetBool(...)`.
    e = by_name["id_b_is_thi_frequency"]
    assert e.param_id == 177
    assert e.strategy == "documented_get"
    assert e.variant == "bool"

    # documented_get / int -- description has `ParamGetInt(...)`.
    e = by_name["id_get_current_beamformer_code"]
    assert e.param_id == 915
    assert e.strategy == "documented_get"
    assert e.variant == "int"

    # documented_get / string -- description has `ParamGetString(...)`.
    e = by_name["id_get_current_beamformer_name"]
    assert e.param_id == 916
    assert e.strategy == "documented_get"
    assert e.variant == "string"

    # shift_inferred -- name ends in _shift, no explicit ParamGet hint,
    # works via ParamGetInt per the production convention.
    e = by_name["id_b_depth_shift"]
    assert e.param_id == 305
    assert e.strategy == "shift_inferred"
    assert e.variant == "int"

    # action_only -- description ends with `val = 0;` and has no
    # ParamGet hint.
    e = by_name["id_freeze_run"]
    assert e.param_id == 100
    assert e.strategy == "action_only"
    assert e.variant is None


def test_parse_doc_dedupes_repeated_names():
    """Each id_* name should appear at most once in the parsed list,
    even if the doc defines it twice (a handful are re-listed in
    different sections)."""
    entries = mp.parse_doc()
    names = [e.name for e in entries]
    assert len(names) == len(set(names)), "duplicate names in parsed entries"


def test_parse_doc_returns_nontrivial_count():
    """Sanity floor: the doc declares dozens of ids; if the parser
    only returns a handful, the regex has broken silently."""
    assert len(mp.parse_doc()) > 50
