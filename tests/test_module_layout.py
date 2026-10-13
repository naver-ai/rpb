# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

from __future__ import annotations

from pathlib import Path


def test_public_tools_reexport_the_focused_library_api(collector_module, validator_module):
    from rpb_release import artifacts, collection, scoring, spans

    assert collector_module.detect_spans is spans.detect_spans
    assert collector_module.router_probabilities is scoring.router_probabilities
    assert collector_module.validate_moonlight_router is scoring.validate_moonlight_router
    assert collector_module.GateLogitsCollector is collection.GateLogitsCollector
    assert collector_module.make_payload is artifacts.make_payload
    assert collector_module.data_sha256 is artifacts.data_sha256
    assert validator_module.validate_prior is artifacts.validate_prior


def test_megatron_patch_matches_its_canonical_source(rpb_source_from_patch):
    repository_root = Path(__file__).resolve().parents[1]
    canonical_source = repository_root / "patches" / "source" / "rpb.py"

    assert canonical_source.read_text(encoding="utf-8") == rpb_source_from_patch
