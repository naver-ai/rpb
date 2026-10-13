# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = REPOSITORY_ROOT / "patches" / "rpb-megatron-bridge-v0.3.1.patch"
PATCH_SOURCE_PATH = REPOSITORY_ROOT / "patches" / "source" / "rpb.py"
RPB_TARGET = "src/megatron/bridge/training/rpb.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _materialize_rpb_from_patch(destination: Path) -> Path:
    lines = PATCH_PATH.read_text(encoding="utf-8").splitlines()
    target_marker = f"+++ b/{RPB_TARGET}"
    try:
        start = lines.index(target_marker) + 1
    except ValueError as error:
        raise RuntimeError(f"patch does not add {RPB_TARGET}") from error
    source: list[str] = []
    in_hunk = False
    for line in lines[start:]:
        if line.startswith("diff --git "):
            break
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk or line.startswith("\\ No newline"):
            continue
        if line.startswith("+"):
            source.append(line[1:])
        elif line.startswith(" "):
            source.append(line[1:])
    module_path = destination / "rpb.py"
    module_path.write_text("\n".join(source) + "\n", encoding="utf-8")
    return module_path


@pytest.fixture(scope="session")
def rpb_source_from_patch(tmp_path_factory):
    module_path = _materialize_rpb_from_patch(tmp_path_factory.mktemp("patch-source"))
    return module_path.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def rpb_module():
    bridge_root = os.environ.get("MEGATRON_BRIDGE_ROOT")
    if bridge_root:
        module_path = Path(bridge_root) / RPB_TARGET
    else:
        module_path = PATCH_SOURCE_PATH
    if not module_path.is_file():
        pytest.fail(f"RPB patch module is missing: {module_path}")
    return _load_module("rpb_under_test", module_path)


@pytest.fixture
def collector_module():
    return _load_module(
        "collect_rpb_prior_under_test",
        REPOSITORY_ROOT / "tools" / "collect_rpb_prior.py",
    )


@pytest.fixture
def validator_module():
    return _load_module(
        "validate_rpb_prior_under_test",
        REPOSITORY_ROOT / "tools" / "validate_rpb_prior.py",
    )


@pytest.fixture
def prior_payload():
    def build(values: torch.Tensor, **overrides):
        metadata = {
            "schema": "rpb_prior_v2",
            "shape": list(values.shape),
            "span_names": ["question", "answer"],
            "model_id": "example/model",
            "tokenizer_id": "example/model",
            "family": "moonlight",
            "router_score_function": "sigmoid",
            "prior_score_function": "sigmoid",
            "top_k": 1,
            "max_seq_len": 128,
            "data_sha256": "0" * 64,
            "n_missing_uniform_fill": 0,
        }
        metadata.update(overrides)
        return {"prior_qa": values, "metadata": metadata}

    return build
