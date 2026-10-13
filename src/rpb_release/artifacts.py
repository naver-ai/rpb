# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""RPB prior artifact construction and validation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .scoring import MOONLIGHT_SCORE_FUNCTION


SCHEMAS = ("rpb_prior_v1", "rpb_prior_v2")
SPAN_NAMES = ["question", "answer"]
REQUIRED_METADATA = (
    "model_id",
    "tokenizer_id",
    "family",
    "router_score_function",
    "prior_score_function",
    "top_k",
    "max_seq_len",
    "data_sha256",
)


def data_sha256(paths: Sequence[Path]) -> str:
    """Hash ordered file contents without embedding machine-specific paths."""
    digest = hashlib.sha256()
    for raw_path in paths:
        content = Path(raw_path).read_bytes()
        digest.update(len(content).to_bytes(8, byteorder="big"))
        digest.update(content)
    return digest.hexdigest()


def make_payload(
    priors: torch.Tensor,
    *,
    model_id: str,
    tokenizer_id: str,
    top_k: int,
    max_seq_len: int,
    source_hash: str,
) -> dict[str, object]:
    """Build the stable v2 artifact payload."""
    if priors.ndim != 4 or priors.shape[1] != 2:
        raise ValueError("priors must have shape [samples, 2, layers, experts]")
    prior_qa = priors.detach().cpu().to(torch.float16).contiguous()
    return {
        "prior_qa": prior_qa,
        "metadata": {
            "schema": "rpb_prior_v2",
            "shape": list(prior_qa.shape),
            "span_names": SPAN_NAMES,
            "model_id": model_id,
            "tokenizer_id": tokenizer_id,
            "family": "moonlight",
            "router_score_function": MOONLIGHT_SCORE_FUNCTION,
            "prior_score_function": MOONLIGHT_SCORE_FUNCTION,
            "top_k": top_k,
            "max_seq_len": max_seq_len,
            "data_sha256": source_hash,
            "n_missing_uniform_fill": 0,
        },
    }


def _normalized_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict) or metadata.get("schema") not in SCHEMAS:
        raise ValueError("unsupported or missing RPB schema")
    normalized = dict(metadata)
    if normalized["schema"] == "rpb_prior_v1":
        normalized["router_score_function"] = normalized.get("score_function")
        normalized["prior_score_function"] = normalized.get("score_function")
    return normalized


def validate_payload(
    payload: Any,
    *,
    data_files: Sequence[Path] | None = None,
    expected_model: str | None = None,
    expected_layers: int | None = None,
    expected_experts: int | None = None,
) -> dict[str, object]:
    """Validate a loaded artifact and return a compact public summary."""
    if not isinstance(payload, dict) or "prior_qa" not in payload or "metadata" not in payload:
        raise ValueError("artifact must contain prior_qa and metadata")
    prior = payload["prior_qa"]
    if not isinstance(prior, torch.Tensor) or prior.ndim != 4 or prior.shape[1] != 2:
        raise ValueError("prior_qa must have shape [samples, 2, layers, experts]")
    metadata = _normalized_metadata(payload["metadata"])

    if list(prior.shape) != metadata.get("shape"):
        raise ValueError("metadata shape does not match prior_qa")
    if metadata.get("span_names") != SPAN_NAMES:
        raise ValueError("span_names must be ['question', 'answer']")
    if metadata.get("n_missing_uniform_fill") != 0:
        raise ValueError("artifact contains missing/uniform-filled prior rows")
    missing = [key for key in REQUIRED_METADATA if metadata.get(key) in (None, "")]
    if missing:
        raise ValueError(f"artifact metadata is missing required fields: {missing}")
    if metadata["family"] != "moonlight":
        raise ValueError("artifact family must be moonlight")
    for key in ("router_score_function", "prior_score_function"):
        if metadata[key] != MOONLIGHT_SCORE_FUNCTION:
            raise ValueError(f"{key} must be sigmoid for Moonlight")

    values = prior.float()
    if not torch.isfinite(values).all():
        raise ValueError("prior probabilities must be finite")
    if bool((values < 0).any()):
        raise ValueError("prior probabilities must be non-negative")
    sums = values.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=5e-3, rtol=0):
        raise ValueError("prior probabilities must sum to one over experts")

    if expected_model is not None and metadata["model_id"] != expected_model:
        raise ValueError("model ID does not match --expected-model")
    if expected_layers is not None and prior.shape[2] != expected_layers:
        raise ValueError("layer count does not match --expected-layers")
    if expected_experts is not None and prior.shape[3] != expected_experts:
        raise ValueError("expert count does not match --expected-experts")
    if data_files and data_sha256(data_files) != metadata["data_sha256"]:
        raise ValueError("data SHA-256 does not match the artifact")

    return {
        "schema": metadata["schema"],
        "shape": list(prior.shape),
        "model_id": metadata["model_id"],
        "router_score_function": metadata["router_score_function"],
        "prior_score_function": metadata["prior_score_function"],
        "data_sha256": metadata["data_sha256"],
        "status": "valid",
    }


def validate_prior(
    prior_path: Path,
    *,
    data_files: Sequence[Path] | None = None,
    expected_model: str | None = None,
    expected_layers: int | None = None,
    expected_experts: int | None = None,
) -> dict[str, object]:
    """Load and validate an RPB prior artifact."""
    payload = torch.load(prior_path, map_location="cpu", weights_only=True)
    return validate_payload(
        payload,
        data_files=data_files,
        expected_model=expected_model,
        expected_layers=expected_layers,
        expected_experts=expected_experts,
    )
