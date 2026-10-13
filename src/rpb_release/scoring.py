# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Moonlight router score transforms and configuration checks."""

from __future__ import annotations

from typing import Any

import torch


MOONLIGHT_SCORE_FUNCTION = "sigmoid"


def router_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert Moonlight router logits to normalized sigmoid probabilities."""
    scores = torch.sigmoid(logits.float())
    denominator = scores.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)
    return scores / denominator


def validate_moonlight_router(model: Any) -> None:
    """Require the sigmoid router used by Moonlight-16B-A3B."""
    model_config = getattr(model, "config", None)
    detected = next(
        (
            value
            for attribute in (
                "scoring_func",
                "moe_router_score_function",
                "router_score_function",
            )
            if (value := getattr(model_config, attribute, None)) is not None
        ),
        None,
    )
    if detected != MOONLIGHT_SCORE_FUNCTION:
        raise ValueError(
            "Moonlight prior collection requires a sigmoid router; "
            f"model config reports {detected!r}"
        )
