# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Reusable utilities for the Router Prior Bias code release."""

from .artifacts import data_sha256, make_payload, validate_prior
from .collection import GateLogitsCollector, aggregate_qa_prior, collect, encode_sample
from .scoring import router_probabilities, validate_moonlight_router
from .spans import ANSWER_SPAN, QUESTION_SPAN, STRUCTURAL_SPAN, detect_spans

__all__ = [
    "ANSWER_SPAN",
    "GateLogitsCollector",
    "QUESTION_SPAN",
    "STRUCTURAL_SPAN",
    "aggregate_qa_prior",
    "collect",
    "data_sha256",
    "detect_spans",
    "encode_sample",
    "make_payload",
    "router_probabilities",
    "validate_moonlight_router",
    "validate_prior",
]
