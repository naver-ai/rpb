# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Moonlight Q/A span detection."""

from __future__ import annotations

from typing import Any


QUESTION_SPAN = 0
ANSWER_SPAN = 2
STRUCTURAL_SPAN = -1


def _marker_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"tokenizer is missing required marker {token!r}")
    return int(token_id)


def detect_spans(token_ids: list[int], tokenizer: Any) -> list[int]:
    """Label Moonlight semantic Q/A bodies and leave chat-control tokens structural."""
    user = _marker_id(tokenizer, "<|im_user|>")
    assistant = _marker_id(tokenizer, "<|im_assistant|>")
    system = _marker_id(tokenizer, "<|im_system|>")
    middle = _marker_id(tokenizer, "<|im_middle|>")
    end = _marker_id(tokenizer, "<|im_end|>")
    labels = {user: QUESTION_SPAN, assistant: ANSWER_SPAN, system: None}
    spans = [STRUCTURAL_SPAN] * len(token_ids)
    cursor = 0
    while cursor < len(token_ids):
        if token_ids[cursor] not in labels:
            cursor += 1
            continue
        middle_index = next(
            (index for index in range(cursor + 1, len(token_ids)) if token_ids[index] == middle),
            -1,
        )
        if middle_index < 0:
            break
        end_index = next(
            (index for index in range(middle_index + 1, len(token_ids)) if token_ids[index] == end),
            len(token_ids),
        )
        label = labels[token_ids[cursor]]
        if label is not None:
            spans[middle_index + 1 : end_index] = [label] * (end_index - middle_index - 1)
        cursor = end_index + 1
    return spans
