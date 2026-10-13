# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Frozen-model router-prior collection."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .scoring import router_probabilities
from .spans import ANSWER_SPAN, QUESTION_SPAN, detect_spans


def aggregate_qa_prior(probabilities: torch.Tensor, span_ids: Sequence[int]) -> torch.Tensor:
    """Average ``[layers, tokens, experts]`` probabilities over Q and A bodies."""
    if probabilities.ndim != 3:
        raise ValueError("router probabilities must have shape [layers, tokens, experts]")
    if len(span_ids) != probabilities.shape[1]:
        raise ValueError("span IDs must align with router probability tokens")
    spans = torch.tensor(span_ids, dtype=torch.long)
    question_mask = spans == QUESTION_SPAN
    answer_mask = spans == ANSWER_SPAN
    if not bool(question_mask.any()) or not bool(answer_mask.any()):
        raise ValueError("question and answer spans must both be non-empty")
    question = probabilities[:, question_mask, :].mean(dim=1)
    answer = probabilities[:, answer_mask, :].mean(dim=1)
    return torch.stack((question, answer), dim=0)


class GateLogitsCollector:
    """Capture Moonlight ``MoEGate`` logits by replaying its projection."""

    def __init__(self, model: torch.nn.Module):
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._captured: list[torch.Tensor] = []
        gates = [module for module in model.modules() if module.__class__.__name__ == "MoEGate"]
        if not gates:
            raise ValueError("No modules named MoEGate were found in the model")
        for gate in gates:
            if not hasattr(gate, "weight"):
                raise ValueError("MoEGate module has no weight parameter")
            self._handles.append(gate.register_forward_pre_hook(self._hook))

    def _hook(self, module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise ValueError("MoEGate did not receive a tensor hidden state")
        hidden = inputs[0]
        if hidden.ndim == 3:
            if hidden.shape[0] != 1:
                raise ValueError("prior collection requires batch size 1")
            hidden = hidden[0]
        if hidden.ndim != 2:
            raise ValueError("MoEGate hidden state must have shape [tokens, hidden]")
        logits = F.linear(hidden.float(), module.weight.float())
        self._captured.append(logits.detach().cpu())

    def pop(self) -> torch.Tensor:
        if not self._captured:
            raise RuntimeError("No router logits were captured")
        captured, self._captured = self._captured, []
        token_counts = {tensor.shape[0] for tensor in captured}
        expert_counts = {tensor.shape[1] for tensor in captured}
        if len(token_counts) != 1 or len(expert_counts) != 1:
            raise ValueError("captured router logits have inconsistent shapes")
        return torch.stack(captured, dim=0)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> GateLogitsCollector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def iter_records(paths: Sequence[Path]) -> Iterable[tuple[int, dict[str, object]]]:
    """Yield validated JSONL rows with stable, ordered sample IDs."""
    sample_id = 0
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                for key in ("input", "output"):
                    if not isinstance(record.get(key), str) or not record[key]:
                        raise ValueError(
                            f"{path}:{line_number}: {key!r} must be a non-empty string"
                        )
                yield sample_id, record
                sample_id += 1


def encode_sample(
    tokenizer: Any,
    record: dict[str, object],
    *,
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    """Tokenize a rendered row once and label its semantic Q/A bodies."""
    rendered = str(record["input"]) + str(record["output"])
    input_ids = list(tokenizer.encode(rendered, add_special_tokens=False))[:max_seq_len]
    if not input_ids:
        raise ValueError("tokenization produced an empty example")
    span_ids = detect_spans(input_ids, tokenizer)
    if QUESTION_SPAN not in span_ids or ANSWER_SPAN not in span_ids:
        raise ValueError("question and answer spans must both be non-empty after truncation")
    return input_ids, span_ids


def _model_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        return next(model.parameters()).device


@torch.inference_mode()
def collect(
    model: torch.nn.Module,
    tokenizer: Any,
    data_files: Sequence[Path],
    *,
    max_seq_len: int,
) -> torch.Tensor:
    """Collect one Q/A expert-prior tensor per ordered training row."""
    priors: list[torch.Tensor] = []
    input_device = _model_input_device(model)
    with GateLogitsCollector(model) as capture:
        for sample_id, record in iter_records(data_files):
            input_ids, span_ids = encode_sample(tokenizer, record, max_seq_len=max_seq_len)
            tokens = torch.tensor([input_ids], dtype=torch.long, device=input_device)
            model(input_ids=tokens, use_cache=False)
            probabilities = router_probabilities(capture.pop())
            try:
                priors.append(aggregate_qa_prior(probabilities, span_ids))
            except ValueError as error:
                raise ValueError(f"sample {sample_id}: {error}") from error
    if not priors:
        raise ValueError("no samples were found")
    if len({tuple(prior.shape) for prior in priors}) != 1:
        raise ValueError("router layer/expert counts changed between samples")
    return torch.stack(priors, dim=0)
