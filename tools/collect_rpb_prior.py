#!/usr/bin/env python3
# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Collect question/answer router priors from a Hugging Face MoE checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rpb_release.artifacts import data_sha256, make_payload
from rpb_release.collection import (
    GateLogitsCollector,
    aggregate_qa_prior,
    collect,
    encode_sample as _encode_sample,
    iter_records as _records,
)
from rpb_release.scoring import router_probabilities, validate_moonlight_router
from rpb_release.spans import detect_spans


__all__ = [
    "GateLogitsCollector",
    "_encode_sample",
    "_parse_args",
    "_records",
    "aggregate_qa_prior",
    "collect",
    "data_sha256",
    "detect_spans",
    "main",
    "make_payload",
    "router_probabilities",
    "validate_moonlight_router",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local checkpoint")
    parser.add_argument("--tokenizer", help="Tokenizer ID; defaults to --model")
    parser.add_argument("--data-jsonl", required=True, type=Path)
    parser.add_argument("--out-path", required=True, type=Path)
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def _load_model_and_tokenizer(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_id = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    return model, tokenizer, tokenizer_id


def main() -> None:
    args = _parse_args()
    if args.top_k <= 0 or args.max_seq_len <= 1:
        raise ValueError("top-k must be positive and max-seq-len must be greater than one")
    if not args.data_jsonl.is_file():
        raise FileNotFoundError(args.data_jsonl)

    data_files = [args.data_jsonl]
    model, tokenizer, tokenizer_id = _load_model_and_tokenizer(args)
    validate_moonlight_router(model)
    priors = collect(
        model,
        tokenizer,
        data_files,
        max_seq_len=args.max_seq_len,
    )
    payload = make_payload(
        priors,
        model_id=args.model,
        tokenizer_id=tokenizer_id,
        top_k=args.top_k,
        max_seq_len=args.max_seq_len,
        source_hash=data_sha256(data_files),
    )
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_path.with_suffix(args.out_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.out_path)
    print(json.dumps(payload["metadata"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
