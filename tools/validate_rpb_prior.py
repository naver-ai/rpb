#!/usr/bin/env python3
# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Validate an RPB prior artifact and its optional source-data fingerprint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rpb_release.artifacts import data_sha256 as _data_sha256
from rpb_release.artifacts import validate_prior


__all__ = ["_data_sha256", "_parse_args", "main", "validate_prior"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior", type=Path)
    parser.add_argument("--data-jsonl", nargs="+", type=Path)
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-layers", type=int)
    parser.add_argument("--expected-experts", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = validate_prior(
        args.prior,
        data_files=args.data_jsonl,
        expected_model=args.expected_model,
        expected_layers=args.expected_layers,
        expected_experts=args.expected_experts,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
