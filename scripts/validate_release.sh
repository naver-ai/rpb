#!/usr/bin/env bash
# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/rpb-uv-cache} uv sync --extra test
UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/rpb-uv-cache} uv run pytest -q
UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/rpb-uv-cache} uv run python scripts/sync_patch.py --check
UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/rpb-uv-cache} uv run python scripts/check_release_content.py
UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/rpb-uv-cache} uv run python -m compileall -q src tools examples scripts patches/source
UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/rpb-uv-cache} uv run ruff check src tools examples tests scripts patches/source
UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/rpb-uv-cache} uv run ruff format --check src tools examples tests scripts patches/source
bash -n scripts/check_patch.sh scripts/validate_release.sh

tracked_artifacts='^(artifacts|checkpoints|data|experiments|logs|nemo_experiments|outputs|wandb)/|(^|/)__pycache__/'
if git ls-files | rg -n "$tracked_artifacts"; then
  echo "release contains tracked runtime artifacts" >&2
  exit 1
fi

echo "release validation passed"
