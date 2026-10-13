#!/usr/bin/env bash
# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/Megatron-Bridge" >&2
  exit 2
fi

bridge_root=$1
expected_bridge=9c9dd848966322fc3ed7706747ec13219ef49dda
expected_megatron=55ac7082517c3878ae653c07c09c534b8aed49f6
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
patch_path=$script_dir/../patches/rpb-megatron-bridge-v0.3.1.patch

actual_bridge=$(git -C "$bridge_root" rev-parse HEAD)
gitlink_megatron=$(git -C "$bridge_root" rev-parse HEAD:3rdparty/Megatron-LM)
submodule_root=$bridge_root/3rdparty/Megatron-LM
submodule_status=$(git -C "$bridge_root" submodule status -- 3rdparty/Megatron-LM)
if [[ $submodule_status == -* ]]; then
  echo "Megatron-LM submodule is not initialized; run git submodule update --init --recursive" >&2
  exit 1
fi
if ! actual_megatron=$(git -C "$submodule_root" rev-parse HEAD 2>/dev/null); then
  echo "Megatron-LM submodule is not initialized; run git submodule update --init --recursive" >&2
  exit 1
fi

if [[ $actual_bridge != "$expected_bridge" ]]; then
  echo "Megatron-Bridge mismatch: expected $expected_bridge, got $actual_bridge" >&2
  exit 1
fi
if [[ $gitlink_megatron != "$expected_megatron" || $actual_megatron != "$expected_megatron" ]]; then
  echo "Megatron-LM mismatch: expected $expected_megatron, gitlink=$gitlink_megatron checkout=$actual_megatron" >&2
  exit 1
fi

if git -C "$bridge_root" apply --check "$patch_path" 2>/dev/null; then
  echo "patch applies cleanly to the pinned upstream checkout"
elif git -C "$bridge_root" apply --reverse --check "$patch_path" 2>/dev/null; then
  echo "patch is already applied cleanly to the pinned upstream checkout"
else
  echo "patch neither applies nor cleanly reverses on this checkout" >&2
  exit 1
fi
