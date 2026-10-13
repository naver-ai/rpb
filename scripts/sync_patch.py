#!/usr/bin/env python3
# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Keep the readable Megatron-Bridge source and distributable patch in sync."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = "src/megatron/bridge/training/rpb.py"
SOURCE_PATH = ROOT / "patches" / "source" / "rpb.py"
PATCH_PATH = ROOT / "patches" / "rpb-megatron-bridge-v0.3.1.patch"


def extract_source(patch: str) -> str:
    """Extract the added file from a unified Git patch."""
    lines = patch.splitlines()
    marker = f"+++ b/{TARGET}"
    try:
        start = lines.index(marker) + 1
    except ValueError as error:
        raise ValueError(f"patch does not add {TARGET}") from error

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
        if line.startswith(("+", " ")):
            source.append(line[1:])
    return "\n".join(source) + "\n"


def render_patch(source: str) -> str:
    """Render a deterministic new-file patch for the pinned Bridge checkout."""
    if not source.endswith("\n"):
        raise ValueError("canonical source must end with a newline")
    source_bytes = source.encode()
    blob = hashlib.sha1(f"blob {len(source_bytes)}\0".encode() + source_bytes).hexdigest()[:7]
    lines = source[:-1].split("\n")
    header = (
        f"diff --git a/{TARGET} b/{TARGET}\n"
        "new file mode 100644\n"
        f"index 0000000..{blob}\n"
        "--- /dev/null\n"
        f"+++ b/{TARGET}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
    )
    return header + "".join(f"+{line}\n" for line in lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="Fail if source and patch differ")
    action.add_argument("--extract", action="store_true", help="Bootstrap source from the patch")
    args = parser.parse_args()

    if args.extract:
        SOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SOURCE_PATH.write_text(
            extract_source(PATCH_PATH.read_text(encoding="utf-8")), encoding="utf-8"
        )
        return

    rendered = render_patch(SOURCE_PATH.read_text(encoding="utf-8"))
    if args.check:
        if PATCH_PATH.read_text(encoding="utf-8") != rendered:
            raise SystemExit("patch is stale; run scripts/sync_patch.py")
        return
    PATCH_PATH.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
