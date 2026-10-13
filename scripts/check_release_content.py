#!/usr/bin/env python3
# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""Check release files for missing attribution and private content candidates."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
HEADER = "# Router Prior Bias (RPB)\n# Copyright (c) 2026-present NAVER Corp.\n# Apache-2.0\n"
SOURCE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml"}
# Add only reviewed, wholly upstream or generated files here. Adapted files
# must retain both the NAVER header and their upstream notices.
HEADER_EXEMPTIONS: set[str] = set()
PRIVATE_PATTERNS = {
    "internal hostname": re.compile(
        r"\b(?:(?:[a-z0-9-]+\.)*(?:navercorp\.com|navercloud\.com)|"
        r"(?:[a-z0-9-]+\.)+(?:internal|corp))(?=[/:\s]|$)",
        re.IGNORECASE,
    ),
    "machine-specific absolute path": re.compile(r"/(?:Users|home|mnt|scratch)/[\w.-]+"),
    "private network address": re.compile(
        r"(?<![\w.-])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
        r"172\.(?:1[6-9]|2[0-9]|3[01])(?:\.\d{1,3}){2})(?![\w.])"
    ),
    "private cluster identifier": re.compile(r"\b(?:p-(?:sp-)?medical-agi|private-[a-z0-9-]+)\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential candidate": re.compile(
        r"\b(?:AKIA[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,})\b"
    ),
}


def main() -> None:
    result = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    names = sorted(set(result.decode().strip("\0").split("\0")))
    issues: list[str] = []
    for name in names:
        path = ROOT / name
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        source = content.split("\n", 1)[1] if content.startswith("#!") else content
        if path.suffix in SOURCE_SUFFIXES and name not in HEADER_EXEMPTIONS:
            if not source.startswith(HEADER):
                issues.append(f"{name}: missing NAVER copyright/license header")
        for number, line in enumerate(content.splitlines(), start=1):
            for label, pattern in PRIVATE_PATTERNS.items():
                if name == "uv.lock" and label == "private network address":
                    if line.startswith("version = "):
                        continue
                if pattern.search(line):
                    # Report locations without echoing potential secrets.
                    issues.append(f"{name}:{number}: {label}")
    if issues:
        raise SystemExit("\n".join(issues))
    print(f"release attribution and public-content checks passed ({len(names)} files)")


if __name__ == "__main__":
    main()
