# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

from pathlib import Path

import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_readme_pins_upstream():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "9c9dd848966322fc3ed7706747ec13219ef49dda" in readme
    assert "55ac7082517c3878ae653c07c09c534b8aed49f6" in readme


def test_adapted_code_has_third_party_notices():
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    rpb_source = (ROOT / "patches" / "source" / "rpb.py").read_text(encoding="utf-8")
    launcher = (ROOT / "examples" / "train.py").read_text(encoding="utf-8")

    assert "55ac7082517c3878ae653c07c09c534b8aed49f6" in notices
    assert "9c9dd848966322fc3ed7706747ec13219ef49dda" in notices
    assert "BSD 3-Clause" in notices
    assert "Apache License" in notices
    assert "THIRD_PARTY_NOTICES.md" in rpb_source
    assert "THIRD_PARTY_NOTICES.md" in launcher


def test_patch_only_adds_the_rpb_module():
    patch = (ROOT / "patches" / "rpb-megatron-bridge-v0.3.1.patch").read_text(encoding="utf-8")
    changed = [line[6:] for line in patch.splitlines() if line.startswith("+++ b/")]
    assert changed == ["src/megatron/bridge/training/rpb.py"]
    assert "Router Prior Bias" in patch


def test_prior_extra_contains_moonlight_runtime_dependencies():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    dependencies = project["optional-dependencies"]["prior"]

    assert any(item.startswith("accelerate") for item in dependencies)
    assert any(item.startswith("tiktoken") for item in dependencies)


def test_readme_limits_the_release_to_single_turn_rows():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())

    assert "single-turn Q/A rows only" in readme
    assert "Do not put earlier assistant turns in `input`" in normalized


def test_runtime_artifacts_are_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "artifacts/" in ignore
    assert "experiments/" in ignore
    assert "nemo_experiments/" in ignore
    assert "*.log" in ignore
