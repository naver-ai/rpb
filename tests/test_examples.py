# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_train_module():
    path = ROOT / "examples" / "train.py"
    spec = importlib.util.spec_from_file_location("rpb_example_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_public_moonlight_config_is_rpb_safe():
    config = yaml.safe_load((ROOT / "examples/moonlight/config.yaml").read_text(encoding="utf-8"))
    assert config["model"]["id"] == "moonshotai/Moonlight-16B-A3B"
    assert "family" not in config["model"]
    assert "router_score_function" not in config["model"]
    assert config["model"]["pipeline_parallel"] == 1
    assert config["model"]["context_parallel"] == 1
    assert config["model"]["tensor_parallel"] == 2
    assert config["model"]["expert_parallel"] == 8
    assert config["model"]["preserve_expert_bias"] is True
    assert config["model"]["recompute_granularity"] == "selective"
    assert config["model"]["recompute_modules"] == [
        "layernorm",
        "moe",
        "moe_act",
        "core_attn",
    ]
    assert config["data"]["packed"] is False
    assert config["data"]["prompt_template"] == "{input}{output}"
    assert config["data"]["seq_length"] == 8192
    assert config["data"]["seed"] == 42
    assert config["train"] == {
        "iterations": 782,
        "global_batch_size": 128,
        "micro_batch_size": 1,
        "seed": 42,
    }
    assert config["optimizer"] == {
        "max_lr": 1.0e-5,
        "min_lr": 1.0e-6,
        "warmup_iterations": 40,
        "weight_decay": 0.1,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_eps": 1.0e-8,
        "clip_grad": 1.0,
        "lr_decay_style": "cosine",
    }
    assert config["rpb"] == {
        "prior_path": "artifacts/rpb_prior.pt",
        "beta": 1.0,
        "eps": 0.05,
        "c": 5.0,
        "route_only": True,
    }


def test_example_config_validator_rejects_unsupported_modes():
    module = _load_train_module()
    bad = {
        "model": {
            "id": "moonshotai/Moonlight-16B-A3B",
            "pipeline_parallel": 2,
            "context_parallel": 1,
        },
        "data": {"packed": False},
        "rpb": {"route_only": True},
    }
    with pytest.raises(ValueError, match="pipeline_parallel"):
        module.validate_public_config(bad)


def test_router_configuration_preserves_frozen_moonlight_expert_bias():
    module = _load_train_module()
    provider = type(
        "Provider",
        (),
        {
            "moe_router_score_function": "sigmoid",
            "moe_router_enable_expert_bias": True,
            "moe_router_bias_update_rate": 1e-3,
        },
    )()

    module.configure_router(
        provider,
        {
            "preserve_expert_bias": True,
        },
    )

    assert provider.moe_router_enable_expert_bias is True
    assert provider.moe_router_bias_update_rate == 0.0
    assert provider.moe_router_load_balancing_type == "none"
    assert provider.moe_aux_loss_coeff == 0.0


def test_moonlight_reserved_tokenizer_ids_do_not_resize_checkpoint_vocab():
    module = _load_train_module()

    def original(model_vocab_size, tokenizer_vocab_size):
        if model_vocab_size < tokenizer_vocab_size:
            raise ValueError("model vocabulary is smaller than tokenizer vocabulary")
        return model_vocab_size, False

    setup = SimpleNamespace(_validate_and_set_vocab_size=original)
    module._allow_moonlight_tokenizer_len_mismatch(163840, setup_module=setup)

    assert setup._validate_and_set_vocab_size(
        model_vocab_size=163840, tokenizer_vocab_size=163842
    ) == (163840, False)
    with pytest.raises(ValueError, match="smaller"):
        setup._validate_and_set_vocab_size(163839, 163842)


def test_vocab_validation_is_unchanged_for_non_moonlight_sizes():
    module = _load_train_module()

    def original(model_vocab_size, tokenizer_vocab_size):
        return model_vocab_size, True

    setup = SimpleNamespace(_validate_and_set_vocab_size=original)

    module._allow_moonlight_tokenizer_len_mismatch(32000, setup_module=setup)

    assert setup._validate_and_set_vocab_size is original


def test_example_config_validator_rejects_other_models():
    module = _load_train_module()
    with pytest.raises(ValueError, match="Moonlight-16B-A3B"):
        module.validate_public_config({"model": {"id": "example/other"}})


def test_readme_uses_reported_world_size():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "torchrun --nnodes=16 --nproc-per-node=8" in readme
