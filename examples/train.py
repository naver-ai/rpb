#!/usr/bin/env python3
# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this launcher are adapted from the NVIDIA Megatron-Bridge
# Moonlight SFT recipe, licensed under Apache-2.0. See THIRD_PARTY_NOTICES.md.
# Upstream portions: Copyright (c) 2025 NVIDIA CORPORATION.

"""Minimal Megatron-Bridge v0.3.1 SFT launcher with Router Prior Bias."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml


MOONLIGHT_MODEL_ID = "moonshotai/Moonlight-16B-A3B"
LOGGER = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    validate_public_config(config)
    return config


def validate_public_config(config: dict[str, Any]) -> None:
    model = config.get("model", {})
    data = config.get("data", {})
    rpb = config.get("rpb", {})
    if model.get("id") != MOONLIGHT_MODEL_ID:
        raise ValueError(f"this release supports only model.id={MOONLIGHT_MODEL_ID}")
    if model.get("pipeline_parallel") != 1:
        raise ValueError("RPB examples require model.pipeline_parallel=1")
    if model.get("context_parallel") != 1:
        raise ValueError("RPB examples require model.context_parallel=1")
    if data.get("packed") is not False:
        raise ValueError("RPB examples require data.packed=false")
    if data.get("prompt_template") != "{input}{output}":
        raise ValueError("RPB examples require prompt_template='{input}{output}'")
    if rpb.get("route_only") is not True:
        raise ValueError("this release supports only rpb.route_only=true")


def configure_router(provider: Any, model: dict[str, Any]) -> None:
    detected_score_function = getattr(provider, "moe_router_score_function", None)
    if detected_score_function != "sigmoid":
        raise ValueError(
            "Moonlight requires a sigmoid router, "
            f"but the provider reports {detected_score_function!r}"
        )
    provider.moe_router_load_balancing_type = "none"
    provider.moe_aux_loss_coeff = 0.0
    provider.moe_router_enable_expert_bias = bool(model.get("preserve_expert_bias", False))
    provider.moe_router_bias_update_rate = 0.0
    provider.moe_expert_capacity_factor = None
    provider.moe_router_fusion = False


def _allow_moonlight_tokenizer_len_mismatch(
    model_vocab_size: int | None, *, setup_module: Any | None = None
) -> None:
    """Allow Moonlight's two reserved tokenizer IDs without resizing weights.

    Moonlight's custom tokenizer reports 163842 entries, but the checkpoint
    embedding and output layers contain 163840 rows. The two trailing IDs are
    reserved and are not emitted by the chat tokenizer. Megatron-Bridge v0.3.1
    rejects this known pair before checkpoint loading, so preserve the
    checkpoint vocabulary and relax only that exact validation case.
    """
    if model_vocab_size != 163840:
        return

    if setup_module is None:
        import megatron.bridge.training.setup as setup_module

    original = getattr(
        setup_module,
        "_rpb_original_validate_and_set_vocab_size",
        setup_module._validate_and_set_vocab_size,
    )
    setup_module._rpb_original_validate_and_set_vocab_size = original

    def validate_vocab_size(
        model_vocab_size: int | None, tokenizer_vocab_size: int
    ) -> tuple[int, bool]:
        if model_vocab_size == 163840 and tokenizer_vocab_size == 163842:
            LOGGER.warning(
                "Allowing Moonlight tokenizer size 163842 with checkpoint "
                "vocabulary size 163840; the final two tokenizer IDs are reserved"
            )
            return model_vocab_size, False
        return original(model_vocab_size, tokenizer_vocab_size)

    setup_module._validate_and_set_vocab_size = validate_vocab_size


def build_bridge_config(raw: dict[str, Any]):
    import torch
    from megatron.bridge import AutoBridge
    from megatron.bridge.recipes.common import _sft_common
    from megatron.bridge.training.config import FinetuningDatasetConfig
    from megatron.bridge.training.rpb import RPBConfig, enable_rpb

    model = raw["model"]
    data = raw["data"]
    train = raw["train"]
    optimizer = raw["optimizer"]
    checkpoint = raw["checkpoint"]
    rpb = raw["rpb"]

    cfg = _sft_common()
    cfg.model = AutoBridge.from_hf_pretrained(
        model["id"], trust_remote_code=bool(model.get("trust_remote_code", False))
    ).to_megatron_provider(load_weights=False)
    _allow_moonlight_tokenizer_len_mismatch(getattr(cfg.model, "vocab_size", None))
    configure_router(cfg.model, model)
    if getattr(cfg.model, "q_lora_rank", None) is not None:
        # Moonlight's HF config deliberately disables Q-LoRA compression.
        cfg.model.q_lora_rank = None

    cfg.model.tensor_model_parallel_size = int(model["tensor_parallel"])
    cfg.model.pipeline_model_parallel_size = int(model["pipeline_parallel"])
    cfg.model.context_parallel_size = int(model["context_parallel"])
    cfg.model.expert_model_parallel_size = int(model["expert_parallel"])
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = cfg.model.tensor_model_parallel_size > 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.seq_length = int(data["seq_length"])
    cfg.model.mtp_num_layers = None
    cfg.model.recompute_granularity = model["recompute_granularity"]
    cfg.model.recompute_modules = list(model["recompute_modules"])

    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = model["id"]
    cfg.tokenizer.hf_tokenizer_kwargs = {
        "trust_remote_code": bool(model.get("trust_remote_code", False))
    }
    cfg.dataset = FinetuningDatasetConfig(
        dataset_root=Path(data["root"]),
        seq_length=int(data["seq_length"]),
        seed=int(data.get("seed", 1234)),
        memmap_workers=int(data.get("memmap_workers", 1)),
        packed_sequence_specs=None,
        do_validation=False,
        do_test=False,
        dataset_kwargs={
            "prompt_template": data["prompt_template"],
            "label_key": "output",
            "answer_only_loss": True,
            "truncation_field": "input",
            "truncation_method": "right",
            "add_bos": False,
            "add_eos": True,
            "global_sample_mapping": False,
            "pad_to_max_length": False,
        },
    )

    cfg.train.train_iters = int(train["iterations"])
    cfg.train.global_batch_size = int(train["global_batch_size"])
    cfg.train.micro_batch_size = int(train["micro_batch_size"])
    cfg.rng.seed = int(train["seed"])
    cfg.optimizer.lr = float(optimizer["max_lr"])
    cfg.optimizer.weight_decay = float(optimizer["weight_decay"])
    cfg.optimizer.adam_beta1 = float(optimizer["adam_beta1"])
    cfg.optimizer.adam_beta2 = float(optimizer["adam_beta2"])
    cfg.optimizer.adam_eps = float(optimizer["adam_eps"])
    cfg.optimizer.clip_grad = float(optimizer["clip_grad"])
    cfg.scheduler.lr_warmup_iters = int(optimizer["warmup_iterations"])
    cfg.scheduler.lr_decay_iters = cfg.train.train_iters
    cfg.scheduler.lr_decay_style = optimizer["lr_decay_style"]
    cfg.scheduler.min_lr = float(optimizer["min_lr"])

    cfg.checkpoint.pretrained_checkpoint = checkpoint["pretrained_checkpoint"]
    cfg.checkpoint.save = checkpoint["save"]
    cfg.checkpoint.load = checkpoint["save"]
    cfg.checkpoint.save_interval = int(checkpoint["save_interval"])

    return enable_rpb(
        cfg,
        RPBConfig(
            prior_path=rpb["prior_path"],
            beta=float(rpb["beta"]),
            eps=float(rpb["eps"]),
            c=float(rpb["c"]),
            route_only=bool(rpb["route_only"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.vlm_step import forward_step

    finetune(build_bridge_config(load_config(args.config)), forward_step)


if __name__ == "__main__":
    main()
