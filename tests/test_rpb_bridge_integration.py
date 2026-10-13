# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch


class FakeDataset:
    def __init__(self):
        self.samples_mapping = torch.tensor([[1, 0, 0], [0, 0, 0]])

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        mapped = int(self.samples_mapping[int(idx)][0])
        examples = [
            {"input_ids": [10, 11, 20, 21, 2], "answer_start_idx": 2},
            {"input_ids": [30, 31, 32, 40, 2], "answer_start_idx": 3},
        ]
        return examples[mapped]

    def collate_fn(self, batch):
        tokens = torch.tensor([item["input_ids"][:-1] for item in batch])
        return {"tokens": tokens}


def test_dataset_patch_preserves_mapped_sample_id_and_qa_spans(rpb_module):
    rpb_module.patch_sft_dataset_class(FakeDataset)
    dataset = FakeDataset()

    item = dataset[0]
    batch = dataset.collate_fn([item])

    assert item["rpb_sample_id"] == 1
    torch.testing.assert_close(item["rpb_span_ids"], torch.tensor([0, 0, 0, 2]))
    torch.testing.assert_close(batch["rpb_sample_ids"], torch.tensor([1]))
    torch.testing.assert_close(batch["rpb_span_ids"], torch.tensor([[0, 0, 0, 2]]))


def test_dataset_patch_is_idempotent(rpb_module):
    original_getitem = FakeDataset.__getitem__
    rpb_module.patch_sft_dataset_class(FakeDataset)
    assert FakeDataset.__getitem__ is original_getitem


def test_dataset_patch_pads_span_ids_to_collated_token_length(rpb_module):
    class PaddedDataset(FakeDataset):
        def collate_fn(self, batch):
            return {"tokens": torch.zeros(len(batch), 8, dtype=torch.long)}

    rpb_module.patch_sft_dataset_class(PaddedDataset)
    dataset = PaddedDataset()

    batch = dataset.collate_fn([dataset[1]])

    torch.testing.assert_close(batch["rpb_span_ids"], torch.tensor([[0, 0, 2, 2, -1, -1, -1, -1]]))


def test_dataset_patch_recreates_paper_token_stream_and_semantic_spans(rpb_module):
    class Tokenizer:
        unk_token_id = -999
        eos_token_id = 99
        token_ids = {
            "<|im_user|>": 10,
            "<|im_assistant|>": 11,
            "<|im_system|>": 12,
            "<|im_middle|>": 13,
            "<|im_end|>": 14,
        }

        def encode(self, text, add_special_tokens=True):
            assert text == " Q  A "
            assert add_special_tokens is False
            return [10, 100, 13, 20, 21, 14, 11, 101, 13, 30, 31, 14]

        def convert_tokens_to_ids(self, token):
            return self.token_ids[token]

    class Wrapper:
        eos_id = 99

        def __init__(self):
            # Megatron-LM's HuggingFaceTokenizer exposes the underlying
            # transformers tokenizer through this public attribute.
            self.tokenizer = SimpleNamespace(tokenizer=Tokenizer())

    class PaperDataset:
        label_key = "output"
        max_seq_length = 32
        add_eos = True

        def __init__(self):
            self.tokenizer = Wrapper()
            self.samples_mapping = None
            self.examples = [{"input": " Q ", "output": " A "}]

        def __len__(self):
            return len(self.examples)

        def _process_example(self, example):
            raise AssertionError("the generic field-wise processor must be replaced")

        def __getitem__(self, idx):
            return self._process_example(self.examples[idx])

        def collate_fn(self, batch):
            return {"tokens": torch.tensor([item["input_ids"][:-1] for item in batch])}

    rpb_module.patch_sft_dataset_class(PaperDataset)
    item = PaperDataset()[0]

    assert item["input_ids"] == [10, 100, 13, 20, 21, 14, 11, 101, 13, 30, 31, 14, 99]
    assert item["answer_start_idx"] == 9
    torch.testing.assert_close(
        item["rpb_span_ids"],
        torch.tensor([-1, -1, -1, 0, 0, -1, -1, -1, -1, 2, 2, -1]),
    )


def test_hf_tokenizer_unwrap_supports_bridge_and_legacy_wrappers(rpb_module):
    inner = object()

    bridge = SimpleNamespace(tokenizer=SimpleNamespace(tokenizer=inner))
    legacy = SimpleNamespace(_tokenizer=SimpleNamespace(_tokenizer=inner))

    assert rpb_module._unwrap_hf_tokenizer(bridge) is inner
    assert rpb_module._unwrap_hf_tokenizer(legacy) is inner


def test_vlm_patch_publishes_context_and_does_not_leak_it(rpb_module):
    module = SimpleNamespace()

    def get_batch_from_iterator(data_iterator, *args, **kwargs):
        raw = next(data_iterator)
        return {"tokens": raw["tokens"]}

    def get_batch(data_iterator, *args, **kwargs):
        batch = module.get_batch_from_iterator(data_iterator)
        return (batch["tokens"], None)

    module.get_batch_from_iterator = get_batch_from_iterator
    module.get_batch = get_batch
    rpb_module.patch_vlm_step_module(module)

    tokens = torch.ones(1, 3, dtype=torch.long)
    module.get_batch(
        iter(
            [
                {
                    "tokens": tokens,
                    "rpb_sample_ids": torch.tensor([7]),
                    "rpb_span_ids": torch.tensor([[0, 0, 2]]),
                }
            ]
        )
    )
    context = rpb_module.get_batch_context()
    assert context is not None
    torch.testing.assert_close(context.sample_ids, torch.tensor([7]))
    torch.testing.assert_close(context.span_ids, torch.tensor([[0, 0, 2]]))

    module.get_batch(iter([{"tokens": tokens}]))
    assert rpb_module.get_batch_context() is None


def test_batch_context_is_visible_to_activation_recompute_thread(rpb_module):
    context = rpb_module.RPBBatchContext(
        sample_ids=torch.tensor([3]), span_ids=torch.tensor([[0, 2]])
    )
    rpb_module.set_batch_context(context)
    observed = []

    thread = threading.Thread(target=lambda: observed.append(rpb_module.get_batch_context()))
    thread.start()
    thread.join()

    assert len(observed) == 1
    assert observed[0] is context


def test_enable_rpb_rejects_unsupported_bridge_configuration(rpb_module, tmp_path):
    path = tmp_path / "prior.pt"
    torch.save({"prior_qa": torch.full((1, 2, 1, 3), 1 / 3)}, path)

    def config(
        *,
        packing=False,
        pp=1,
        cp=1,
        aux="none",
        expert_bias=False,
        expert_bias_update_rate=0.0,
    ):
        model = SimpleNamespace(
            pipeline_model_parallel_size=pp,
            context_parallel_size=cp,
            moe_router_load_balancing_type=aux,
            moe_aux_loss_coeff=0.0,
            moe_router_enable_expert_bias=expert_bias,
            moe_router_bias_update_rate=expert_bias_update_rate,
            register_pre_wrap_hook=lambda hook: None,
        )
        return SimpleNamespace(
            model=model,
            dataset=SimpleNamespace(pack_sequences_in_batch=packing),
        )

    rpb = rpb_module.RPBConfig(str(path))
    with pytest.raises(ValueError, match="packing"):
        rpb_module.validate_bridge_config(config(packing=True), rpb)
    with pytest.raises(ValueError, match="pipeline parallelism"):
        rpb_module.validate_bridge_config(config(pp=2), rpb)
    with pytest.raises(ValueError, match="context parallelism"):
        rpb_module.validate_bridge_config(config(cp=2), rpb)
    with pytest.raises(ValueError, match="auxiliary router loss"):
        rpb_module.validate_bridge_config(config(aux="aux_loss"), rpb)
    rpb_module.validate_bridge_config(config(expert_bias=True), rpb)
    with pytest.raises(ValueError, match="expert-bias updates"):
        rpb_module.validate_bridge_config(
            config(expert_bias=True, expert_bias_update_rate=1e-3), rpb
        )


def test_training_alignment_checks_data_and_model_metadata(rpb_module, tmp_path, prior_payload):
    data_root = tmp_path / "data"
    data_root.mkdir()
    training = data_root / "training.jsonl"
    training.write_text('{"input":"q","output":"a"}\n', encoding="utf-8")
    values = torch.full((1, 2, 1, 3), 1 / 3)
    path = tmp_path / "prior.pt"
    torch.save(
        prior_payload(
            values,
            data_sha256=rpb_module.data_sha256([training]),
            router_score_function="sigmoid",
            prior_score_function="sigmoid",
            top_k=1,
            max_seq_len=128,
        ),
        path,
    )
    prior = rpb_module.RPBPrior(path)
    config = SimpleNamespace(
        model=SimpleNamespace(
            hf_model_id="example/model",
            num_moe_experts=3,
            moe_router_topk=1,
            moe_router_score_function="sigmoid",
        ),
        tokenizer=SimpleNamespace(tokenizer_model="example/model"),
        dataset=SimpleNamespace(dataset_root=data_root, seq_length=128),
    )

    rpb_module.validate_prior_alignment(config, prior)
    training.write_text('{"input":"changed","output":"a"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="data SHA-256"):
        rpb_module.validate_prior_alignment(config, prior)


def test_training_alignment_rejects_non_moonlight_router_score(rpb_module, tmp_path, prior_payload):
    data_root = tmp_path / "data"
    data_root.mkdir()
    training = data_root / "training.jsonl"
    training.write_text('{"input":"q","output":"a"}\n', encoding="utf-8")
    values = torch.full((1, 2, 1, 3), 1 / 3)
    path = tmp_path / "prior.pt"
    torch.save(
        prior_payload(
            values,
            data_sha256=rpb_module.data_sha256([training]),
            router_score_function="sigmoid",
            prior_score_function="sigmoid",
        ),
        path,
    )
    prior = rpb_module.RPBPrior(path)
    config = SimpleNamespace(
        model=SimpleNamespace(
            hf_model_id="example/model",
            num_moe_experts=3,
            moe_router_topk=1,
            moe_router_score_function="sigmoid",
        ),
        tokenizer=SimpleNamespace(tokenizer_model="example/model"),
        dataset=SimpleNamespace(dataset_root=data_root, seq_length=128),
    )
    rpb_module.validate_prior_alignment(config, prior)

    config.model.moe_router_score_function = "softmax"
    with pytest.raises(ValueError, match="router score function"):
        rpb_module.validate_prior_alignment(config, prior)
