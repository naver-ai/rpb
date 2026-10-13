# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

from __future__ import annotations

import hashlib
import json
import sys

import pytest
import torch


def test_collector_defaults_to_reported_sequence_length(collector_module, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_rpb_prior.py",
            "--model",
            "example/model",
            "--data-jsonl",
            "train.jsonl",
            "--out-path",
            "prior.pt",
            "--top-k",
            "6",
        ],
    )
    assert collector_module._parse_args().max_seq_len == 8192


def test_router_probabilities_uses_normalized_sigmoid(collector_module):
    logits = torch.tensor([[[0.0, 1.0], [2.0, -1.0]]])
    sigmoid = collector_module.router_probabilities(logits)
    torch.testing.assert_close(sigmoid.sum(dim=-1), torch.ones(1, 2))


def test_aggregate_qa_prior_excludes_structural_tokens(collector_module):
    probabilities = torch.tensor(
        [
            [[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]],
            [[0.8, 0.2], [0.6, 0.4], [0.4, 0.6]],
        ]
    )
    prior = collector_module.aggregate_qa_prior(probabilities, span_ids=[-1, 0, 2])
    assert prior.shape == (2, 2, 2)
    torch.testing.assert_close(prior[0], probabilities[:, 1])
    torch.testing.assert_close(prior[1], probabilities[:, 2])


@pytest.mark.parametrize("span_ids", [[-1, -1, 2], [0, -1, -1]])
def test_aggregate_qa_prior_rejects_empty_spans(collector_module, span_ids):
    probabilities = torch.full((2, 3, 4), 0.25)
    with pytest.raises(ValueError, match="non-empty"):
        collector_module.aggregate_qa_prior(probabilities, span_ids=span_ids)


def test_data_sha256_is_path_independent_and_ordered(collector_module, tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    for directory in (first_dir, second_dir):
        (directory / "a.jsonl").write_bytes(b'{"input":"q","output":"a"}\n')
        (directory / "b.jsonl").write_bytes(b'{"input":"x","output":"y"}\n')
    digest_one = collector_module.data_sha256([first_dir / "a.jsonl", first_dir / "b.jsonl"])
    digest_two = collector_module.data_sha256([second_dir / "a.jsonl", second_dir / "b.jsonl"])
    digest_reversed = collector_module.data_sha256([first_dir / "b.jsonl", first_dir / "a.jsonl"])
    assert digest_one == digest_two
    assert digest_one != digest_reversed
    assert len(digest_one) == hashlib.sha256().digest_size * 2


def test_gate_logits_collector_replays_moe_gate_projection(collector_module):
    class MoEGate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))

        def forward(self, hidden_states):
            return hidden_states

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = MoEGate()
            self.second = MoEGate()

        def forward(self, hidden_states):
            return self.second(self.first(hidden_states))

    model = FakeModel()
    hidden = torch.tensor([[[1.0, 3.0], [2.0, 4.0]]])
    with collector_module.GateLogitsCollector(model) as capture:
        model(hidden)
        logits = capture.pop()
    assert logits.shape == (2, 2, 2)
    expected = torch.tensor([[1.0, 6.0], [2.0, 8.0]])
    torch.testing.assert_close(logits[0], expected)
    torch.testing.assert_close(logits[1], expected)
    assert not model.first._forward_pre_hooks
    assert not model.second._forward_pre_hooks


def test_make_payload_has_public_schema_without_source_text(collector_module):
    priors = torch.full((2, 2, 3, 4), 0.25)
    payload = collector_module.make_payload(
        priors,
        model_id="example/model",
        tokenizer_id="example/tokenizer",
        top_k=2,
        max_seq_len=128,
        source_hash="abc123",
    )
    assert payload["prior_qa"].dtype == torch.float16
    assert payload["metadata"]["schema"] == "rpb_prior_v2"
    assert payload["metadata"]["shape"] == [2, 2, 3, 4]
    assert payload["metadata"]["n_missing_uniform_fill"] == 0
    assert payload["metadata"]["family"] == "moonlight"
    assert payload["metadata"]["router_score_function"] == "sigmoid"
    assert payload["metadata"]["prior_score_function"] == "sigmoid"
    metadata_json = json.dumps(payload["metadata"])
    assert "input" not in metadata_json
    assert "output" not in metadata_json


def test_encode_sample_concatenates_once_without_special_tokens_and_detects_moonlight_bodies(
    collector_module,
):
    class Tokenizer:
        unk_token_id = -999
        encoded: list[tuple[str, bool]] = []
        token_ids = {
            "<|im_user|>": 10,
            "<|im_assistant|>": 11,
            "<|im_system|>": 12,
            "<|im_middle|>": 13,
            "<|im_end|>": 14,
        }

        def encode(self, text, add_special_tokens=True):
            self.encoded.append((text, add_special_tokens))
            assert text == " Q  A "
            return [10, 100, 13, 20, 21, 14, 11, 101, 13, 30, 31, 14]

        def convert_tokens_to_ids(self, token):
            return self.token_ids[token]

    tokenizer = Tokenizer()
    tokens, span_ids = collector_module._encode_sample(
        tokenizer,
        {"input": " Q ", "output": " A "},
        max_seq_len=32,
    )

    assert tokenizer.encoded == [(" Q  A ", False)]
    assert tokens == [10, 100, 13, 20, 21, 14, 11, 101, 13, 30, 31, 14]
    assert span_ids == [-1, -1, -1, 0, 0, -1, -1, -1, -1, 2, 2, -1]


def test_collector_rejects_non_moonlight_router(collector_module):
    good = type("Model", (), {"config": type("Config", (), {"scoring_func": "sigmoid"})()})()
    collector_module.validate_moonlight_router(good)
    bad = type("Model", (), {"config": type("Config", (), {"scoring_func": "softmax"})()})()
    with pytest.raises(ValueError, match="requires a sigmoid router"):
        collector_module.validate_moonlight_router(bad)


def test_validate_prior_accepts_valid_artifact_and_checks_data_hash(
    collector_module, validator_module, tmp_path
):
    data = tmp_path / "train.jsonl"
    data.write_text('{"input":"q","output":"a"}\n', encoding="utf-8")
    priors = torch.full((1, 2, 2, 4), 0.25)
    payload = collector_module.make_payload(
        priors,
        model_id="example/model",
        tokenizer_id="example/model",
        top_k=2,
        max_seq_len=32,
        source_hash=collector_module.data_sha256([data]),
    )
    artifact = tmp_path / "prior.pt"
    torch.save(payload, artifact)
    summary = validator_module.validate_prior(
        artifact,
        data_files=[data],
        expected_model="example/model",
        expected_layers=2,
        expected_experts=4,
    )
    assert summary["shape"] == [1, 2, 2, 4]
    assert summary["data_sha256"] == payload["metadata"]["data_sha256"]
    assert summary["router_score_function"] == "sigmoid"
    assert summary["prior_score_function"] == "sigmoid"
    data.write_text('{"input":"changed","output":"a"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="data SHA-256"):
        validator_module.validate_prior(artifact, data_files=[data])


def test_validate_prior_rejects_invalid_probabilities(collector_module, validator_module, tmp_path):
    priors = torch.full((1, 2, 2, 4), 0.25)
    payload = collector_module.make_payload(
        priors,
        model_id="example/model",
        tokenizer_id="example/model",
        top_k=2,
        max_seq_len=32,
        source_hash="abc123",
    )
    payload["prior_qa"][0, 0, 0, 0] = float("nan")
    artifact = tmp_path / "prior.pt"
    torch.save(payload, artifact)
    with pytest.raises(ValueError, match="finite"):
        validator_module.validate_prior(artifact)
