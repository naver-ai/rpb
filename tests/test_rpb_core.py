# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_prior_to_log_bias_clamps_normalizes_and_centers(rpb_module):
    prior = torch.tensor([[0.0, 0.10, 0.90]], dtype=torch.float32)

    bias = rpb_module.prior_to_zero_mean_log_bias(prior, eps=0.3, c=1.5)

    expected = torch.tensor([[-0.5364793, -0.5364793, 1.0729586]])
    torch.testing.assert_close(bias, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(bias.mean(dim=-1), torch.zeros(1), atol=1e-6, rtol=0)


def test_prior_rejects_invalid_shape_and_missing_rows(rpb_module, tmp_path):
    bad_shape = tmp_path / "bad-shape.pt"
    torch.save({"prior_qa": torch.ones(2, 3, 4)}, bad_shape)
    with pytest.raises(ValueError, match=r"\[N, 2, L, E\]"):
        rpb_module.RPBPrior(bad_shape)

    missing = tmp_path / "missing.pt"
    torch.save(
        {
            "prior_qa": torch.full((2, 2, 1, 3), 1 / 3),
            "metadata": {"n_missing_uniform_fill": 1},
        },
        missing,
    )
    with pytest.raises(ValueError, match="missing rows"):
        rpb_module.RPBPrior(missing)


def test_prior_selects_requested_sample_span_and_layer(rpb_module, tmp_path, prior_payload):
    values = torch.tensor(
        [
            [[[0.1, 0.2, 0.7]], [[0.6, 0.3, 0.1]]],
            [[[0.2, 0.3, 0.5]], [[0.5, 0.4, 0.1]]],
        ],
        dtype=torch.float16,
    )
    path = tmp_path / "prior.pt"
    torch.save(prior_payload(values), path)
    prior = rpb_module.RPBPrior(path)

    selected = prior.log_bias_for_layer(torch.tensor([1]), 0, eps=0.05, c=5.0)

    assert selected.shape == (1, 2, 3)
    expected_q = rpb_module.prior_to_zero_mean_log_bias(
        torch.tensor([[0.2, 0.3, 0.5]]), eps=0.05, c=5.0
    )
    torch.testing.assert_close(selected[:, 0], expected_q, rtol=2e-3, atol=2e-3)


def test_prior_rejects_out_of_range_sample_and_layer(rpb_module, tmp_path, prior_payload):
    path = tmp_path / "prior.pt"
    torch.save(prior_payload(torch.full((1, 2, 2, 3), 1 / 3)), path)
    prior = rpb_module.RPBPrior(path)

    with pytest.raises(IndexError, match="sample id"):
        prior.log_bias_for_layer(torch.tensor([1]), 0, eps=0.05, c=5.0)
    with pytest.raises(IndexError, match="layer index"):
        prior.log_bias_for_layer(torch.tensor([0]), 2, eps=0.05, c=5.0)


def test_config_rejects_non_positive_bounds_and_beta(rpb_module):
    with pytest.raises(ValueError, match="beta"):
        rpb_module.RPBConfig("prior.pt", beta=-1).validate()
    with pytest.raises(ValueError, match="eps"):
        rpb_module.RPBConfig("prior.pt", eps=0).validate()
    with pytest.raises(ValueError, match="c"):
        rpb_module.RPBConfig("prior.pt", c=0).validate()
    with pytest.raises(ValueError, match="route_only"):
        rpb_module.RPBConfig("prior.pt", route_only=False).validate()


def test_prior_requires_complete_v1_metadata_and_normalized_probabilities(
    rpb_module, tmp_path, prior_payload
):
    values = torch.full((1, 2, 1, 3), 1 / 3)
    missing_metadata = tmp_path / "missing-metadata.pt"
    torch.save({"prior_qa": values, "metadata": {}}, missing_metadata)
    with pytest.raises(ValueError, match="schema"):
        rpb_module.RPBPrior(missing_metadata)

    not_normalized = tmp_path / "not-normalized.pt"
    torch.save(prior_payload(torch.ones_like(values)), not_normalized)
    with pytest.raises(ValueError, match="sum to one"):
        rpb_module.RPBPrior(not_normalized)


def test_select_span_ids_slices_tensor_parallel_partition(rpb_module, monkeypatch):
    monkeypatch.setattr(rpb_module, "_tensor_parallel_rank_and_size", lambda: (1, 2))
    router = SimpleNamespace(config=SimpleNamespace(sequence_parallel=True))
    span_ids = torch.tensor([[0, 0, 2, 2]])

    selected = rpb_module.select_span_ids_for_router_logits(span_ids, 2, router)

    torch.testing.assert_close(selected, torch.tensor([[2, 2]]))


def test_select_span_ids_rejects_unexplained_length_mismatch(rpb_module):
    router = SimpleNamespace(config=SimpleNamespace(sequence_parallel=False))
    with pytest.raises(RuntimeError, match="span_ids length"):
        rpb_module.select_span_ids_for_router_logits(torch.zeros(1, 4), 2, router)
