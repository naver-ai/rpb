# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_build_token_bias_uses_q_and_a_and_leaves_other_tokens_zero(
    rpb_module, tmp_path, prior_payload
):
    path = tmp_path / "prior.pt"
    prior_values = torch.tensor([[[[0.8, 0.2]], [[0.1, 0.9]]]], dtype=torch.float16)
    torch.save(prior_payload(prior_values), path)
    prior = rpb_module.RPBPrior(path)
    context = rpb_module.RPBBatchContext(
        sample_ids=torch.tensor([0]),
        span_ids=torch.tensor([[0, 1, 2, -1]]),
    )
    router = SimpleNamespace(config=SimpleNamespace(sequence_parallel=False))

    bias = rpb_module.build_token_bias(
        prior,
        context,
        layer_index=0,
        seq_len=4,
        router=router,
        eps=0.05,
        c=5.0,
        beta=2.0,
    )

    assert bias.shape == (4, 1, 2)
    assert bias[0, 0, 0] > bias[0, 0, 1]
    torch.testing.assert_close(bias[1], torch.zeros_like(bias[1]))
    assert bias[2, 0, 1] > bias[2, 0, 0]
    torch.testing.assert_close(bias[3], torch.zeros_like(bias[3]))


def test_route_only_sigmoid_normalizes_base_scores_on_biased_selection(rpb_module):
    logits = torch.tensor([[2.0, 0.0, -2.0]])
    bias = torch.tensor([[-10.0, 0.0, 10.0]])

    probs, routing_map = rpb_module.route_only_topk(
        logits,
        bias,
        topk=2,
    )

    torch.testing.assert_close(routing_map, torch.tensor([[False, True, True]]))
    base = torch.sigmoid(torch.tensor([0.0, -2.0]))
    expected = torch.tensor([[0.0, base[0] / base.sum(), base[1] / base.sum()]])
    torch.testing.assert_close(probs, expected, rtol=1e-5, atol=1e-6)


def test_group_limited_selection_keeps_only_highest_scoring_group(rpb_module):
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    bias = torch.tensor([[-10.0, -10.0, 0.0, 0.0]])

    _, routing_map = rpb_module.route_only_topk(
        logits,
        bias,
        topk=2,
        num_groups=2,
        group_topk=1,
    )

    torch.testing.assert_close(routing_map, torch.tensor([[False, False, True, True]]))


class FakeTopKRouter(torch.nn.Module):
    def __init__(self, *, training: bool = True, pre_softmax: bool = False):
        super().__init__()
        self.config = SimpleNamespace(
            num_moe_experts=3,
            sequence_parallel=False,
            moe_router_pre_softmax=pre_softmax,
            moe_router_num_groups=None,
            moe_router_group_topk=None,
            moe_router_topk_scaling_factor=None,
            moe_expert_capacity_factor=None,
        )
        self.topk = 1
        self.score_function = "sigmoid"
        self.expert_bias = None
        self.enable_expert_bias = False
        self.original_inputs = []
        self.train(training)

    def is_aux_loss_enabled(self):
        return False

    def apply_z_loss(self, logits, padding_mask=None):
        return logits

    def routing(self, logits, padding_mask=None):
        self.original_inputs.append(logits.clone())
        flat = logits.reshape(-1, logits.shape[-1])
        indices = flat.argmax(dim=-1, keepdim=True)
        probs = torch.zeros_like(flat).scatter(1, indices, 1.0)
        return probs, probs.bool()


class FakeModel(torch.nn.Module):
    def __init__(self, routers):
        super().__init__()
        self.routers = torch.nn.ModuleList(routers)


def test_installed_router_applies_rpb_only_while_training(rpb_module, tmp_path, prior_payload):
    path = tmp_path / "prior.pt"
    torch.save(prior_payload(torch.tensor([[[[0.01, 0.01, 0.98]], [[0.01, 0.01, 0.98]]]])), path)
    prior = rpb_module.RPBPrior(path)
    router = FakeTopKRouter(training=True)
    model = FakeModel([router])
    rpb_module.set_batch_context(
        rpb_module.RPBBatchContext(sample_ids=torch.tensor([0]), span_ids=torch.tensor([[0]]))
    )
    rpb_module.install_rpb_on_model([model], rpb_module.RPBConfig(str(path), beta=2.0), prior=prior)
    logits = torch.tensor([[[5.0, 0.0, 0.0]]])

    _, train_map = router.routing(logits)
    router.eval()
    _, eval_map = router.routing(logits)

    torch.testing.assert_close(train_map, torch.tensor([[False, False, True]]))
    torch.testing.assert_close(eval_map, torch.tensor([[True, False, False]]))


def test_installed_router_uses_rpb_during_train_mode_no_grad(rpb_module, tmp_path, prior_payload):
    path = tmp_path / "prior.pt"
    values = torch.tensor([[[[0.01, 0.01, 0.98]], [[0.01, 0.01, 0.98]]]])
    torch.save(prior_payload(values), path)
    prior = rpb_module.RPBPrior(path)
    router = FakeTopKRouter(training=True)
    rpb_module.set_batch_context(
        rpb_module.RPBBatchContext(sample_ids=torch.tensor([0]), span_ids=torch.tensor([[0]]))
    )
    rpb_module.install_rpb_on_model(
        [FakeModel([router])], rpb_module.RPBConfig(str(path), beta=2.0), prior=prior
    )
    with torch.no_grad():
        _, routing_map = router.routing(torch.tensor([[[5.0, 0.0, 0.0]]]))
    torch.testing.assert_close(routing_map, torch.tensor([[False, False, True]]))


def test_installed_sigmoid_router_accepts_moonlight_pre_softmax_flag(
    rpb_module, tmp_path, prior_payload
):
    path = tmp_path / "prior.pt"
    values = torch.tensor([[[[0.01, 0.01, 0.98]], [[0.01, 0.01, 0.98]]]])
    torch.save(prior_payload(values), path)
    prior = rpb_module.RPBPrior(path)
    router = FakeTopKRouter(pre_softmax=True)
    rpb_module.set_batch_context(
        rpb_module.RPBBatchContext(sample_ids=torch.tensor([0]), span_ids=torch.tensor([[0]]))
    )
    rpb_module.install_rpb_on_model(
        [FakeModel([router])], rpb_module.RPBConfig(str(path), beta=2.0), prior=prior
    )

    _, routing_map = router.routing(torch.tensor([[[5.0, 0.0, 0.0]]]))

    torch.testing.assert_close(routing_map, torch.tensor([[False, False, True]]))


def test_install_rejects_prior_router_layer_mismatch(rpb_module, tmp_path, prior_payload):
    path = tmp_path / "prior.pt"
    torch.save(prior_payload(torch.full((1, 2, 2, 3), 1 / 3)), path)
    prior = rpb_module.RPBPrior(path)
    model = FakeModel([FakeTopKRouter()])

    with pytest.raises(ValueError, match="layer count"):
        rpb_module.install_rpb_on_model([model], rpb_module.RPBConfig(str(path)), prior=prior)


def test_installed_router_requires_batch_context(rpb_module, tmp_path, prior_payload):
    path = tmp_path / "prior.pt"
    torch.save(prior_payload(torch.full((1, 2, 1, 3), 1 / 3)), path)
    prior = rpb_module.RPBPrior(path)
    router = FakeTopKRouter()
    rpb_module.set_batch_context(None)
    rpb_module.install_rpb_on_model(
        [FakeModel([router])], rpb_module.RPBConfig(str(path)), prior=prior
    )

    with pytest.raises(RuntimeError, match="batch context"):
        router.routing(torch.zeros(1, 1, 3))
