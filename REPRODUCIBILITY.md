# Reproducibility scope

This repository releases the Moonlight RPB method and training configuration,
with the implementation notes below.

See the [README](README.md) for the workflow and
[`examples/moonlight/config.yaml`](examples/moonlight/config.yaml) for training
settings. The seed-42 configuration represents 20,000 ordered single-turn
examples over five epochs (782 optimizer steps at global batch 128).

## Compatibility notes

- Moonlight priors use normalized sigmoid scores, rather than the softmax
  described in the paper.
- RPB remains active throughout training, including activation recomputation,
  and is disabled for evaluation. Unlike the research worktree, this release
  also applies RPB during no-gradient training forwards.

## Parallelism

Ordinary DP is `128 / (TP 2 * PP 1 * CP 1) = 64`. Expert DP is
`128 / (expert-TP 1 * EP 8 * PP 1) = 16` for this CP=1 configuration.
These are distinct process groups; EP does not further divide ordinary DP.

## Effective weight decay

The paper and optimizer field specify `0.1`; the inspected research setup and
this release use a constant scheduled value of `0.033` for non-exempt parameters.
Changing the optimizer field alone does not override the scheduler.

## Not distributed

- Data and model artifacts: training corpora, prior tensors, and checkpoints.
- Research workflows: other model families, baseline and ablation launch
  matrices, corpus selection, evaluation, and routing-analysis tools and outputs.
- Cluster operations: deployment manifests and execution logs.

## Validation scope

The local release checks cover CPU tests, static checks, and patch consistency.
They do not establish full distributed-training or benchmark reproduction, or
bitwise equivalence to the research environment.
