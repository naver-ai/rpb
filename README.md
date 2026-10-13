# Router Prior Bias (RPB)

Code for **Router Prior Bias: Preserving Base Routing Structure in MoE
Post-Training**. RPB measures the routing preferences of a frozen base model
and uses them as a training-time bias for expert selection. The current router
still determines the dispatch weights, and the bias is removed for evaluation.

This repository provides a public Moonlight-16B-A3B implementation of RPB:

- prior collection and validation tools;
- a patch for a pinned Megatron-Bridge revision;
- the paper-scale Moonlight training configuration; and
- CPU tests for the method and integration contracts.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for release scope, implementation
notes, and validation coverage. Data and model artifacts are supplied separately
by the user.

## Repository layout

```text
src/rpb_release/       Prior collection, span detection, scoring, and artifacts
tools/                 Prior collection and validation commands
examples/              Moonlight training entry point and configuration
patches/source/rpb.py  Megatron-Bridge integration source
patches/*.patch        Generated patch for the pinned upstream revision
scripts/               Patch and release validation commands
tests/                 CPU unit and integration tests
```

`patches/source/rpb.py` is the source of truth for the integration patch. After
editing it, regenerate the patch with `uv run python scripts/sync_patch.py`.

## Supported configuration

- Moonlight-16B-A3B
- Megatron-Bridge `v0.3.1` at commit
  `9c9dd848966322fc3ed7706747ec13219ef49dda`
- Megatron-LM submodule at commit
  `55ac7082517c3878ae653c07c09c534b8aed49f6`
- single-turn Q/A examples without sequence packing
- dropless sigmoid routing with top-k 6 over 64 experts
- tensor parallelism 2, expert parallelism 8, pipeline parallelism 1, and
  context parallelism 1
- route-only RPB with `beta=1`, `eps=0.05`, and `c=5`

The example launcher disables auxiliary router loss, token dropping, MTP, and
native expert-bias updates. Moonlight's pretrained expert-bias tensor remains
enabled and frozen.

## Installation

Install the release utilities from this repository. This environment is used
for data rendering, prior collection, validation, and the local release tests:

```bash
uv sync --extra prior --extra test
```

Clone and pin Megatron-Bridge separately, then apply the supplied patch:

```bash
git clone --recurse-submodules \
  https://github.com/NVIDIA-NeMo/Megatron-Bridge.git \
  /path/to/Megatron-Bridge

git -C /path/to/Megatron-Bridge checkout \
  9c9dd848966322fc3ed7706747ec13219ef49dda
git -C /path/to/Megatron-Bridge submodule update --init --recursive

/path/to/rpb-code-release/scripts/check_patch.sh /path/to/Megatron-Bridge
git -C /path/to/Megatron-Bridge apply \
  /path/to/rpb-code-release/patches/rpb-megatron-bridge-v0.3.1.patch
```

Install the pinned Megatron-Bridge checkout using its upstream instructions.
After its `.venv` has been created, install this repository into that same
environment:

```bash
uv pip install \
  --python /path/to/Megatron-Bridge/.venv/bin/python \
  --editable '/path/to/rpb-code-release[prior]'
```

Use `/path/to/Megatron-Bridge/.venv/bin/python` and the executables from the
same `.venv` for checkpoint conversion and training. Do not launch training
with a `torchrun` from an unrelated environment. `check_patch.sh` verifies both
upstream revisions and also accepts an already-applied patch.

## Data format

This release supports single-turn Q/A rows only. Each JSONL row must contain
non-empty `input` and `output` strings rendered with the Moonlight chat
template:

```json
{"input":"<user turn><assistant prefix>","output":"<assistant body><turn end>"}
```

For example, the following converts a raw JSONL file with `question` and
`answer` fields. Using the tokenizer instead of assembling marker strings by
hand keeps the data consistent with the selected Moonlight revision:

```bash
uv run --extra prior python - <<'PY'
import json
from pathlib import Path

from transformers import AutoTokenizer

source = Path("data/raw/training.jsonl")
destination = Path("data/moonlight/training.jsonl")
tokenizer = AutoTokenizer.from_pretrained(
    "moonshotai/Moonlight-16B-A3B",
    trust_remote_code=True,
)

destination.parent.mkdir(parents=True, exist_ok=True)
with source.open(encoding="utf-8") as src, destination.open("w", encoding="utf-8") as dst:
    for line_number, line in enumerate(src, start=1):
        row = json.loads(line)
        question = row["question"].strip()
        answer = row["answer"].strip()
        if not question or not answer:
            raise ValueError(f"blank question or answer on line {line_number}")

        user_message = [{"role": "user", "content": question}]
        conversation = user_message + [{"role": "assistant", "content": answer}]
        prefix = tokenizer.apply_chat_template(
            user_message,
            tokenize=False,
            add_generation_prompt=True,
        )
        rendered = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not rendered.startswith(prefix):
            raise ValueError(f"chat-template prefix mismatch on line {line_number}")
        dst.write(json.dumps(
            {"input": prefix, "output": rendered[len(prefix):]},
            ensure_ascii=False,
        ) + "\n")
PY
```

`input` ends immediately before the assistant body. Do not put earlier
assistant turns in `input`. The collector and trainer must use the same ordered
file; rows must not be filtered, skipped, or reordered between the two stages.

Both paths concatenate `input + output` and tokenize once with
`add_special_tokens=False`. Q/A bodies are detected from Moonlight's chat
markers, structural tokens receive no RPB bias, and the sequence is truncated
on the right to 8,192 routed positions. The collector rejects blank, malformed,
marker-free, and empty-span rows.

## Collect and validate the prior

```bash
uv run python tools/collect_rpb_prior.py \
  --model moonshotai/Moonlight-16B-A3B \
  --data-jsonl data/moonlight/training.jsonl \
  --out-path artifacts/moonlight-rpb-prior.pt \
  --top-k 6 \
  --max-seq-len 8192 \
  --trust-remote-code
```

The output uses schema `rpb_prior_v2`. Its `prior_qa` tensor has shape
`[samples, 2, MoE layers, experts]`; metadata records the model, tokenizer,
router transform, top-k, maximum sequence length, and source-data fingerprint.
Prompts and answers are not stored in the artifact.

Validate the tensor and its correspondence to the training file:

```bash
uv run python tools/validate_rpb_prior.py \
  artifacts/moonlight-rpb-prior.pt \
  --data-jsonl data/moonlight/training.jsonl \
  --expected-model moonshotai/Moonlight-16B-A3B \
  --expected-layers 26 \
  --expected-experts 64
```

## Train

Convert the Hugging Face checkpoint with the converter from the pinned and
patched Megatron-Bridge checkout:

```bash
/path/to/Megatron-Bridge/.venv/bin/python \
  /path/to/Megatron-Bridge/examples/conversion/convert_checkpoints.py import \
  --hf-model moonshotai/Moonlight-16B-A3B \
  --megatron-path checkpoints/moonlight-16b-a3b \
  --torch-dtype bfloat16 \
  --device-map auto \
  --trust-remote-code
```

In `examples/moonlight/config.yaml`, set:

- `checkpoint.pretrained_checkpoint` to the base checkpoint;
- `data.root` to a directory containing `training.jsonl`; and
- `rpb.prior_path` to the validated prior artifact.

Megatron-Bridge reads the JSONL through a memory-mapped index. If all workers
see the same shared filesystem, create the index once before launching. If the
dataset is copied to node-local storage, run this command once on every node
after the copy and before `torchrun`:

```bash
/path/to/Megatron-Bridge/.venv/bin/python - <<'PY'
from megatron.bridge.data.datasets.utils import build_index_files

build_index_files(
    ["/absolute/path/to/data/moonlight/training.jsonl"],
    newline_int=10,
    workers=1,
)
PY
```

This creates `training.jsonl.idx.npy` and `training.jsonl.idx.info` beside the
JSONL file. The data file and both index files must be readable from every
training process.

Launch the reported 16-node, 128-GPU topology:

```bash
/path/to/Megatron-Bridge/.venv/bin/torchrun --nnodes=16 --nproc-per-node=8 \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  /path/to/rpb-code-release/examples/train.py \
  --config /path/to/rpb-code-release/examples/moonlight/config.yaml
```

The configuration uses TP=2, EP=8, PP=1, CP=1, expert-TP=1, global batch 128,
micro-batch 1, sequence length 8,192, and 782 optimizer steps. At world size
128, ordinary data parallelism is DP=64 and expert data parallelism is 16.
The default seed is 42; to run the three main seeds, set both `data.seed` and
`train.seed` to 42, 43, and 44 in separate runs.

See [training configuration notes](REPRODUCIBILITY.md) for implementation
differences and effective weight decay.

The trainer verifies the data fingerprint, row count, model and tokenizer IDs,
router transform, top-k, sequence length, layer count, and expert count before
installing RPB.

## Method behavior

For expert prior `p`, RPB clamps each value to `[eps/E, c/E]`, renormalizes over
experts, and creates a zero-mean log bias. During training, `beta` times this
bias affects top-k selection only. Dispatch weights come from the current
router without the RPB bias. Structural and padding tokens remain unbiased.

RPB is active whenever the router is in training mode, including activation
recomputation. In evaluation mode the original router is used without RPB.

## Validation

Run the local release checks with:

```bash
bash scripts/validate_release.sh
```

This runs the CPU tests, lint and formatting checks, Python compilation, shell
syntax checks, patch synchronization, and the public-content scan. See
[validation scope](REPRODUCIBILITY.md#validation-scope).

## Citation

```bibtex
@inproceedings{lee2026rpb,
  title     = {Router Prior Bias: Preserving Base Routing Structure in {MoE} Post-Training},
  author    = {Lee, Jaedeok and Kim, Keonwoo and Han, Dongyoon and Yun, Sangdoo and Choi, Yera and Yoo, Haanju},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

## License

Router Prior Bias (RPB)
Copyright (c) 2026-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use the original RPB code except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Adapted third-party code retains its upstream license:

| Source | License |
|--------|---------|
| NVIDIA Megatron-LM | BSD 3-Clause |
| NVIDIA Megatron-Bridge | Apache 2.0 |

These licenses apply to the corresponding third-party portions. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for covered files,
attribution, and full license terms.
