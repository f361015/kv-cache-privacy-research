# KV Cache Privacy Research

> **Current priority (through Monday, August 31, 2026): literature and novelty review.**
> Implementation is paused until the group has mapped the closest work, separated the
> relevant threat models, and identified a defensible research gap. See
> [`docs/LITERATURE_SURVEY_PLAN.md`](docs/LITERATURE_SURVEY_PLAN.md) and
> [`docs/PRELIMINARY_NOVELTY_MAP.md`](docs/PRELIMINARY_NOVELTY_MAP.md).

This repository studies how KV-cache quantization changes privacy leakage relative to language-model utility.

The initial question is deliberately empirical:

> When KV-cache precision is reduced from FP16 to INT8 and INT4, does recoverable private information decrease faster, slower, or at the same rate as inference utility?

Quantization is **not** assumed to be a privacy defense. The project first establishes whether a reproducible phenomenon exists, then uses that evidence to motivate a mathematical improvement.

## Initial scope

- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Precisions: FP16, INT8, INT4
- Data: synthetic public, personal, financial, medical, confidential, and later RAG prompts
- Leakage objectives: exact/token recovery, semantic recovery, sensitive-attribute inference
- Utility objectives: logit divergence, generation agreement, QA accuracy, and long-context retrieval
- Threat model: attacker knows the model, weights, quantizer, and scales, but not the victim prompt or response

## Team workstreams

1. **Person A - Quantization and utility:** KV extraction, controlled quantizers, online quantized-cache inference, and utility measurements.
2. **Person B - Leakage and attacks:** FP16 attack baseline, adaptive INT8/INT4 attacks, and leakage metrics.
3. **Person C - Data, evaluation, and reproducibility:** controlled dataset, manifests, experiment orchestration, statistical analysis, and later information-concentration/multimodal studies.

All workstreams use the same model revision, prompts, seeds, cache schema, and experiment manifest.

## Monday deliverable

The immediate milestone, due Monday, August 31, 2026, is a working pilot pipeline and an initial **privacy-utility-bitrate result** on the shared smoke-test prompts. The three workstreams run in parallel; they are separated by ownership and interfaces, not by week.

See [docs/WORKPLAN.md](docs/WORKPLAN.md) for responsibilities, interfaces, completion gates, and the Monday submission package.

## Repository status

This initial commit contains the research contract and project scaffold. Model download, attack implementations, and experimental results will be added only after the team freezes the threat model and cache schema.

## Safety and data policy

Use synthetic secrets only. Do not commit real personal, medical, financial, organizational, or confidential information. Large model checkpoints, raw KV tensors, and generated experiment caches must stay outside Git and be referenced through local paths or artifact storage.
