# KV-Cache Privacy Research

This repository contains a minimal research skeleton and a controlled diagnostic pilot at the
intersection of KV-cache compression and privacy.

## Current status

A small GPT-2 pilot is active. It tests reconstruction from FP32, INT8, and INT4 cache entries under
both a float-oriented matcher and a matcher that knows and emulates the quantizer. It is a mechanics
check, not evidence of cross-model generality or a claim that quantization provides privacy.

The working question is:

> How does lossy KV-cache compression change information recoverable by an adaptive attacker,
> relative to genuine online model utility and bitrate?

This is a research question, not a claim that quantization provides privacy.

## Documentation

- [Project context](docs/PROJECT_CONTEXT.md) records the idea, boundaries, closest-work risks,
  and decisions that should not be lost.
- [Literature research guide](docs/LITERATURE_RESEARCH_GUIDE.md) defines how to search and judge
  novelty.
- [Paper review template](docs/PAPER_REVIEW_TEMPLATE.md) keeps reviews comparable.
- [Quantized reconstruction pilot](docs/QUANTIZED_RECONSTRUCTION_PILOT.md) explains the current
  experiment, threat model, and meaning of an attacker adapted to quantization.

## Current pilot

`experiments/quantized_collision_pilot.py` implements a deliberately small diagnostic based on
KV-Cloak's first-layer collision idea. It compares FP32, INT8, and INT4 cache targets under both an
unchanged float-space matcher and a matcher that re-encodes every candidate with the known
quantizer. It also measures online cache-feedback utility. The initial GPT-2 run is a smoke test,
not evidence that the result generalizes to current Llama-family models.

## Before expanding development

1. Finish the closest-work review and write a cautious novelty statement.
2. Freeze one threat model; do not mix tensor reconstruction, timing leakage, cache integrity,
   and jailbreak safety.
3. Reproduce an FP16 attack baseline before interpreting compressed-cache results.
4. Evaluate a quantization-aware attacker, not only an attack written for FP16.
5. Measure utility with a compressed cache that online attention actually reads.
6. Add code only after the research question survives these checks.

Use synthetic secrets only. Do not commit model checkpoints, raw KV tensors, credentials, or
sensitive user data.
