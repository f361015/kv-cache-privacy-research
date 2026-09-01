# KV-Cache Privacy Research

This repository contains a minimal research skeleton and controlled diagnostics at the intersection
of KV-cache compression and privacy.

## Current status

A Llama-3.2-1B experiment now uses the real HQQ-backed
`transformers.cache_utils.QuantizedCache`. It compares a BF16 `DynamicCache` with packed INT8 and
INT4 targets under both a float-oriented matcher and a matcher that knows and emulates the
quantizer. The earlier GPT-2 implementation remains a mechanics smoke test.

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
- [Quantized reconstruction pilot](docs/QUANTIZED_RECONSTRUCTION_PILOT.md) explains the GPT-2
  smoke test and the meaning of an attacker adapted to quantization.
- [Llama Transformers QuantizedCache experiment](docs/LLAMA_TRANSFORMERS_QUANTIZED_CACHE.md)
  records the exact checkpoint, real cache behavior, attack/utility metrics, and current result.

## Current experiment

`experiments/transformers_quantized_cache_experiment.py` implements a deliberately small diagnostic
based on KV-Cloak's first-layer Collision signal. It scans the complete Llama vocabulary, uses
actual packed `QuantizedCache` targets, and makes online attention read the compressed cache for
continuation and MMLU-format utility checks. The completed six-prompt run recovered every synthetic
secret in BF16, INT8, and INT4. This is a scoped result, not a claim that it generalizes to other
models, attacks, prompts, or production quantizers.

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
