# KV-Cache Privacy Research

This repository is a minimal documentation skeleton for a possible research project at the
intersection of KV-cache compression and privacy.

## Current status

No implementation or experiment is active. The current work is literature research: establish
what has already been done, identify the closest papers, and decide whether a defensible research
gap remains.

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

## If development resumes

1. Finish the closest-work review and write a cautious novelty statement.
2. Freeze one threat model; do not mix tensor reconstruction, timing leakage, cache integrity,
   and jailbreak safety.
3. Reproduce an FP16 attack baseline before interpreting compressed-cache results.
4. Evaluate a quantization-aware attacker, not only an attack written for FP16.
5. Measure utility with a compressed cache that online attention actually reads.
6. Add code only after the research question survives these checks.

Use synthetic secrets only. Do not commit model checkpoints, raw KV tensors, credentials, or
sensitive user data.
