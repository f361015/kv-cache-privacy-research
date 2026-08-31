# Project Context

## Research possibility

The project may study whether low-precision or otherwise compressed KV tensors change prompt or
sensitive-attribute reconstruction under a strong attacker, and whether that change can be used in
a principled privacy-utility-rate objective.

A possible later formulation is to choose cache bit allocations `b` that balance:

```text
attention or task distortion + rate cost - adaptive attack loss
```

This formulation is only a hypothesis. The literature review must establish whether an equivalent
objective or evaluation already exists.

## Boundaries that matter

These are different research problems and must be reviewed separately:

- **Tensor confidentiality:** an attacker obtains K/V tensors and attempts reconstruction.
- **Shared-cache side channels:** an attacker infers prefixes through cache hits or timing.
- **Cache integrity:** an attacker modifies or replaces cache state.
- **Model safety:** compression changes jailbreak or harmful-generation behavior.
- **Latent sharing:** K/V states are communicated between models or agents.

A result in one category is not automatically evidence for another.

## Closest-work risks already identified

- KV-Cloak introduces inversion, collision, and injection attacks on leaked K/V and uses a
  reversible matrix transform as a defense.
- KV-Shield uses permutation for on-device cache protection; the supplied thesis argues that its
  raw feature-axis permutation is not RoPE-safe on modern models.
- The supplied thesis compares KV-Cloak-style rotation with head permutation and explicitly lists
  an end-to-end INT8/INT4 extension as future work. Implementing that extension alone is therefore
  not a safe independent novelty claim.
- KIVI, KVQuant, AQUA-KV, and related work already cover low-bit, asymmetric, non-uniform,
  predictive, and sensitivity-aware KV quantization.
- TurboQuant already provides a mathematical distortion-rate treatment using rotations and
  quantization.
- *KV Cache Compression Through the Lens of Transform Coding* already derives attention-aware
  distortion and reverse-water-filling bit allocation.
- *When Efficiency Meets Safety* studies compression against jailbreak attacks, which is relevant
  but distinct from prompt reconstruction.
- LCGuard learns privacy-oriented transformations for K/V communication between agents.
- SafeKV and related serving work address cross-tenant cache sharing and timing leakage rather
  than exfiltrated-tensor reconstruction.
- The supplied Partial RoPE paper is adjacent architectural work. It studies how many dimensions
  receive RoPE and the resulting training/memory behavior, not KV-tensor confidentiality.

## Current novelty hypothesis

The working gap is narrower than generic KV-cache quantization:

> A controlled evaluation of quantized KV tensors against a quantization-aware reconstruction
> attacker, jointly reporting exact, semantic, and attribute leakage; genuine online utility; and
> bitrate or memory cost.

The literature search has not yet established that this is novel. Use the wording "we have not yet
located a direct study" until citation chaining and full-text review are complete.

## Pilot status (2026-08-31)

A controlled GPT-2 smoke test now implements the smallest version of this evaluation. It uses
synthetic prompts, a full-vocabulary first-layer collision matcher, symmetric INT8/INT4 cache
encoding, a quantizer-aware candidate re-encoding path, and online cache-feedback utility. The first
ten-prompt run recovered every secret in every precision condition. This is evidence that the
experiment mechanics work and that quantization was not protective in this particular setting; it is
not evidence of general behavior on modern Llama-family models. See
[the pilot document](QUANTIZED_RECONSTRUCTION_PILOT.md) for the exact scope and results.

## Claims to avoid

- Quantization is a privacy defense because a naive attack performs worse.
- Lower cache fidelity necessarily means lower semantic or attribute leakage.
- A post-hoc quantized cache measures model utility when the model never reads it.
- Jailbreak robustness, timing isolation, and tensor confidentiality are interchangeable.
- A rotation, permutation, mixed-precision allocator, or rate-distortion objective is novel without
  direct comparison to the closest mathematical work.

## Sensible first experiment if the project resumes

Use one pinned model and one synthetic prompt set. Compare FP16 with uniform INT8 and INT4 under
the same prompts and seeds. Run both an FP16-oriented attack and an attacker that knows the
quantizer, grouping, clipping, scales, and zero points. Separately run online inference in which
attention consumes the compressed cache. Report per-sample leakage, utility, bitrate, and failure
cases before proposing a new optimizer.
