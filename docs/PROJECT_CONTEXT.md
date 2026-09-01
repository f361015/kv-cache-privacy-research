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

## Experiment status (2026-09-01)

The GPT-2 smoke test has been followed by a controlled Llama-3.2-1B run using Transformers 5.10.2's
real HQQ-backed `QuantizedCache`. The experiment used six synthetic prompts, 66 tokens, 24
secret-overlapping tokens, and a full 128,256-token vocabulary search at layer 0. BF16, INT8 naive,
INT8 adapted, INT4 naive, and INT4 adapted matching all achieved 100% secret-token top-1 recovery
and 6/6 exact synthetic secrets. INT4 reduced the measured prefill cache tensor payload from 16 to
4.5 bits per represented element but changed 11.7% of BF16 continuation argmax decisions. See
[the Llama experiment document](LLAMA_TRANSFORMERS_QUANTIZED_CACHE.md) for the precise threat model,
cache semantics, metrics, and limitations. This result is evidence about one controlled cell only;
it is not evidence that every KV quantizer or reconstruction attack behaves the same way.

## Claims to avoid

- Quantization is a privacy defense because a naive attack performs worse.
- Lower cache fidelity necessarily means lower semantic or attribute leakage.
- A post-hoc quantized cache measures model utility when the model never reads it.
- Jailbreak robustness, timing isolation, and tensor confidentiality are interchangeable.
- A rotation, permutation, mixed-precision allocator, or rate-distortion objective is novel without
  direct comparison to the closest mathematical work.

## Sensible first experiment if the project resumes

Use one pinned model and one synthetic prompt set. Compare BF16 with uniform INT8 and INT4 under
the same prompts and seeds. Run both a BF16-oriented attack and an attacker that knows the
quantizer, grouping, clipping, scales, and zero points. Separately run online inference in which
attention consumes the compressed cache. Report per-sample leakage, utility, bitrate, and failure
cases before proposing a new optimizer.
