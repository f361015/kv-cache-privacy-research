# Llama Transformers QuantizedCache Experiment

## Bottom line

In this controlled run, replacing BF16 `DynamicCache` with Transformers' real HQQ-backed INT8 or
INT4 `QuantizedCache` did not stop first-layer token reconstruction. Every one of the 24 synthetic
secret-overlapping tokens, and all six complete synthetic secrets, was recovered as the top-1
nearest vocabulary candidate in every condition. INT4 still changed model behavior: its online
continuation argmax agreed with BF16 on 88.33% of the 60 evaluated decisions.

This does **not** establish that quantization never improves privacy. It establishes only that the
tested HQQ configuration did not hide tokens from this specific layer-0 full-vocabulary collision
diagnostic on this model and prompt set.

## What was actually run

- Model: `unsloth/Llama-3.2-1B`, revision
  `9535bd9b1d1dea6acafbdc4813b728796aeb28da`.
- Weights: 1,235,814,400 parameters loaded in BF16.
- Model architecture: 16 layers, 32 query heads, 8 KV heads, 64 dimensions per head, and a
  128,256-token vocabulary.
- Baseline: BF16 `DynamicCache`.
- Compressed caches: Transformers 5.10.2 `QuantizedCache`, HQQ 0.2.8.post1, 8 or 4 bits, key/value
  axis 1, group size 64, and configured `residual_length=0`.
- Privacy data: six built-in synthetic prompts only; 66 total tokens and 24 tokens overlapping a
  synthetic identifier, codename, or reference number.
- Attack: all 128,256 vocabulary entries were evaluated at every attacked position using the
  Llama layer-0 rotated K/V distance signal.
- Utility: 60 sequential next-token decisions plus 12 built-in MMLU-format multiple-choice
  questions.
- Hardware: NVIDIA GeForce RTX 3060 Laptop GPU, PyTorch 2.11.0+cu126.

No real secret, remote service, access-control bypass, cache injection, or deployed system was
involved.

## Relationship to official KV-Cloak

The official repository's Collision attack constructs candidate K/V entries and compares their
distance to a directly accessed target cache. This experiment uses that same first-layer signal but
changes the target storage cell from full-precision `DynamicCache` to Transformers
`QuantizedCache`.

It is not an unchanged run of the official KV-Cloak CLI. Upstream commit `6b40f36` assumes the
older Transformers cache API and iterable full-precision tensors. This experiment uses Transformers
5.10.2 so the actual current HQQ backend can be instantiated and its packed tensors inspected. It
also does not apply the KV-Cloak obfuscation defense. The comparison here is:

```text
unprotected BF16 cache vs unprotected quantized cache
```

It is not:

```text
KV-Cloak defense vs quantization defense
```

## What attacker adaptation means

Let `Q` denote the known HQQ encoder and `D` its decoder.

The naive or mismatched attacker compares a normal BF16 candidate with the decoded target:

```text
distance(candidate, D(Q(target)))
```

The adapted attacker passes every candidate through the same bit width, grouping, scale, zero
point, packing, and decoding implementation:

```text
distance(D(Q(candidate)), D(Q(target)))
```

This matters because a fixed BF16 distance threshold may reject a quantized match merely because
only one side of the comparison was quantized. Re-encoding is a minimal adaptation, not a proof of
the strongest possible attacker. A stronger attacker could compare integer codes and metadata
directly or model each quantization cell probabilistically.

## Reconstruction result

| Condition | Secret-token top-1 | Exact secrets | Mean winning distance | Mean top-1/top-2 margin |
|---|---:|---:|---:|---:|
| BF16 | 100% | 6/6 | 0.1167 | 8.2935 |
| INT8 naive | 100% | 6/6 | 0.3561 | 8.0628 |
| INT8 adapted | 100% | 6/6 | 0.3712 | 8.0530 |
| INT4 naive | 100% | 6/6 | 4.4216 | 5.2054 |
| INT4 adapted | 100% | 6/6 | 1.6744 | 8.7496 |

The headers mean:

- **Secret-token top-1:** fraction of secret-overlapping tokens whose nearest candidate is the true
  token.
- **Exact secrets:** prompts for which every token overlapping the synthetic secret was recovered
  top-1.
- **Mean winning distance:** mean Euclidean K/V distance from the target to the nearest candidate.
  Lower is a closer numerical match. Because every winner was correct here, it is also the
  true-token distance.
- **Mean top-1/top-2 margin:** second-smallest distance minus smallest distance. A larger positive
  value means the winning token is more separated from the runner-up.

The nonzero BF16 and adapted distances are not evidence of privacy. The intact-sequence layer-0
projection and HQQ re-encoding matched the stored target with zero maximum error. Candidate tokens
are evaluated in large independent BF16 batches, whose matrix-multiplication shape produced up to
0.0625 absolute projection drift relative to the one-sequence target. That normal numerical detail
is recorded in the JSON rather than hidden.

The useful observation is that INT4 strongly increased the naive distance and reduced its margin,
but the correct token still remained nearest. Quantizer-aware re-encoding reduced the INT4 distance
and restored a larger margin. In contrast, INT8 quantization error was already small relative to
the BF16 batch-shape drift, so adaptation did not improve its average distance.

## Cache payload

These values count the persistent tensors in the one-shot prefill caches used as attack targets.
They include packed codes and stored scale/zero tensors, exclude temporary dequantization buffers
and Python object overhead, and are not process-memory measurements.

| Cache | Payload bytes over six prompts | Bits per represented K/V element | Relative to BF16 |
|---|---:|---:|---:|
| BF16 | 2,162,688 | 16.0 | 100% |
| HQQ INT8 | 1,148,928 | 8.5 | 53.1% |
| HQQ INT4 | 608,256 | 4.5 | 28.1% |

HQQ stores BF16 scale and zero metadata for each 64-element group in this configuration. That is
why the physical tensor payload is 8.5 rather than 8 bits/element, and 4.5 rather than 4.

## Online continuation fidelity

Each prompt was processed one token at a time. Later attention steps therefore consumed the cache
returned by the previous step instead of measuring a cache that the model never read.

| Cache | Decisions | BF16 argmax agreement | Mean KL from BF16 | Next-token accuracy | Perplexity |
|---|---:|---:|---:|---:|---:|
| BF16 | 60 | 100% | 0 | 23.33% | 192.53 |
| INT8 | 60 | 98.33% | 0.000846 | 23.33% | 194.04 |
| INT4 | 60 | 88.33% | 0.020101 | 20.00% | 197.75 |

- **Argmax agreement** is the fraction of positions where the cache condition and BF16 chose the
  same most likely next token.
- **KL from BF16** measures how much the full next-token probability distribution moved. Zero means
  identical distributions; larger is more drift.
- **Next-token accuracy** asks whether the most likely prediction equals the actual next token in
  the synthetic text.
- **Perplexity** is `exp(mean next-token negative log-likelihood)`; lower is better on the same
  evaluation set.

The set is too small and artificial for a reportable language-model perplexity. Agreement and KL
are the more useful fidelity checks here.

## MMLU-format diagnostic

The script also scored A/B/C/D logits on 12 built-in questions formatted like zero-shot MMLU. This
is a mechanics and cache-fidelity check, not an official MMLU score: the questions are not the MMLU
dataset and the sample is too small.

| Cache | Accuracy | Prediction agreement with BF16 | Mean correct-choice NLL | Mean KL from BF16 |
|---|---:|---:|---:|---:|
| BF16 | 58.33% | 100% | 1.0469 | 0 |
| INT8 | 58.33% | 100% | 1.0396 | 0.000813 |
| INT4 | 58.33% | 66.67% | 1.0088 | 0.031490 |

Equal accuracy does not mean identical behavior: INT4 changed four of the twelve BF16 predictions,
but gains and losses happened to cancel in aggregate accuracy. A larger official evaluation is
needed before making a task-performance claim.

## Important QuantizedCache residual behavior

The Transformers default is `residual_length=128`. On a 12-token stepwise prompt, it stored one
token in packed form and eleven as a full-precision residual. That default would not represent a
fully compressed short-context experiment.

This run configured `residual_length=0`. In a one-shot prefill—the intercepted attack target—all 12
tokens were packed and the full-precision residual was empty. During token-by-token decoding, the
Transformers 5.10.2 update logic still alternated with at most one current token in the
full-precision residual; after 12 steps the split was 11 packed and one residual token. The result
must therefore be described as an all-packed prefill target and a near-fully-quantized online cache,
not as an invariant that every stored token is packed after every update.

## What attacks were and were not run

Run:

- A layer-0, full-vocabulary nearest-neighbour Collision-style reconstruction diagnostic.
- A mismatched candidate path and an HQQ-aware candidate re-encoding path.

Not run:

- KV-Cloak's inversion attack.
- KV-Cloak's injection attack.
- Collision+ at later layers.
- The KV-Cloak obfuscation defense.
- Any attack against a remote or third-party system.

## Reproduce locally

Install the HQQ backend into an environment with compatible PyTorch and Transformers versions:

```powershell
python -m pip install hqq==0.2.8.post1
```

Keep the checkpoint outside the Git repository, then run from the repository root:

```powershell
python experiments\transformers_quantized_cache_experiment.py `
  --model-path ..\models\Llama-3.2-1B-unsloth `
  --model-revision 9535bd9b1d1dea6acafbdc4813b728796aeb28da `
  --max-prompts 6 `
  --max-tokens 16 `
  --candidate-batch-size 2048 `
  --i-understand-risks
```

The committed summary is
`experiments/results/llama_transformers_quantized_cache.json`. Per-token and per-question CSVs are
regenerable and ignored by Git.

## Professor-facing claim boundary

A careful one-sentence report is:

> On one pinned Llama-3.2-1B checkpoint and six synthetic prompts, actual HQQ INT8/INT4
> `QuantizedCache` reduced cache payload and changed online outputs, but it did not reduce top-1
> token recovery under our layer-0 full-vocabulary Collision-style diagnostic; all 24 synthetic
> secret tokens remained recoverable by both mismatched and quantizer-aware matching.

Do not shorten this to “quantization gives no privacy” or “KV caches are always recoverable.” The
experiment is too narrow for either conclusion.
