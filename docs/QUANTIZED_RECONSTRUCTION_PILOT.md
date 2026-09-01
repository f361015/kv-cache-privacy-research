# Quantized KV Reconstruction Pilot

## Purpose

This pilot asks one narrow question:

> When a directly accessed KV cache is stored at lower precision, does reconstruction remain harder
> after the attacker accounts for the known quantization process?

It is a diagnostic, not a claim that quantization is a privacy defense. The first run uses GPT-2 so
that the mechanics can be tested on the available 6 GB GPU. A Llama-family follow-up using the real
Transformers cache is now recorded in
[LLAMA_TRANSFORMERS_QUANTIZED_CACHE.md](LLAMA_TRANSFORMERS_QUANTIZED_CACHE.md); this document remains
the record of the earlier hand-written-quantizer smoke test.

## Authorized and ethical scope

The experiment runs locally using a public model, the public KV-Cloak research repository, and
synthetic prompts defined in the script. It does not access another user's cache, query a deployed
service, bypass access controls, or process real secrets. The attack flag is an explicit reminder of
this boundary.

## Relationship to KV-Cloak

KV-Cloak's Collision attack reconstructs tokens by generating candidate K/V entries and comparing
them with an accessed target cache. This pilot ports that first-layer distance signal into a smaller
full-vocabulary nearest-neighbour experiment. It references upstream commit `6b40f36` but does not
claim to run the upstream command-line attack unchanged; the installed Transformers version has a
different cache API from the version pinned by KV-Cloak.

At GPT-2 layer 0, the K/V vector for a token and position can be calculated directly from the token
embedding, positional embedding, layer normalization, and K/V projection. The script verifies that
this optimized calculation matches the cache emitted by a complete model forward pass before using
it in the attack.

## What "attacker adaptation" means

Suppose the stored cache is produced by encoder `E_Q`, which includes rounding, clipping, bit width,
grouping, and scale calculation.

The **naive attacker** observes a quantized target, dequantizes it, and compares ordinary
floating-point candidates against it:

```text
distance(float candidate, dequantize(stored target))
```

This attacker is mismatched: the target passed through the quantizer but its candidates did not. A
drop in attack accuracy could therefore be caused by implementation mismatch rather than by a
meaningful reduction in recoverable information.

The pilot's **adapted attacker** knows the quantization scheme and re-encodes every candidate before
comparison:

```text
distance(dequantize(E_Q(candidate)), dequantize(E_Q(target)))
```

For this pilot, `E_Q` is signed symmetric INT8 or INT4 quantization with a separate scale for every
K/V token and attention head. This is a minimal re-encoding adaptation. It is not necessarily the
strongest possible attacker: a later attacker could score integer codes and scale metadata directly,
model quantization cells probabilistically, or use language priors.

## Experimental cells

| Stored target | Candidate treatment | Name |
|---|---|---|
| FP32 | FP32 | `fp32` |
| INT8 | FP32 | `int8_naive` |
| INT8 | INT8 quantize/dequantize | `int8_adapted` |
| INT4 | FP32 | `int4_naive` |
| INT4 | INT4 quantize/dequantize | `int4_adapted` |

Every target token is ranked against the complete GPT-2 vocabulary. Results are reported separately
for all tokens and tokens overlapping the synthetic secret.

## Utility and rate

Utility is measured through sequential teacher-forced inference. After every token, all cache layers
are quantized and dequantized before the next token is processed, so subsequent attention actually
reads the altered cache. Metrics are next-token negative log-likelihood, perplexity, next-token
accuracy, agreement with FP32, and KL divergence from FP32.

The Python pilot does not pack INT4 codes or benchmark a quantized kernel. It reports a logical rate:

```text
code bits + 32 scale bits / head dimension
```

For GPT-2's 64-dimensional heads, this is 8.5 bits per element for INT8 and 4.5 for INT4. These are
logical storage estimates, not measured process memory.

## Initial diagnostic result (2026-08-31)

The first completed run used ten prompts, 103 total tokens, 45 secret-overlapping tokens, and the
complete 50,257-token GPT-2 vocabulary. Each of the ten synthetic secrets was reconstructed exactly
in every condition.

| Condition | Secret-token top-1 | Exact secrets | Mean true-match distance | Mean top-1/top-2 margin |
|---|---:|---:|---:|---:|
| FP32 | 100% | 10/10 | 0.0077 | 18.9971 |
| INT8 naive | 100% | 10/10 | 0.2373 | 18.7658 |
| INT8 adapted | 100% | 10/10 | 0.0054 | 18.9992 |
| INT4 naive | 100% | 10/10 | 4.2985 | 15.2202 |
| INT4 adapted | 100% | 10/10 | 0.0168 | 20.0052 |

Small nonzero distances in the FP32 and adapted rows are numerical error from the batched distance
calculation. The important comparison is that INT4 makes the naive true-match distance much larger,
but the correct token is still the nearest vocabulary entry. Re-encoding the candidates removes that
mismatch and restores a near-zero match. A fixed threshold calibrated on FP32 could therefore reject
an INT4 match even though the identifying information remains available to a quantizer-aware
nearest-neighbour attacker.

The online cache-feedback check evaluated 93 next-token decisions:

| Cache | Logical bits/element | FP32 argmax agreement | KL from FP32 | Perplexity on synthetic prompts |
|---|---:|---:|---:|---:|
| FP32 | 32.0 | 100% | 0 | 428.42 |
| INT8 | 8.5 | 100% | 0.000052 | 426.47 |
| INT4 | 4.5 | 87.10% | 0.019127 | 422.67 |

The tiny synthetic set is not suitable for comparing absolute perplexity, so its apparent decrease
must not be interpreted as an improvement. The meaningful diagnostic is that INT4 changed about 13%
of FP32 argmax decisions while still allowing complete first-layer reconstruction in this run.

This result currently supports only the statement that simple per-head INT8/INT4 quantization did
not prevent first-layer full-vocabulary token identification on GPT-2. It does not establish what
happens on Llama, under later-layer Collision+, or with a production KV quantizer.

## Interpretation

- If the naive attack fails but the adapted attack recovers, the apparent protection was attack
  mismatch.
- If both attacks weaken while online utility remains close to FP32, a privacy-utility-rate effect
  may deserve a larger study.
- If reconstruction and utility degrade together, the result may only reflect ordinary information
  destruction.
- If exact secret tokens become ambiguous but semantic or attribute information remains, lower
  exact recovery must not be called privacy.

## Run locally

The completed run used Python 3.11.15, PyTorch 2.11.0 with CUDA 12.6, and Transformers 5.10.2.
Download the public checkpoint outside the repository so model weights cannot be committed:

```powershell
hf download openai-community/gpt2 `
  config.json generation_config.json merges.txt vocab.json tokenizer.json `
  tokenizer_config.json model.safetensors `
  --local-dir ..\models\gpt2
```

Then, from the repository root, run:

```powershell
python experiments\quantized_collision_pilot.py `
  --model-path ..\models\gpt2 `
  --max-prompts 10 `
  --candidate-batch-size 2048 `
  --i-understand-risks
```

The JSON summary is written to `experiments/results/gpt2_quantized_collision_pilot.json`. The
per-token CSV is intentionally ignored because it is fully reproducible.

## Limitations before any professor-facing claim

- GPT-2 is a smoke-test model and lacks modern GQA/RoPE architecture.
- Only layer 0 and one simple symmetric quantizer are evaluated.
- Re-encoding is a plausible adaptation, not a proof of optimal attack strength.
- Prompts are synthetic and short.
- No packed-cache kernel, latency, or physical memory measurement is included.
- An official Llama/KV-Cloak baseline and at least one production quantizer remain necessary.
