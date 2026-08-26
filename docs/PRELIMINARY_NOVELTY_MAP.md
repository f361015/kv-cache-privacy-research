# Preliminary KV-Cache Literature and Novelty Map

**Status:** seed map, not a completed systematic review

**Search date:** August 26, 2026

**Purpose:** identify the papers most likely to establish or destroy novelty before implementation

## Preliminary conclusion

The broad idea "apply mathematics to improve KV-cache quantization" is not novel enough.
Quantization, adaptive bit allocation, random rotations, transform coding, and rate-distortion
objectives are already active and technically mature research areas.

The narrower intersection below remains a **novelty hypothesis to test**:

> Evaluate and optimize KV-cache precision against a quantization-aware tensor reconstruction
> attacker while jointly measuring confidentiality leakage, online model utility, and bitrate.

The current search located work covering pairs of these concerns, but not yet a primary paper
covering this exact three-way intersection. That observation is provisional and must be tested by
citation chaining through Monday.

## Closest-work matrix

| Work | Main object and contribution | Security/privacy property | Compression or mathematical component | Why it is close | What it does not yet settle for us |
| --- | --- | --- | --- | --- | --- |
| [Shadow in the Cache / KV-Cloak](https://arxiv.org/abs/2508.09442) (NDSS 2026) | Inversion, collision, and injection attacks on leaked K/V; reversible matrix defense | Tensor confidentiality and prompt reconstruction | Orthogonal transforms and operator fusion, not a quantization study | Supplies the attacks and strongest direct defense baseline | Does not map leakage across bitrates or optimize a privacy-utility-rate objective |
| [KV-Shield](https://arxiv.org/abs/2409.04040) (2024) | Permutes K/V-related representations with TEE support for on-device inference | GPU-visible tensor leakage | Permutation rather than lossy compression | Directly connects cache representation changes to confidentiality | The supplied thesis argues its raw feature-axis permutation is not RoPE-safe on modern models |
| *Permutation over Precision* (supplied MTech thesis, 2026) | Compares KV-Cloak rotation with RoPE-safe head permutation | Tensor reconstruction under three reproduced attacks | Exact permutation; discusses integer compatibility | Directly overlaps rotation/permutation plus quantized deployment | Explicitly states that no end-to-end INT8/INT4 checkpoint was built and names that extension as future work; implementing it alone is not safely claimable as our independent novelty |
| [KIVI](https://openreview.net/pdf?id=L057s2Rq8O) (ICML 2024) | Asymmetric 2-bit K/V quantization using different granularities | None | Low-bit post-training quantization | Standard low-bit baseline | Evaluates fidelity and efficiency, not reconstruction privacy |
| [KVQuant](https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html) (NeurIPS 2024) | Per-channel, pre-RoPE, non-uniform, dense-and-sparse K/V quantization | None | Sensitivity-aware low-bit quantization and kernels | Occupies RoPE-aware and non-uniform quantization ideas | Does not evaluate leaked-cache confidentiality |
| [AQUA-KV](https://openreview.net/pdf?id=COowwJOAZi) (2025) | Predicts K/V and quantizes residual information | None | Adaptive/predictive compression | Occupies learned dependency-aware quantization | Its loss is fidelity-oriented, not adversarial privacy leakage |
| [TurboQuant](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5c802ef38ab6e366c2ea06eee554c088-Abstract-Conference.html) (ICLR 2026) | Online vector quantization with near-optimal distortion rates | None | Random rotations, scalar quantizers, and unbiased inner-product correction | A strong mathematical baseline for any transform-based idea | Optimizes geometric distortion rather than secret recoverability |
| [KV Cache Compression Through the Lens of Transform Coding](https://arxiv.org/abs/2608.14191) (arXiv, August 2026) | Attention-aware distortion decomposition and reverse-water-filling bit allocation | None | Attention-Aware Transform Coding and rate-distortion allocation | The closest mathematical overlap to a proposed adaptive bit allocator | Uses attention/utility distortion, not a reconstruction adversary or privacy constraint |
| [When Efficiency Meets Safety](https://aclanthology.org/2026.acl-long.1123/) (ACL 2026) | Measures how cache compression affects jailbreak attacks; proposes Safe-CAM | Model safety and jailbreak robustness | Eviction/merging behavior under attacks | Proves compression can change a security outcome | Jailbreak ASR is not tensor confidentiality or prompt reconstruction |
| [LCGuard](https://arxiv.org/abs/2605.22786) (2026) | Learns transformations for K/V communication between agents | Reconstruction privacy in latent multi-agent communication | Adversarial representation transformation | Covers privacy, task utility, and K/V transforms | Not primarily a bitrate/low-precision allocation study; deployment and attacker assumptions differ |
| [SafeKV](https://openreview.net/forum?id=jhDsbd5eXL) (MLArchSys 2025) | Separates private and shareable cache entries | Multi-tenant timing/reuse leakage | Privacy-aware cache management | Covers privacy-efficiency trade-offs in shared serving | Does not defend an exfiltrated K/V tensor against reconstruction |
| *Fractional Rotation, Full Potential?* (supplied paper, 2026) | Applies RoPE to a fraction of head dimensions and studies convergence/memory | None | Structural dimension allocation and QK-Norm stabilization | Relevant to RoPE structure, dimensional allocation, and adjacent memory costs | Studies the RoPE trigonometric cache and training behavior, not KV-tensor leakage or low-bit KV storage |

## What the supplied materials change

### Supplied thesis

The thesis draws a careful boundary around its claims: head permutation is exactly reversible and
appears compatible with integer cache values, but the reported quantization evidence is toy-scale
or indirect. Its future-work chapter explicitly proposes building an INT8/INT4 LLaMA checkpoint and
measuring task accuracy and reconstruction fidelity end to end.

Consequences for our group:

- We may reproduce or extend that experiment, but we must cite it as stated future work.
- "Permutation works with quantized KV" is not a clean independent novelty claim.
- A defensible delta would need a new adversary, objective, guarantee, method, or deployment result.

### Supplied Partial RoPE paper

The paper reports that rotating roughly 10% or more of head dimensions can match full-RoPE
convergence and final loss across its tested settings, while extremely small fractions or NoPE can
slow training or create loss spikes. It concerns the RoPE cache and training architecture rather
than cached K/V tensor confidentiality. It is relevant as an adjacent structural idea and as a
warning that RoPE-stage assumptions must be explicit, but it is not direct prior work on the
proposed privacy-quantization question.

## Candidate directions, ranked provisionally

### 1. Privacy-aware attention-rate allocation - strongest current hypothesis

Let each cache group receive a bit allocation `b_i`. Existing rate-distortion work minimizes an
attention or task distortion subject to a rate budget. A privacy-aware formulation could add an
adaptive reconstruction term or constraint:

```text
min_b  D_attention(b) + lambda_R R(b) - lambda_P L_adaptive_attack(b)
```

or

```text
min_b  D_attention(b) + lambda_R R(b)
subject to Leakage_adaptive(b) <= epsilon
```

This is only promising if the leakage term measures a strong attacker rather than numerical cache
error. The survey must determine whether LCGuard, information-bottleneck methods, or newer
privacy-aware quantizers already instantiate an equivalent objective.

**Required first experiment later:** compare FP16, a uniform quantizer, KIVI/KVQuant-style
quantization, and an adaptive bit allocator against the same quantization-aware reconstruction
attack and genuine online utility path.

### 2. Benchmark: compression versus tensor confidentiality - strong fallback

Build a controlled benchmark that separates:

- exact token recovery;
- semantic recovery;
- sensitive-attribute recovery;
- online task utility;
- bitrate, memory, and latency;
- naive versus quantization-aware attackers.

The benchmark is publishable only if existing evaluations have not already covered equivalent
attacks and metrics. It should not interpret a failed naive attack as a defense.

### 3. RoPE- and head-aware privacy allocation - higher overlap risk

Study whether privacy leakage and utility sensitivity concentrate differently before/after RoPE,
between K and V, and across layers or KV heads. Allocate precision using those differences.

**Overlap risks:** KVQuant already uses pre-RoPE key quantization; mixed-precision papers already
allocate bits by utility sensitivity; the supplied thesis already analyzes which permutations
commute safely with RoPE. A privacy-specific allocation criterion would need to provide the delta.

## Directions not safe to claim as novel now

- Merely applying INT8 or INT4 to the thesis's head permutation.
- Merely evaluating whether lower precision reduces a naive attack's score.
- A generic mixed-precision KV quantizer optimized only for perplexity or reconstruction MSE.
- A random or orthogonal transform followed by quantization without comparison to TurboQuant,
  KVQuant, KV-Cloak, and transform-coding work.
- Treating jailbreak robustness, prefix timing isolation, and exfiltrated-tensor confidentiality as
  interchangeable security results.
- Claiming Partial RoPE as a KV-cache privacy defense without a threat model and adaptive attack.

## Evidence needed before a novelty statement

1. Full-text review of the twelve closest works above, not only their abstracts.
2. Forward and backward citation chaining from KV-Cloak, LCGuard, TurboQuant, and the August 2026
   transform-coding paper.
3. Exact-title and keyword searches in arXiv, OpenReview, ACL Anthology, IEEE/ACM, NDSS, USENIX,
   and Google Scholar or Semantic Scholar.
4. A comparison of objectives at the equation level: optimized variables, constraints, attack
   adaptivity, and assumptions.
5. Written evidence for both the novelty hypothesis and the strongest argument against it.

## Current recommendation

For Monday, recommend **a survey-backed benchmark question first**, with privacy-aware
rate-distortion as the mathematical direction to investigate next. This ordering prevents the group
from designing an optimizer for a phenomenon that may disappear under an adaptive attacker or that
may already be covered by a very recent paper.
