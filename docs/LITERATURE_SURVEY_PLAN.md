# Literature Survey and Novelty Review Plan

## Decision and deadline

Until Monday, August 31, 2026, the group will not treat implementation as the primary
deliverable. The immediate goal is to determine what has already been established, which
papers are closest to the proposed direction, and whether a precise research gap remains.

All three people work in parallel. This is an ownership split, not a time schedule.

## Questions the survey must answer

1. What information about prompts, responses, or attributes can be recovered from KV
   tensors, shared prefix caches, persistent caches, or cache-dependent timing?
2. What compression mechanisms already modify KV caches: quantization, mixed precision,
   eviction, merging, low rank, transforms, architectural changes, and offloading?
3. Which papers jointly measure compression and security or privacy, and which security
   property do they actually measure?
4. Has anyone evaluated low-precision KV tensors against a quantization-aware prompt
   reconstruction attacker while also measuring online model utility and bitrate?
5. Which mathematical objectives are already occupied by rate-distortion, transform coding,
   mixed-precision allocation, or adversarial representation learning?
6. What is the smallest defensible research question that is not merely an implementation of
   future work already stated by the supplied thesis?

## Scope and threat-model taxonomy

Every reviewed work must be assigned to one or more categories. Results from different
categories must not be combined as if they measure the same property.

| Category | Adversary observes or changes | Representative outcome |
| --- | --- | --- |
| Tensor confidentiality | Leaked K/V tensors and model information | Prompt or attribute reconstruction |
| Shared-cache side channel | Cache-hit behavior or latency across tenants | Prefix membership or prompt recovery |
| Cache integrity | Writable or replaceable cache blocks | History manipulation or response steering |
| Model safety | Compressed-cache behavior under malicious prompts | Jailbreak attack success rate |
| Latent communication privacy | K/V states transmitted between agents | Reconstruction of agent-private inputs |
| Efficiency and fidelity | Quantized, pruned, merged, or transformed K/V | Bitrate, latency, memory, and task utility |

## Person A - Compression and mathematical foundations

**Primary responsibility:** map the compression literature and determine which mathematical
ideas are already claimed.

Minimum paper families:

- KIVI and KVQuant.
- AQUA-KV and other predictive or residual quantizers.
- TurboQuant and its distortion-rate guarantees.
- Attention-Aware Transform Coding and reverse water-filling.
- Mixed-precision approaches such as PM-KVQ or layer/token/head-adaptive allocation.
- At least one eviction or merging baseline to distinguish lossy token removal from numerical
  quantization.

Deliverables:

- Six or more completed evidence records.
- A comparison of mathematical objectives, assumptions, guarantees, and optimized variables.
- A list of ideas that are already occupied and cannot be claimed as new.
- A short note on which methods have runnable code and realistic baselines.

## Person B - Privacy, attacks, and defenses

**Primary responsibility:** map confidentiality, integrity, side-channel, and safety work
without conflating their threat models.

Minimum paper families:

- *Shadow in the Cache* / KV-Cloak and its three tensor attacks.
- KV-Shield and the supplied thesis's rotation-versus-permutation analysis.
- SafeKV and at least one prefix-cache timing attack or governance defense.
- LCGuard or another learned privacy transform for shared latent K/V communication.
- *When Efficiency Meets Safety* to separate jailbreak robustness from confidentiality.
- At least one cache-integrity or cache-poisoning work.

Deliverables:

- Six or more completed evidence records.
- An attacker-capability matrix: tensor access, model weights, queries, timing, write access,
  quantizer metadata, and adaptive knowledge.
- A defense matrix showing protected property, secret state, overhead, and known bypasses.
- A list of security claims that would be invalid without an adaptive attack.

## Person C - Architecture, systems, and novelty synthesis

**Primary responsibility:** cover adjacent KV-cache work and maintain the integrated novelty
argument.

Minimum paper families:

- The supplied Partial RoPE paper and representative RoPE-aware cache work.
- H2O, StreamingLLM, SnapKV/PyramidKV, or comparable eviction approaches.
- Prefix/persistent-cache systems such as CacheBlend or Cache-Augmented Generation.
- PagedAttention or disaggregated prefill/decode systems that change the exposure surface.
- One current KV-cache survey, used only as a discovery aid and checked against primary papers.
- Citation chaining from the two closest papers found by Persons A and B.

Deliverables:

- Six or more completed evidence records.
- A taxonomy separating representation, architecture, serving, and threat-model changes.
- The integrated closest-work table and bibliography.
- Three ranked research questions with overlap risks, falsifying papers, and a recommendation.

## Shared evidence record

Each paper receives one record with the following fields. A title and abstract summary alone do
not count as a completed review.

1. Full citation, stable URL, venue/status, and version date.
2. Problem and exact contribution in the reviewers' own words.
3. Cache object: K, V, both, prefix hashes, RoPE cache, semantic cache, or another artifact.
4. Stage: pre-RoPE, post-RoPE, post-hoc storage, online attention, cross-machine transport, or
   cross-request reuse.
5. Threat model and protected property, if any.
6. Optimized variable and mathematical objective.
7. Models, datasets, context lengths, bit widths, and hardware.
8. Privacy, utility, and efficiency metrics actually reported.
9. Strongest baselines and whether the attacker is adaptive.
10. Main result with the table or figure location.
11. Limitations and future work stated by the authors.
12. Code/data availability and reproducibility concerns.
13. Relationship to our candidate direction: duplicate, adjacent, enabling, or orthogonal.
14. Confidence: abstract-only, skimmed, or fully reviewed.

## Search protocol

Use primary sources whenever available: proceedings, ACL Anthology, NDSS/USENIX pages,
OpenReview, and arXiv author versions. Surveys, blogs, and repositories are discovery aids, not
the sole evidence for a novelty claim.

Search at least these combinations:

- `KV cache` with `privacy`, `leakage`, `reconstruction`, `inversion`, and `attribute`.
- `KV cache` with `quantization`, `mixed precision`, `rate distortion`, `transform coding`,
  `information bottleneck`, and `adaptive bits`.
- `prefix cache` with `timing`, `side channel`, `multi tenant`, and `prompt recovery`.
- `KV cache compression` with `security`, `safety`, `jailbreak`, `integrity`, and `poisoning`.
- `RoPE` with `quantization`, `commute`, `permutation`, `partial`, and `cache`.

For the two closest papers, perform backward references, forward citations, author-page search,
and exact-title search. Record the search date and database.

## Novelty standard

The survey may say **"we did not locate a direct evaluation"** after documenting its search. It
must not say **"no prior work exists"** merely because a keyword search returned no match.

A candidate is retained only if all three tests pass:

1. **Distinct question:** it changes the protected property, adversary, optimized variable, or
   theoretical objective rather than only the implementation.
2. **Closest-work delta:** the contribution can be explained against the two closest papers in
   one precise paragraph.
3. **Falsifiability:** the group can name the paper or result that would invalidate the novelty
   hypothesis if discovered.

## Monday submission package

- At least 18 fully reviewed primary papers, with approximately six owned by each person.
- A deduplicated bibliography and evidence records using the shared schema.
- One threat-model taxonomy and one closest-work/novelty matrix.
- Three ranked research questions, each with overlap risk and required experiment.
- One recommended direction and one explicit fallback direction.
- A one-page professor brief separating established facts, preliminary inferences, and open
  questions.
- A limitations paragraph describing search coverage and papers not yet fully reviewed.

## Review gates

- No paper is marked fully reviewed without methods, experiments, limitations, and references.
- No privacy claim is inferred from lower task accuracy or a failed naive attack.
- No utility claim is accepted if the quantized cache was never read by online attention.
- No method is called mathematically novel until TurboQuant, transform-coding, mixed-precision,
  and adversarial-representation objectives have been compared explicitly.
- The final recommendation must survive the supplied thesis's future-work section.
