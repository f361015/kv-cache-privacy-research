# Literature Research Guide

## Goal

Determine whether the proposed privacy-compression question is new, useful, and experimentally
testable. Prioritize primary papers and equation-level comparison over collecting many abstracts.

## Search areas

Search the following combinations and follow references and later citations from the closest work:

- `KV cache` + privacy, leakage, reconstruction, inversion, attribute inference.
- `KV cache` + quantization, mixed precision, rate distortion, transform coding.
- `prefix cache` + timing, side channel, prompt recovery, multi-tenant.
- `KV cache compression` + security, safety, jailbreak, poisoning, integrity.
- `RoPE` + quantization, permutation, commutativity, partial dimensions.

Prefer proceedings, ACL Anthology, NDSS/USENIX pages, OpenReview, and author-posted arXiv versions.
Use surveys and blogs for discovery, then verify claims in the primary paper.

## Minimum information for each paper

Record:

1. Citation, venue/status, version date, and stable URL.
2. Exact problem and claimed contribution.
3. Cache object and lifecycle stage.
4. Threat model and attacker knowledge.
5. Optimized variables, objective, constraints, and guarantees.
6. Models, datasets, context lengths, bit widths, and hardware.
7. Privacy, utility, and efficiency metrics actually reported.
8. Strongest baselines and whether attacks are adaptive.
9. Main result with table or figure location.
10. Limitations, future work, code availability, and relation to this project.

Use [the review template](PAPER_REVIEW_TEMPLATE.md) for the record.

## Novelty test

A possible contribution is credible only when:

- its difference from the two closest papers is precise;
- it changes the adversary, protected property, objective, guarantee, or evidence rather than only
  implementation details;
- the group can identify what newly discovered paper would invalidate the claim;
- both supporting and contradicting prior work have been recorded.

Avoid "no prior work exists." Prefer: "Within the documented search, we did not locate a study
that jointly evaluates X, Y, and Z."

## High-priority starting set

- *Shadow in the Cache* / KV-Cloak.
- The supplied MTech thesis, especially related work, limitations, and future work.
- KIVI and KVQuant.
- AQUA-KV.
- TurboQuant.
- *KV Cache Compression Through the Lens of Transform Coding*.
- *When Efficiency Meets Safety*.
- LCGuard and SafeKV.
- The supplied Partial RoPE paper.

## Literature deliverable

The useful output is a short closest-work matrix, a threat-model taxonomy, a ranked list of
research gaps, and a limitations paragraph describing search coverage. A long list of papers
without these comparisons is not sufficient.
