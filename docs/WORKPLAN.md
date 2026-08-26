# KV Cache Quantization x Privacy: Three-Person Research Workplan

**Deadline:** Monday, August 31, 2026

## Research objective

Determine how KV-cache precision affects exact, semantic, and sensitive-information leakage relative to inference utility. Quantization is initially an independent variable, not a proposed defense.

The first scientific deliverable is a privacy-utility-bitrate curve for one pinned model checkpoint under FP16, INT8, and INT4 KV caches.

## Person A - Quantization and utility

**Primary question:** Can the model use a lower-precision cache without losing task performance?

Responsibilities:

- Pin the model/tokenizer revision and deterministic generation settings.
- Implement standardized K/V capture hooks.
- Implement controlled symmetric INT8 and INT4 quantizers.
- Record scales, zero points, grouping, clipping, layer/head layout, and RoPE stage.
- Integrate quantized caches into autoregressive attention for genuine utility evaluation.
- Measure logit divergence, generation agreement, QA, long-context retrieval, memory, and runtime.

Completion gate: repeated FP16 extraction produces identical tensors and metadata.

Completion gate: the model genuinely reads the quantized cache during decoding, and the pilot utility result can be reproduced from one command.

## Person B - Leakage and attacks

**Primary question:** How much information can an informed attacker recover at each precision?

Responsibilities:

- Reproduce one reliable FP16 reconstruction attack before testing quantization.
- Implement token-level, semantic, and sensitive-attribute objectives.
- Evaluate both naive and quantization-adaptive attackers.
- Keep attack budgets and initialization identical across precision levels.
- Write per-sample leakage outputs with convergence and failure diagnostics.
- Perform later K-versus-V, layer, head, token, and partial-cache analysis.

Completion gate: FP16 attack performance is stable across repeated runs and meaningfully exceeds a control baseline.

Completion gate: the adaptive attacker consumes quantized-cache metadata correctly; reduced leakage is not merely a format mismatch.

## Person C - Dataset, evaluation, and reproducibility

**Primary question:** Are the measured differences valid, statistically meaningful, and reproducible?

Responsibilities:

- Build a controlled synthetic dataset with public, personal, financial, medical, and confidential categories.
- Define secret spans and attribute labels before running attacks.
- Maintain immutable prompt splits and experiment manifests.
- Validate paired comparisons across precisions.
- Aggregate per-sample results, confidence intervals, failure cases, and privacy-utility plots.
- Audit cache bundles and pull requests for schema and metadata compliance.
- After the text pipeline passes, lead RAG information-concentration and multimodal extensions.

Completion gate: the pilot manifest, threat model, metrics, and cache schema are frozen before the final paired run.

Completion gate: every reported aggregate is traceable to per-sample records and a reproducible command.

## Shared interface

Person C publishes prompt manifests. Person A produces versioned cache bundles. Person B consumes those bundles and produces per-sample attack records. Person C joins leakage and utility outputs using `prompt_id`, precision, and experiment ID.

Minimum cache metadata:

- model and tokenizer revision;
- prompt ID and token IDs;
- layer/head and K/V shape;
- capture stage and RoPE status;
- original dtype and target bits;
- quantization granularity, scale, zero point, and clipping;
- code revision and random seed.

## Parallel execution

All three people begin immediately and work concurrently:

- Person C publishes the shared smoke-test manifest and experiment identifiers.
- Person A and Person B use those identifiers independently for cache/utility and attack outputs.
- Person C continuously validates schema compatibility and merges completed results.
- A result enters the Monday report only when its reproduction command and limitations are recorded.

## Metrics

Privacy must be reported as three separate dimensions:

- exact recovery: token accuracy, exact sequence accuracy, edit distance;
- semantic recovery: embedding similarity plus manually inspectable examples;
- sensitive leakage: attribute classification and exact secret-span recovery.

Utility must include at least:

- logit divergence or next-token agreement;
- deterministic generation agreement;
- one task metric such as QA accuracy;
- one long-context retrieval metric;
- cache memory and latency.

## Review gates

1. **Protocol gate:** no experiments before model revision, threat model, schema, and dataset version are frozen.
2. **FP16 gate:** quantization work is not interpreted until the FP16 attack baseline is reliable.
3. **Adaptive-attack gate:** naive attack degradation cannot be called a privacy benefit.
4. **Utility gate:** utility requires an actually used quantized cache, not post-hoc cache conversion.
5. **Claim gate:** report confidence intervals, per-category results, and failure cases before drawing conclusions.

## Monday submission package

The repository should contain:

- one end-to-end FP16 smoke test;
- controlled INT8 and INT4 cache conversion on the same prompts;
- at least one adaptive leakage attack across the available precisions;
- an online quantized-cache utility smoke test, or an explicit documented blocker if integration cannot be completed;
- per-sample machine-readable results;
- initial privacy-versus-precision and utility-versus-precision plots;
- a concise limitations section distinguishing completed results from planned experiments.

This is a pilot submission. It does not need to prove a new defense by Monday.

## Later mathematical direction

Only after the baseline curve reveals a stable phenomenon should the group optimize a privacy-aware quantizer. A possible constrained objective is:

`min_Q  L_utility(Q) + lambda_c * R(Q) - lambda_p * L_adaptive_attack(Q)`

where `R(Q)` is bitrate and the privacy term is measured against a strong adaptive attacker. Any proposed method must be compared with simple uniform, per-channel, and per-head quantization at matched bitrate and utility.

## Immediate kickoff checklist

- Assign names to Persons A, B, and C.
- Agree on the pinned model revision.
- Approve the synthetic-data policy.
- Finalize the cache bundle schema.
- Select the first FP16 reconstruction attack.
- Create one end-to-end smoke test using five prompts.
- Hold the first integration review when the smoke test passes.
