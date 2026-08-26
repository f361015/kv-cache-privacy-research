# Experiment contract

## Paired comparison

Every FP16/INT8/INT4 comparison uses the same:

- prompt and prompt identifier;
- model and tokenizer revision;
- KV extraction point;
- generation settings and token budget;
- random seed;
- attack objective and optimization budget;
- evaluation implementation.

## Two distinct pipelines

### Privacy pipeline

Capture FP16 KV tensors, quantize the same tensors under each precision, and supply each representation to the attacker.

### Utility pipeline

Run autoregressive inference while the attention implementation actually reads the quantized KV cache. Post-hoc quantization of an unused cache cannot measure utility degradation.

## Required metadata

Each cache bundle must record:

- schema version;
- prompt ID and token IDs;
- model/tokenizer revision;
- layer and KV-head layout;
- K/V tensor shape and original dtype;
- precision, quantization granularity, scale, zero point, clipping rule;
- whether RoPE has already been applied to keys;
- capture stage and software revision.

## Required output

All methods write per-sample JSONL records. Aggregate tables and figures must be generated from those records, not copied manually.
