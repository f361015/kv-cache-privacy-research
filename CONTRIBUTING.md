# Contributing

## Branch ownership

- `feat/quantization-*`: Person A
- `feat/attacks-*`: Person B
- `feat/evaluation-*`: Person C
- `main`: reviewed integration only

## Pull request requirements

Every experiment-affecting pull request must state:

- model identifier and immutable revision;
- tokenizer revision;
- random seed;
- prompt-set version;
- cache schema version;
- precision and quantization parameters;
- whether the quantized cache was used during inference or only supplied to an attacker;
- commands used to reproduce the result;
- tests or smoke checks performed.

No result should be merged unless another team member can reproduce at least one representative sample.

## Claim discipline

- Do not call quantization a defense unless an adaptive attacker was evaluated.
- Do not compare utility across different prompts, checkpoints, or generation settings.
- Do not infer privacy from reconstruction metrics alone; report semantic and sensitive-attribute leakage separately.
- Clearly label pilot data, illustrative numbers, incomplete attacks, and exploratory results.
