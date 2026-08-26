# Threat model

## Attacker access

The attacker receives a leaked KV cache at a specified precision. The attacker knows the model architecture, model weights, tokenizer, quantization algorithm, grouping, clipping rule, zero points, and scales.

The attacker does not receive the victim prompt, victim response, or secret labels.

## Attack levels

1. **Naive attack:** the original FP16 attack is run without quantization-specific adaptation.
2. **Adaptive attack:** the attacker dequantizes the cache or explicitly optimizes/trains against the quantized representation.
3. **Oracle diagnostic:** optional upper-bound condition that supplies otherwise unavailable alignment information. Oracle results must never be presented as a realistic attack.

## Leakage objectives

- Exact/token reconstruction
- Semantic reconstruction
- Sensitive-attribute inference

## Claim boundary

A reduction in naive-attack performance is not evidence of privacy protection. Any security claim requires evaluation against the adaptive attacker under the same model, prompts, precision, and attack budget.
