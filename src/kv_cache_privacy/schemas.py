"""Shared metadata contracts for quantization, attacks, and evaluation."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CacheBundleMetadata:
    """Minimum metadata required for a reproducible KV-cache bundle."""

    schema_version: int
    experiment_id: str
    prompt_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    code_revision: str
    seed: int
    layer_count: int
    kv_head_count: int
    head_dim: int
    sequence_length: int
    original_dtype: str
    bits: Literal[4, 8, 16]
    quantization_family: str
    quantization_granularity: str
    rope_applied_to_keys: bool
    used_during_inference: bool
