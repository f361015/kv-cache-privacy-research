"""Real Transformers QuantizedCache privacy/utility experiment.

This is a local, controlled reconstruction diagnostic over built-in synthetic prompts. It uses
Transformers' HQQ-backed ``QuantizedCache`` for both the intercepted target and the attacker's
quantizer-aware candidate model. It neither probes a remote service nor processes real secrets.

The reconstruction is a compact port of KV-Cloak's first-layer collision signal, not an unchanged
run of the upstream KV-Cloak CLI. The MMLU-format questions are a small built-in diagnostic and are
not a reportable MMLU benchmark score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, HQQQuantizedLayer, QuantizedCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


SYNTHETIC_CASES = [
    ("The temporary identifier is Q7M4P9.", "Q7M4P9"),
    ("The temporary identifier is R2K8V6.", "R2K8V6"),
    ("The project codename is SILVER ORBIT.", "SILVER ORBIT"),
    ("The project codename is AMBER COMET.", "AMBER COMET"),
    ("The test reference number is 482917.", "482917"),
    ("The test reference number is 731604.", "731604"),
    ("The synthetic account label is BLUE HARBOR.", "BLUE HARBOR"),
    ("The synthetic account label is QUIET CEDAR.", "QUIET CEDAR"),
    ("The temporary access phrase is MAPLE GLASS.", "MAPLE GLASS"),
    ("The temporary access phrase is COPPER RAIN.", "COPPER RAIN"),
]


MMLU_FORMAT_CASES = [
    {
        "subject": "abstract algebra",
        "question": "Which element is the additive identity in the integers?",
        "choices": ["-1", "0", "1", "There is no identity"],
        "answer": 1,
    },
    {
        "subject": "astronomy",
        "question": "Which planet is closest to the Sun?",
        "choices": ["Venus", "Earth", "Mercury", "Mars"],
        "answer": 2,
    },
    {
        "subject": "biology",
        "question": "Which molecule carries genetic information in most living organisms?",
        "choices": ["ATP", "DNA", "Glucose", "Cholesterol"],
        "answer": 1,
    },
    {
        "subject": "chemistry",
        "question": "What is the chemical symbol for sodium?",
        "choices": ["S", "So", "Na", "N"],
        "answer": 2,
    },
    {
        "subject": "computer science",
        "question": "Which data structure uses first-in, first-out order?",
        "choices": ["Stack", "Queue", "Binary tree", "Hash function"],
        "answer": 1,
    },
    {
        "subject": "economics",
        "question": "If demand rises while supply is unchanged, what usually happens to price?",
        "choices": ["It rises", "It falls", "It becomes zero", "It must remain fixed"],
        "answer": 0,
    },
    {
        "subject": "geography",
        "question": "What is the largest ocean on Earth?",
        "choices": ["Atlantic", "Indian", "Arctic", "Pacific"],
        "answer": 3,
    },
    {
        "subject": "logic",
        "question": "If all A are B and all B are C, which conclusion follows?",
        "choices": ["All C are A", "All A are C", "No A are C", "No conclusion follows"],
        "answer": 1,
    },
    {
        "subject": "physics",
        "question": "What is the SI unit of force?",
        "choices": ["Joule", "Watt", "Pascal", "Newton"],
        "answer": 3,
    },
    {
        "subject": "statistics",
        "question": "Which statistic is the middle value of an ordered sample?",
        "choices": ["Mean", "Variance", "Median", "Range"],
        "answer": 2,
    },
    {
        "subject": "world history",
        "question": "The Renaissance began in which present-day country?",
        "choices": ["Italy", "Norway", "Canada", "Japan"],
        "answer": 0,
    },
    {
        "subject": "mathematics",
        "question": "What is the derivative of x squared with respect to x?",
        "choices": ["x", "2x", "x squared", "2"],
        "answer": 1,
    },
]


@dataclass
class TokenRecord:
    sample_id: int
    prompt: str
    secret: str
    position: int
    token_id: int
    token_text: str
    is_secret: bool
    baseline_kv: torch.Tensor
    quantized_kv: dict[int, torch.Tensor]


def secret_token_mask(
    offsets: Iterable[tuple[int, int]], start: int, end: int
) -> list[bool]:
    """Mark tokens whose character span overlaps the synthetic secret."""
    return [offset_end > start and offset_start < end for offset_start, offset_end in offsets]


def make_quantized_cache(model, bits: int, residual_length: int = 0) -> QuantizedCache:
    """Construct the exact HQQ-backed cache used throughout this experiment."""
    return QuantizedCache(
        backend="hqq",
        config=model.config,
        nbits=bits,
        axis_key=1,
        axis_value=1,
        q_group_size=model.config.head_dim,
        residual_length=residual_length,
    )


def quantized_layer_token_counts(layer) -> tuple[int, int]:
    """Return (packed/dequantized tokens, full-precision residual tokens)."""
    quantized = 0
    if hasattr(layer, "_quantized_keys"):
        quantized = int(layer._dequantize(layer._quantized_keys).shape[-2])
    residual = int(layer.keys.shape[-2]) if layer.keys.dim() == 4 else 0
    return quantized, residual


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def dynamic_cache_payload(cache: DynamicCache) -> dict[str, int]:
    """Count tensor payload bytes/elements, excluding Python object overhead."""
    payload_bytes = 0
    elements = 0
    for layer in cache.layers:
        for tensor in (layer.keys, layer.values):
            payload_bytes += _tensor_bytes(tensor)
            elements += tensor.numel()
    return {"payload_bytes": payload_bytes, "represented_elements": elements}


def quantized_cache_payload(cache: QuantizedCache) -> dict[str, int]:
    """Count packed codes, scale/zero metadata, and any float residual separately."""
    code_bytes = 0
    metadata_bytes = 0
    residual_bytes = 0
    represented_elements = 0
    for layer in cache.layers:
        for quantized_name in ("_quantized_keys", "_quantized_values"):
            packed, metadata = getattr(layer, quantized_name)
            code_bytes += _tensor_bytes(packed)
            represented_elements += math.prod(metadata["shape"])
            for value in metadata.values():
                if torch.is_tensor(value):
                    metadata_bytes += _tensor_bytes(value)
        for tensor in (layer.keys, layer.values):
            residual_bytes += _tensor_bytes(tensor)
            represented_elements += tensor.numel()
    total = code_bytes + metadata_bytes + residual_bytes
    return {
        "payload_bytes": total,
        "packed_code_bytes": code_bytes,
        "scale_zero_metadata_bytes": metadata_bytes,
        "full_precision_residual_bytes": residual_bytes,
        "represented_elements": represented_elements,
    }


@torch.inference_mode()
def first_layer_candidate_kv(model, token_ids: torch.Tensor, position: int) -> torch.Tensor:
    """Compute Llama layer-0 rotated keys and values for independent candidate tokens."""
    if model.config.model_type != "llama":
        raise ValueError("This experiment currently supports Llama checkpoints only.")
    base = model.model
    attention = base.layers[0].self_attn
    hidden = base.embed_tokens(token_ids).unsqueeze(1)
    normalized = base.layers[0].input_layernorm(hidden)
    batch_size = token_ids.shape[0]
    key = attention.k_proj(normalized).view(
        batch_size, 1, model.config.num_key_value_heads, model.config.head_dim
    )
    value = attention.v_proj(normalized).view(
        batch_size, 1, model.config.num_key_value_heads, model.config.head_dim
    )
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    position_ids = torch.full(
        (batch_size, 1), position, dtype=torch.long, device=token_ids.device
    )
    cos, sin = base.rotary_emb(hidden, position_ids)
    _, key = apply_rotary_pos_emb(torch.zeros_like(key), key, cos, sin)
    return torch.stack((key.squeeze(2), value.squeeze(2)), dim=1)


@torch.inference_mode()
def first_layer_sequence_kv(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Project one intact sequence, preserving the model forward's BF16 GEMM shape."""
    base = model.model
    attention = base.layers[0].self_attn
    hidden = base.embed_tokens(input_ids)
    normalized = base.layers[0].input_layernorm(hidden)
    sequence_length = input_ids.shape[1]
    key = attention.k_proj(normalized).view(
        1, sequence_length, model.config.num_key_value_heads, model.config.head_dim
    )
    value = attention.v_proj(normalized).view(
        1, sequence_length, model.config.num_key_value_heads, model.config.head_dim
    )
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    position_ids = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
    cos, sin = base.rotary_emb(hidden, position_ids)
    _, key = apply_rotary_pos_emb(torch.zeros_like(key), key, cos, sin)
    return torch.stack((key[0].transpose(0, 1), value[0].transpose(0, 1)), dim=1)


@torch.inference_mode()
def hqq_quantize_dequantize_kv(kv: torch.Tensor, bits: int) -> torch.Tensor:
    """Re-encode attacker candidates through the same HQQ implementation as QuantizedCache."""
    head_dim = kv.shape[-1]
    layer = HQQQuantizedLayer(
        nbits=bits,
        axis_key=1,
        axis_value=1,
        q_group_size=head_dim,
        residual_length=0,
    )
    key = kv[:, 0].unsqueeze(2)
    value = kv[:, 1].unsqueeze(2)
    layer.update(key, value)
    dequantized_key = layer._dequantize(layer._quantized_keys).squeeze(2)
    dequantized_value = layer._dequantize(layer._quantized_values).squeeze(2)
    return torch.stack((dequantized_key, dequantized_value), dim=1)


@torch.inference_mode()
def collect_targets(model, tokenizer, max_prompts: int, max_tokens: int, device):
    records: list[TokenRecord] = []
    prompt_metadata = []
    storage_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sample_id, (prompt, secret) in enumerate(SYNTHETIC_CASES[:max_prompts]):
        encoded = tokenizer(
            prompt,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids[:, :max_tokens].to(device)
        offsets = [
            tuple(pair) for pair in encoded.offset_mapping[0, : input_ids.shape[1]].tolist()
        ]
        secret_start = prompt.index(secret)
        secret_end = secret_start + len(secret)
        is_secret = secret_token_mask(offsets, secret_start, secret_end)

        baseline_cache = model(input_ids=input_ids, use_cache=True).past_key_values
        baseline_payload = dynamic_cache_payload(baseline_cache)
        for key, value in baseline_payload.items():
            storage_totals["bf16"][key] += value

        quantized_caches = {}
        for bits in (8, 4):
            cache = make_quantized_cache(model, bits, residual_length=0)
            cache = model(
                input_ids=input_ids, past_key_values=cache, use_cache=True
            ).past_key_values
            quantized_caches[bits] = cache
            payload = quantized_cache_payload(cache)
            for key, value in payload.items():
                storage_totals[f"int{bits}"][key] += value

        token_ids = input_ids[0].tolist()
        baseline_layer = baseline_cache.layers[0]
        quantized_layers = {bits: cache.layers[0] for bits, cache in quantized_caches.items()}
        quantized_tensors = {
            bits: (
                layer._dequantize(layer._quantized_keys)[0],
                layer._dequantize(layer._quantized_values)[0],
            )
            for bits, layer in quantized_layers.items()
        }
        for position, token_id in enumerate(token_ids):
            baseline_kv = torch.stack(
                (
                    baseline_layer.keys[0, :, position, :],
                    baseline_layer.values[0, :, position, :],
                )
            ).detach()
            quantized_kv = {
                bits: torch.stack((key[:, position, :], value[:, position, :])).detach()
                for bits, (key, value) in quantized_tensors.items()
            }
            records.append(
                TokenRecord(
                    sample_id=sample_id,
                    prompt=prompt,
                    secret=secret,
                    position=position,
                    token_id=token_id,
                    token_text=tokenizer.convert_ids_to_tokens(token_id),
                    is_secret=is_secret[position],
                    baseline_kv=baseline_kv,
                    quantized_kv=quantized_kv,
                )
            )
        prompt_metadata.append(
            {
                "sample_id": sample_id,
                "prompt": prompt,
                "secret": secret,
                "token_count": len(token_ids),
                "secret_token_count": sum(is_secret),
            }
        )
    return records, prompt_metadata, storage_totals


def update_topk(current_scores, current_ids, batch_scores, batch_ids):
    expanded_ids = batch_ids.unsqueeze(0).expand(batch_scores.shape[0], -1)
    all_scores = torch.cat((current_scores, batch_scores), dim=1)
    all_ids = torch.cat((current_ids, expanded_ids), dim=1)
    scores, indices = torch.topk(
        all_scores, k=current_scores.shape[1], largest=False, dim=1
    )
    return scores, torch.gather(all_ids, 1, indices)


@torch.inference_mode()
def run_reconstruction(model, records, batch_size: int, device):
    by_position: dict[int, list[TokenRecord]] = defaultdict(list)
    for record in records:
        by_position[record.position].append(record)

    conditions = ("bf16", "int8_naive", "int8_adapted", "int4_naive", "int4_adapted")
    output_rows = []
    for position in sorted(by_position):
        position_records = by_position[position]
        targets = {
            "bf16": torch.stack([record.baseline_kv for record in position_records]).to(device),
            "int8_naive": torch.stack(
                [record.quantized_kv[8] for record in position_records]
            ).to(device),
            "int8_adapted": torch.stack(
                [record.quantized_kv[8] for record in position_records]
            ).to(device),
            "int4_naive": torch.stack(
                [record.quantized_kv[4] for record in position_records]
            ).to(device),
            "int4_adapted": torch.stack(
                [record.quantized_kv[4] for record in position_records]
            ).to(device),
        }
        best_scores = {
            name: torch.full((len(position_records), 5), math.inf, device=device)
            for name in conditions
        }
        best_ids = {
            name: torch.full(
                (len(position_records), 5), -1, dtype=torch.long, device=device
            )
            for name in conditions
        }

        for start in range(0, model.config.vocab_size, batch_size):
            candidate_ids = torch.arange(
                start, min(start + batch_size, model.config.vocab_size), device=device
            )
            candidate_bf16 = first_layer_candidate_kv(model, candidate_ids, position)
            candidate_int8 = hqq_quantize_dequantize_kv(candidate_bf16, 8)
            candidate_int4 = hqq_quantize_dequantize_kv(candidate_bf16, 4)
            candidates = {
                "bf16": candidate_bf16,
                "int8_naive": candidate_bf16,
                "int8_adapted": candidate_int8,
                "int4_naive": candidate_bf16,
                "int4_adapted": candidate_int4,
            }
            for name in conditions:
                candidate_flat = candidates[name].reshape(candidate_ids.shape[0], -1).float()
                target_flat = targets[name].reshape(len(position_records), -1).float()
                distances = torch.cdist(target_flat, candidate_flat)
                best_scores[name], best_ids[name] = update_topk(
                    best_scores[name], best_ids[name], distances, candidate_ids
                )

        for row_index, record in enumerate(position_records):
            for name in conditions:
                ids = best_ids[name][row_index].tolist()
                output_rows.append(
                    {
                        "sample_id": record.sample_id,
                        "position": record.position,
                        "token_id": record.token_id,
                        "token_text": record.token_text,
                        "is_secret": record.is_secret,
                        "condition": name,
                        "prediction_id": ids[0],
                        "prediction_text": "",
                        "top5_ids": ids,
                        "top1_correct": ids[0] == record.token_id,
                        "top5_correct": record.token_id in ids,
                        "distance": best_scores[name][row_index, 0].item(),
                        "second_distance": best_scores[name][row_index, 1].item(),
                        "margin": (
                            best_scores[name][row_index, 1]
                            - best_scores[name][row_index, 0]
                        ).item(),
                    }
                )
        print(
            f"Reconstruction position {position}: {len(position_records)} targets complete",
            flush=True,
        )
    return output_rows


def _new_cache(model, condition: str):
    if condition == "bf16":
        return DynamicCache(config=model.config)
    return make_quantized_cache(model, int(condition.removeprefix("int")), residual_length=0)


@torch.inference_mode()
def online_logits(model, input_ids: torch.Tensor, condition: str) -> torch.Tensor:
    cache = _new_cache(model, condition)
    logits = []
    for position in range(input_ids.shape[1] - 1):
        outputs = model(
            input_ids=input_ids[:, position : position + 1],
            past_key_values=cache,
            use_cache=True,
        )
        logits.append(outputs.logits[:, -1, :].float().cpu())
        cache = outputs.past_key_values
    return torch.cat(logits, dim=0)


@torch.inference_mode()
def run_continuation_fidelity(model, tokenizer, max_prompts, max_tokens, device):
    accumulators = {
        name: {"nll": 0.0, "tokens": 0, "correct": 0, "agreement": 0, "kl": 0.0}
        for name in ("bf16", "int8", "int4")
    }
    for prompt, _ in SYNTHETIC_CASES[:max_prompts]:
        input_ids = tokenizer(
            prompt, add_special_tokens=True, return_tensors="pt"
        ).input_ids[:, :max_tokens].to(device)
        if input_ids.shape[1] < 2:
            continue
        targets = input_ids[0, 1:].cpu()
        logits = {
            name: online_logits(model, input_ids, name) for name in accumulators
        }
        baseline_log_probs = F.log_softmax(logits["bf16"], dim=-1)
        baseline_probs = baseline_log_probs.exp()
        baseline_argmax = logits["bf16"].argmax(dim=-1)
        for name, condition_logits in logits.items():
            log_probs = F.log_softmax(condition_logits, dim=-1)
            accumulator = accumulators[name]
            accumulator["nll"] += F.nll_loss(log_probs, targets, reduction="sum").item()
            accumulator["tokens"] += targets.numel()
            predictions = condition_logits.argmax(dim=-1)
            accumulator["correct"] += (predictions == targets).sum().item()
            accumulator["agreement"] += (predictions == baseline_argmax).sum().item()
            accumulator["kl"] += (
                baseline_probs * (baseline_log_probs - log_probs)
            ).sum().item()

    summary = {}
    for name, accumulator in accumulators.items():
        count = accumulator["tokens"]
        mean_nll = accumulator["nll"] / count
        summary[name] = {
            "evaluated_tokens": count,
            "mean_nll": mean_nll,
            "perplexity": math.exp(mean_nll),
            "next_token_accuracy": accumulator["correct"] / count,
            "argmax_agreement_with_bf16": accumulator["agreement"] / count,
            "mean_kl_from_bf16": accumulator["kl"] / count,
        }
    return summary


def format_mmlu_prompt(case: dict) -> str:
    choices = "\n".join(
        f"{'ABCD'[index]}. {choice}" for index, choice in enumerate(case["choices"])
    )
    return (
        "The following is a multiple choice question about "
        f"{case['subject']}.\n\nQuestion: {case['question']}\n{choices}\nAnswer"
    )


@torch.inference_mode()
def mmlu_choice_logits(model, tokenizer, prompt: str, condition: str, choice_ids, device):
    prefix_ids = tokenizer(
        prompt, add_special_tokens=True, return_tensors="pt"
    ).input_ids.to(device)
    trigger_ids = tokenizer(":", add_special_tokens=False, return_tensors="pt").input_ids.to(
        device
    )
    cache = _new_cache(model, condition)
    cache = model(
        input_ids=prefix_ids, past_key_values=cache, use_cache=True
    ).past_key_values
    logits = model(
        input_ids=trigger_ids, past_key_values=cache, use_cache=True
    ).logits[0, -1]
    return logits[choice_ids].float().cpu()


@torch.inference_mode()
def run_mmlu_format_diagnostic(model, tokenizer, device):
    choice_ids = []
    for letter in "ABCD":
        ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Expected one token for choice {letter}, got {ids}")
        choice_ids.append(ids[0])

    rows = []
    for case_index, case in enumerate(MMLU_FORMAT_CASES):
        prompt = format_mmlu_prompt(case)
        condition_logits = {
            condition: mmlu_choice_logits(
                model, tokenizer, prompt, condition, choice_ids, device
            )
            for condition in ("bf16", "int8", "int4")
        }
        baseline_probs = F.softmax(condition_logits["bf16"], dim=-1)
        for condition, logits in condition_logits.items():
            log_probs = F.log_softmax(logits, dim=-1)
            prediction = int(logits.argmax())
            rows.append(
                {
                    "case_id": case_index,
                    "subject": case["subject"],
                    "condition": condition,
                    "correct_index": case["answer"],
                    "prediction_index": prediction,
                    "correct": prediction == case["answer"],
                    "agrees_with_bf16": prediction
                    == int(condition_logits["bf16"].argmax()),
                    "correct_choice_nll": -log_probs[case["answer"]].item(),
                    "kl_from_bf16": (
                        baseline_probs
                        * (F.log_softmax(condition_logits["bf16"], dim=-1) - log_probs)
                    ).sum().item(),
                }
            )

    summary = {}
    for condition in ("bf16", "int8", "int4"):
        selected = [row for row in rows if row["condition"] == condition]
        summary[condition] = {
            "questions": len(selected),
            "accuracy": sum(row["correct"] for row in selected) / len(selected),
            "prediction_agreement_with_bf16": sum(
                row["agrees_with_bf16"] for row in selected
            )
            / len(selected),
            "mean_correct_choice_nll": sum(
                row["correct_choice_nll"] for row in selected
            )
            / len(selected),
            "mean_kl_from_bf16": sum(row["kl_from_bf16"] for row in selected)
            / len(selected),
        }
    return summary, rows


@torch.inference_mode()
def audit_residual_window(model, tokenizer, device, max_tokens: int):
    prompt = SYNTHETIC_CASES[0][0]
    input_ids = tokenizer(
        prompt, add_special_tokens=True, return_tensors="pt"
    ).input_ids[:, :max_tokens].to(device)
    audits = {}
    for residual_length in (128, 0):
        prefill_cache = make_quantized_cache(model, 4, residual_length=residual_length)
        prefill_cache = model(
            input_ids=input_ids, past_key_values=prefill_cache, use_cache=True
        ).past_key_values
        prefill_quantized, prefill_residual = quantized_layer_token_counts(
            prefill_cache.layers[0]
        )

        stepwise_cache = make_quantized_cache(model, 4, residual_length=residual_length)
        for position in range(input_ids.shape[1]):
            stepwise_cache = model(
                input_ids=input_ids[:, position : position + 1],
                past_key_values=stepwise_cache,
                use_cache=True,
            ).past_key_values
        stepwise_quantized, stepwise_residual = quantized_layer_token_counts(
            stepwise_cache.layers[0]
        )
        audits[str(residual_length)] = {
            "sequence_tokens": int(input_ids.shape[1]),
            "one_shot_prefill": {
                "quantized_tokens_layer_0": prefill_quantized,
                "full_precision_residual_tokens_layer_0": prefill_residual,
            },
            "stepwise_after_sequence": {
                "quantized_tokens_layer_0": stepwise_quantized,
                "full_precision_residual_tokens_layer_0": stepwise_residual,
            },
        }
    return audits


def summarize_reconstruction(rows: list[dict]) -> dict:
    summary = {}
    for condition in sorted({row["condition"] for row in rows}):
        condition_rows = [row for row in rows if row["condition"] == condition]
        secret_rows = [row for row in condition_rows if row["is_secret"]]

        def metrics(selected):
            return {
                "tokens": len(selected),
                "top1_accuracy": sum(row["top1_correct"] for row in selected)
                / len(selected),
                "top5_accuracy": sum(row["top5_correct"] for row in selected)
                / len(selected),
                "mean_top1_distance": sum(row["distance"] for row in selected)
                / len(selected),
                "mean_top1_to_top2_margin": sum(row["margin"] for row in selected)
                / len(selected),
            }

        secret_by_sample = defaultdict(list)
        all_by_sample = defaultdict(list)
        for row in condition_rows:
            all_by_sample[row["sample_id"]].append(row)
            if row["is_secret"]:
                secret_by_sample[row["sample_id"]].append(row)
        summary[condition] = {
            "all_tokens": metrics(condition_rows),
            "secret_tokens": metrics(secret_rows),
            "sequences": {
                "prompts": len(all_by_sample),
                "exact_prompt_top1_rate": sum(
                    all(row["top1_correct"] for row in sample_rows)
                    for sample_rows in all_by_sample.values()
                )
                / len(all_by_sample),
                "exact_secret_top1_rate": sum(
                    all(row["top1_correct"] for row in sample_rows)
                    for sample_rows in secret_by_sample.values()
                )
                / len(secret_by_sample),
                "exact_secret_within_top5_rate": sum(
                    all(row["top5_correct"] for row in sample_rows)
                    for sample_rows in secret_by_sample.values()
                )
                / len(secret_by_sample),
            },
        }
    return summary


def summarize_storage(storage_totals) -> dict:
    summary = {}
    for condition, payload in storage_totals.items():
        represented = payload["represented_elements"]
        item = dict(payload)
        item["actual_payload_bits_per_represented_element"] = (
            8 * payload["payload_bytes"] / represented
        )
        summary[condition] = item
    return summary


def write_outputs(output_dir: Path, result: dict, attack_rows: list[dict], utility_rows, tokenizer):
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in attack_rows:
        row["prediction_text"] = tokenizer.convert_ids_to_tokens(row["prediction_id"])
    with (output_dir / "llama_transformers_quantized_cache.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (output_dir / "llama_transformers_quantized_attack_tokens.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(attack_rows[0]))
        writer.writeheader()
        writer.writerows(attack_rows)
    with (output_dir / "llama_mmlu_format_questions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(utility_rows[0]))
        writer.writeheader()
        writer.writerows(utility_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-prompts", type=int, default=6, choices=range(1, 11))
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--candidate-batch-size", type=int, default=2048)
    parser.add_argument(
        "--i-understand-risks",
        action="store_true",
        help="Acknowledge that local reconstruction runs on built-in synthetic data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.i_understand_risks:
        raise ValueError("Add --i-understand-risks after reviewing the documented scope.")
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    torch.manual_seed(42)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    if model.config.model_type != "llama":
        raise ValueError("This experiment intentionally supports Llama only.")
    if model.config.head_dim != 64:
        raise ValueError("The audited HQQ grouping assumes a 64-element head dimension.")

    started = time.perf_counter()
    records, prompts, storage_totals = collect_targets(
        model, tokenizer, args.max_prompts, args.max_tokens, device
    )

    first_ids = tokenizer(
        SYNTHETIC_CASES[0][0], add_special_tokens=True, return_tensors="pt"
    ).input_ids[:, : args.max_tokens].to(device)
    first_records = [record for record in records if record.sample_id == 0]
    sequence_candidates = first_layer_sequence_kv(model, first_ids)
    exact_projection_errors = [
        (sequence_candidates[index] - record.baseline_kv).abs().max().item()
        for index, record in enumerate(first_records)
    ]
    exact_adapted_errors = {8: [], 4: []}
    for bits in (8, 4):
        adapted_sequence = hqq_quantize_dequantize_kv(sequence_candidates, bits)
        exact_adapted_errors[bits] = [
            (adapted_sequence[index] - record.quantized_kv[bits]).abs().max().item()
            for index, record in enumerate(first_records)
        ]

    independent_candidate_errors = []
    for record in records[: min(24, len(records))]:
        token = torch.tensor([record.token_id], device=device)
        candidate = first_layer_candidate_kv(model, token, record.position)[0]
        independent_candidate_errors.append(
            (candidate - record.baseline_kv).abs().max().item()
        )
    max_exact_projection_error = max(exact_projection_errors)
    max_exact_adapted_error = {
        str(bits): max(values) for bits, values in exact_adapted_errors.items()
    }
    max_independent_candidate_error = max(independent_candidate_errors)
    if max_exact_projection_error > 1e-5:
        raise AssertionError(f"Layer-0 sequence projection mismatch: {max_exact_projection_error}")
    if max(max_exact_adapted_error.values()) > 1e-5:
        raise AssertionError(f"Sequence HQQ mismatch: {max_exact_adapted_error}")

    attack_rows = run_reconstruction(model, records, args.candidate_batch_size, device)
    reconstruction = summarize_reconstruction(attack_rows)
    continuation = run_continuation_fidelity(
        model, tokenizer, args.max_prompts, args.max_tokens, device
    )
    mmlu_summary, mmlu_rows = run_mmlu_format_diagnostic(model, tokenizer, device)
    residual_audit = audit_residual_window(model, tokenizer, device, args.max_tokens)
    storage = summarize_storage(storage_totals)

    result = {
        "status": "controlled_local_experiment",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Local Llama-3.2-1B, built-in synthetic secrets, first-layer full-vocabulary "
            "collision diagnostic, and small built-in MMLU-format utility diagnostic"
        ),
        "not_claimed": [
            "Not an official MMLU score",
            "Not an unchanged run of the upstream KV-Cloak CLI",
            "Not an evaluation of KV-Cloak obfuscation",
            "No claim that quantization provides privacy",
            "No claim that the adapted nearest-neighbour matcher is optimal",
        ],
        "upstream_reference": {
            "repository": "https://github.com/SiO-2/kvcloak",
            "local_commit": "6b40f36",
            "relationship": (
                "Uses the same first-layer K/V distance signal as the Collision attack while "
                "replacing the intercepted DynamicCache with Transformers QuantizedCache."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "hqq": __import__("hqq").__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "model_source": "unsloth/Llama-3.2-1B",
            "model_revision": args.model_revision,
            "model_path_argument": str(args.model_path),
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "model_dtype": str(model.dtype),
        },
        "configuration": {
            "seed": 42,
            "prompts": len(prompts),
            "tokens": len(records),
            "secret_tokens": sum(record.is_secret for record in records),
            "max_tokens_per_prompt": args.max_tokens,
            "candidate_batch_size": args.candidate_batch_size,
            "vocabulary_size": model.config.vocab_size,
            "attacked_layer": 0,
            "cache_backend": "transformers.cache_utils.QuantizedCache with HQQ",
            "nbits": [8, 4],
            "axis_key": 1,
            "axis_value": 1,
            "q_group_size": model.config.head_dim,
            "residual_length": 0,
        },
        "validation": {
            "intact_sequence_projection_max_abs_error": max_exact_projection_error,
            "intact_sequence_hqq_max_abs_error": max_exact_adapted_error,
            "independent_candidate_bf16_batch_shape_max_abs_drift": (
                max_independent_candidate_error
            ),
        },
        "residual_window_audit": residual_audit,
        "cache_storage": storage,
        "cache_storage_scope": (
            "Tensor payload of the one-shot prefill caches used as attack targets; excludes "
            "temporary dequantization buffers and Python object overhead."
        ),
        "prompts": prompts,
        "reconstruction": reconstruction,
        "continuation_fidelity": continuation,
        "mmlu_format_diagnostic": mmlu_summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_outputs(args.output_dir, result, attack_rows, mmlu_rows, tokenizer)
    print(
        json.dumps(
            {
                "reconstruction": reconstruction,
                "cache_storage": storage,
                "continuation_fidelity": continuation,
                "mmlu_format_diagnostic": mmlu_summary,
                "residual_window_audit": residual_audit,
            },
            indent=2,
        )
    )
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
