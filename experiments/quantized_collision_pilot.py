"""Controlled pilot for reconstruction from quantized KV-cache entries.

The script is restricted to built-in synthetic prompts and a local model. It ports the first-layer
nearest-neighbour signal used by KV-Cloak's Collision attack into a compact diagnostic; it is not the
upstream KV-Cloak CLI and it does not probe an external service.
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
    ("The trial record code is T9X2L5.", "T9X2L5"),
    ("The trial record code is B4N7C3.", "B4N7C3"),
    ("The mock patient key is P6D8W1.", "P6D8W1"),
    ("The mock patient key is H3F9J2.", "H3F9J2"),
    ("The internal test name is WINTER LANTERN.", "WINTER LANTERN"),
    ("The internal test name is VIOLET BRIDGE.", "VIOLET BRIDGE"),
    ("The synthetic invoice code is 915274.", "915274"),
    ("The synthetic invoice code is 368105.", "368105"),
    ("The temporary routing label is NORTH PINE.", "NORTH PINE"),
    ("The temporary routing label is EAST STONE.", "EAST STONE"),
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
    kv: torch.Tensor


def symmetric_quantize(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply signed symmetric quantization along the final dimension."""
    if bits not in (4, 8):
        raise ValueError("This pilot supports only 4-bit and 8-bit caches.")
    qmax = (1 << (bits - 1)) - 1
    max_abs = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(max_abs > 0, max_abs / qmax, torch.ones_like(max_abs))
    codes = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    return codes, scale


def quantize_dequantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    codes, scale = symmetric_quantize(x, bits)
    return codes.to(x.dtype) * scale


def effective_bits_per_element(bits: int, head_dim: int, scale_bits: int = 32) -> float:
    """Return logical code rate plus one stored scale per head vector."""
    return bits + scale_bits / head_dim


def first_layer_candidate_kv(
    model: AutoModelForCausalLM, token_ids: torch.Tensor, position: int
) -> torch.Tensor:
    """Compute GPT-2 layer-0 K/V without evaluating later transformer blocks."""
    transformer = model.transformer
    block = transformer.h[0]
    position_ids = torch.full_like(token_ids, position)
    hidden = transformer.wte(token_ids) + transformer.wpe(position_ids)
    hidden = transformer.drop(hidden)
    hidden = block.ln_1(hidden)
    qkv = block.attn.c_attn(hidden)
    _, key, value = qkv.split(model.config.n_embd, dim=-1)
    head_dim = model.config.n_embd // model.config.n_head
    key = key.view(-1, model.config.n_head, head_dim)
    value = value.view(-1, model.config.n_head, head_dim)
    return torch.stack((key, value), dim=1)


def secret_token_mask(
    offsets: Iterable[tuple[int, int]], start: int, end: int
) -> list[bool]:
    return [offset_end > start and offset_start < end for offset_start, offset_end in offsets]


@torch.inference_mode()
def collect_targets(model, tokenizer, max_prompts: int, max_tokens: int, device: torch.device):
    records: list[TokenRecord] = []
    prompt_metadata = []
    for sample_id, (prompt, secret) in enumerate(SYNTHETIC_CASES[:max_prompts]):
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids[:, :max_tokens].to(device)
        offsets = [
            tuple(pair)
            for pair in encoded.offset_mapping[0, : input_ids.shape[1]].tolist()
        ]
        secret_start = prompt.index(secret)
        secret_end = secret_start + len(secret)
        is_secret = secret_token_mask(offsets, secret_start, secret_end)

        cache = model(input_ids=input_ids, use_cache=True).past_key_values
        key = cache.layers[0].keys[0]
        value = cache.layers[0].values[0]
        token_ids = input_ids[0].tolist()

        for position, token_id in enumerate(token_ids):
            records.append(
                TokenRecord(
                    sample_id=sample_id,
                    prompt=prompt,
                    secret=secret,
                    position=position,
                    token_id=token_id,
                    token_text=tokenizer.convert_ids_to_tokens(token_id),
                    is_secret=is_secret[position],
                    kv=torch.stack((key[:, position, :], value[:, position, :])).detach(),
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
    return records, prompt_metadata


def update_topk(
    current_scores: torch.Tensor,
    current_ids: torch.Tensor,
    batch_scores: torch.Tensor,
    batch_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded_ids = batch_ids.unsqueeze(0).expand(batch_scores.shape[0], -1)
    all_scores = torch.cat((current_scores, batch_scores), dim=1)
    all_ids = torch.cat((current_ids, expanded_ids), dim=1)
    scores, indices = torch.topk(
        all_scores, k=current_scores.shape[1], largest=False, dim=1
    )
    return scores, torch.gather(all_ids, 1, indices)


@torch.inference_mode()
def run_reconstruction(
    model, records: list[TokenRecord], batch_size: int, device: torch.device
):
    by_position: dict[int, list[TokenRecord]] = defaultdict(list)
    for record in records:
        by_position[record.position].append(record)

    conditions = (
        "fp32",
        "int8_naive",
        "int8_adapted",
        "int4_naive",
        "int4_adapted",
    )
    output_rows = []

    for position in sorted(by_position):
        position_records = by_position[position]
        target_float = torch.stack([record.kv for record in position_records]).to(device)
        target_int8 = quantize_dequantize(target_float, 8)
        target_int4 = quantize_dequantize(target_float, 4)
        targets = {
            "fp32": target_float,
            "int8_naive": target_int8,
            "int8_adapted": target_int8,
            "int4_naive": target_int4,
            "int4_adapted": target_int4,
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
            candidate_float = first_layer_candidate_kv(model, candidate_ids, position)
            candidates = {
                "fp32": candidate_float,
                "int8_naive": candidate_float,
                "int4_naive": candidate_float,
                "int8_adapted": quantize_dequantize(candidate_float, 8),
                "int4_adapted": quantize_dequantize(candidate_float, 4),
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
        print(f"Reconstruction position {position}: {len(position_records)} targets complete")
    return output_rows


def quantize_cache_in_place(cache, bits: int) -> None:
    for layer in cache.layers:
        if layer.keys is not None:
            layer.keys = quantize_dequantize(layer.keys, bits)
            layer.values = quantize_dequantize(layer.values, bits)


@torch.inference_mode()
def online_logits(model, input_ids: torch.Tensor, bits: int | None) -> torch.Tensor:
    cache = None
    logits = []
    for position in range(input_ids.shape[1] - 1):
        outputs = model(
            input_ids=input_ids[:, position : position + 1],
            past_key_values=cache,
            use_cache=True,
        )
        logits.append(outputs.logits[:, -1, :].float().cpu())
        cache = outputs.past_key_values
        if bits is not None:
            quantize_cache_in_place(cache, bits)
    return torch.cat(logits, dim=0)


@torch.inference_mode()
def run_utility(model, tokenizer, max_prompts: int, max_tokens: int, device: torch.device):
    accumulators = {
        name: {"nll": 0.0, "tokens": 0, "correct": 0, "agreement": 0, "kl": 0.0}
        for name in ("fp32", "int8", "int4")
    }
    for prompt, _ in SYNTHETIC_CASES[:max_prompts]:
        input_ids = tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids[:, :max_tokens].to(device)
        if input_ids.shape[1] < 2:
            continue
        targets = input_ids[0, 1:].cpu()
        logits = {
            "fp32": online_logits(model, input_ids, None),
            "int8": online_logits(model, input_ids, 8),
            "int4": online_logits(model, input_ids, 4),
        }
        baseline_log_probs = F.log_softmax(logits["fp32"], dim=-1)
        baseline_probs = baseline_log_probs.exp()
        baseline_argmax = logits["fp32"].argmax(dim=-1)
        for name, condition_logits in logits.items():
            log_probs = F.log_softmax(condition_logits, dim=-1)
            accumulator = accumulators[name]
            accumulator["nll"] += F.nll_loss(
                log_probs, targets, reduction="sum"
            ).item()
            accumulator["tokens"] += targets.numel()
            predictions = condition_logits.argmax(dim=-1)
            accumulator["correct"] += (predictions == targets).sum().item()
            accumulator["agreement"] += (predictions == baseline_argmax).sum().item()
            accumulator["kl"] += (
                baseline_probs * (baseline_log_probs - log_probs)
            ).sum().item()

    summary = {}
    for name, accumulator in accumulators.items():
        token_count = accumulator["tokens"]
        mean_nll = accumulator["nll"] / token_count
        summary[name] = {
            "evaluated_tokens": token_count,
            "mean_nll": mean_nll,
            "perplexity": math.exp(mean_nll),
            "next_token_accuracy": accumulator["correct"] / token_count,
            "argmax_agreement_with_fp32": accumulator["agreement"] / token_count,
            "mean_kl_from_fp32": accumulator["kl"] / token_count,
        }
    return summary


def summarize_reconstruction(rows: list[dict]) -> dict:
    summary = {}
    for condition in sorted({row["condition"] for row in rows}):
        condition_rows = [row for row in rows if row["condition"] == condition]
        secret_rows = [row for row in condition_rows if row["is_secret"]]

        def metrics(selected):
            if not selected:
                return {
                    "tokens": 0,
                    "top1_accuracy": None,
                    "top5_accuracy": None,
                    "mean_top1_distance": None,
                    "mean_top1_to_top2_margin": None,
                }
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

        sequence_metrics = {
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
        }

        summary[condition] = {
            "all_tokens": metrics(condition_rows),
            "secret_tokens": metrics(secret_rows),
            "sequences": sequence_metrics,
        }
    return summary


def write_outputs(output_dir: Path, result: dict, rows: list[dict], tokenizer) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["prediction_text"] = tokenizer.convert_ids_to_tokens(row["prediction_id"])
    with (output_dir / "gpt2_quantized_collision_pilot.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (output_dir / "gpt2_quantized_collision_tokens.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="Local GPT-2 checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-prompts", type=int, default=10, choices=range(1, 21))
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--candidate-batch-size", type=int, default=2048)
    parser.add_argument(
        "--i-understand-risks",
        action="store_true",
        help="Acknowledge that this attack runs locally on built-in synthetic data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.i_understand_risks:
        raise ValueError("Add --i-understand-risks after reviewing the documented scope.")
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable.")

    torch.manual_seed(42)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.float32,
    ).to(device)
    model.eval()
    if model.config.model_type != "gpt2":
        raise ValueError("This first pilot intentionally supports GPT-2 only.")

    start = time.perf_counter()
    records, prompts = collect_targets(
        model, tokenizer, args.max_prompts, args.max_tokens, device
    )

    validation = []
    for record in records[: min(20, len(records))]:
        candidate = first_layer_candidate_kv(
            model, torch.tensor([record.token_id], device=device), record.position
        )[0]
        validation.append((candidate - record.kv).abs().max().item())
    max_projection_error = max(validation)
    if max_projection_error > 1e-5:
        raise AssertionError(f"Layer-0 projection mismatch: {max_projection_error}")

    rows = run_reconstruction(model, records, args.candidate_batch_size, device)
    utility = run_utility(model, tokenizer, args.max_prompts, args.max_tokens, device)
    reconstruction = summarize_reconstruction(rows)
    head_dim = model.config.n_embd // model.config.n_head
    result = {
        "status": "diagnostic_smoke_test",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Local GPT-2, built-in synthetic prompts, first-layer full-vocabulary "
            "nearest-neighbour collision diagnostic"
        ),
        "not_claimed": [
            "No claim that quantization provides privacy",
            "No cross-model generalization",
            "No system-level packed-memory or latency result",
            "No claim that the adapted matcher is an optimal attacker",
        ],
        "upstream_reference": {
            "repository": "https://github.com/SiO-2/kvcloak",
            "local_commit": "6b40f36",
            "relationship": (
                "Ports the first-layer K/V collision distance into a controlled full-vocabulary "
                "diagnostic; does not run the upstream CLI unchanged."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "model_source": "openai-community/gpt2",
            "model_path_argument": str(args.model_path),
            "model_type": model.config.model_type,
            "model_parameters": sum(p.numel() for p in model.parameters()),
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
            "quantizer": (
                "signed symmetric, per K/V token and attention head, scale from max absolute value"
            ),
            "scale_storage_bits": 32,
            "logical_bits_per_element": {
                "fp32": 32.0,
                "int8": effective_bits_per_element(8, head_dim),
                "int4": effective_bits_per_element(4, head_dim),
            },
        },
        "projection_validation_max_abs_error": max_projection_error,
        "prompts": prompts,
        "reconstruction": reconstruction,
        "online_utility": utility,
        "elapsed_seconds": time.perf_counter() - start,
    }
    write_outputs(args.output_dir, result, rows, tokenizer)
    print(json.dumps({"reconstruction": reconstruction, "online_utility": utility}, indent=2))
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
