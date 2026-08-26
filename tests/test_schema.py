from kv_cache_privacy.schemas import CacheBundleMetadata


def test_cache_bundle_metadata_records_precision_and_usage() -> None:
    metadata = CacheBundleMetadata(
        schema_version=1,
        experiment_id="smoke-fp16",
        prompt_id="public-0001",
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        model_revision="pinned-revision",
        tokenizer_revision="pinned-revision",
        code_revision="initial",
        seed=42,
        layer_count=16,
        kv_head_count=8,
        head_dim=64,
        sequence_length=32,
        original_dtype="float16",
        bits=16,
        quantization_family="identity",
        quantization_granularity="none",
        rope_applied_to_keys=True,
        used_during_inference=False,
    )

    assert metadata.bits == 16
    assert metadata.rope_applied_to_keys is True
