# Scripts

Planned command-line entry points:

- `capture_cache.py`: capture deterministic FP16 K/V bundles.
- `quantize_cache.py`: produce controlled INT8/INT4 representations.
- `run_attack.py`: run naive or adaptive leakage attacks.
- `run_utility.py`: evaluate inference using the quantized cache.
- `aggregate_results.py`: generate paired tables and privacy-utility-bitrate curves.

Implement each script only after the experiment contract and cache schema are approved.
