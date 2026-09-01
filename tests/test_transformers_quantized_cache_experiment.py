import importlib.util
import sys
import unittest
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "transformers_quantized_cache_experiment.py"
)
SPEC = importlib.util.spec_from_file_location(
    "transformers_quantized_cache_experiment", MODULE_PATH
)
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
assert SPEC.loader is not None
SPEC.loader.exec_module(experiment)


class FakeLayer:
    def __init__(self):
        self.keys = torch.empty(0)
        self.values = torch.empty(0)
        metadata = {
            "shape": torch.Size([1, 1, 1, 4]),
            "scale": torch.zeros(1, 1, dtype=torch.float32),
            "zero": torch.zeros(1, 1, dtype=torch.float32),
        }
        self._quantized_keys = (torch.zeros(1, 4, dtype=torch.uint8), metadata)
        self._quantized_values = (torch.zeros(1, 4, dtype=torch.uint8), metadata)


class FakeCache:
    def __init__(self, layer):
        self.layers = [layer]


class TransformersQuantizedCacheExperimentTests(unittest.TestCase):
    def test_secret_mask_ignores_bos_zero_span(self):
        offsets = [(0, 0), (0, 3), (4, 10)]
        self.assertEqual(
            experiment.secret_token_mask(offsets, start=4, end=10),
            [False, False, True],
        )

    def test_quantized_payload_counts_codes_and_scale_zero(self):
        payload = experiment.quantized_cache_payload(FakeCache(FakeLayer()))
        self.assertEqual(payload["packed_code_bytes"], 8)
        self.assertEqual(payload["scale_zero_metadata_bytes"], 16)
        self.assertEqual(payload["full_precision_residual_bytes"], 0)
        self.assertEqual(payload["payload_bytes"], 24)
        self.assertEqual(payload["represented_elements"], 8)

    def test_mmlu_prompt_is_split_before_trigger_colon(self):
        prompt = experiment.format_mmlu_prompt(experiment.MMLU_FORMAT_CASES[0])
        self.assertTrue(prompt.endswith("Answer"))
        self.assertIn("A. ", prompt)
        self.assertIn("D. ", prompt)


if __name__ == "__main__":
    unittest.main()
