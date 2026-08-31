import importlib.util
import sys
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "experiments" / "quantized_collision_pilot.py"
SPEC = importlib.util.spec_from_file_location("quantized_collision_pilot", MODULE_PATH)
pilot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pilot
assert SPEC.loader is not None
SPEC.loader.exec_module(pilot)


class QuantizedCollisionPilotTests(unittest.TestCase):
    def test_symmetric_quantizer_respects_code_range(self):
        values = torch.tensor([[[[-3.0, -0.5, 0.0, 2.0]]]])
        codes, scale = pilot.symmetric_quantize(values, 4)
        self.assertGreaterEqual(codes.min().item(), -7)
        self.assertLessEqual(codes.max().item(), 7)
        self.assertEqual(scale.shape, (1, 1, 1, 1))

    def test_quantize_dequantize_is_idempotent_for_its_own_output(self):
        torch.manual_seed(7)
        values = torch.randn(3, 2, 4, 64)
        once = pilot.quantize_dequantize(values, 4)
        twice = pilot.quantize_dequantize(once, 4)
        torch.testing.assert_close(once, twice)

    def test_effective_rate_includes_scale_metadata(self):
        self.assertEqual(pilot.effective_bits_per_element(8, 64), 8.5)
        self.assertEqual(pilot.effective_bits_per_element(4, 64), 4.5)


if __name__ == "__main__":
    unittest.main()
