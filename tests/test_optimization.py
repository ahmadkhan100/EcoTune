import unittest
import torch
from ecotune.models.model_loader import load_model
from ecotune.optimization.pruning import prune_model
from ecotune.optimization.quantization import quantize_model

class TestOptimization(unittest.TestCase):
    def setUp(self):
        self.model, _ = load_model("bert-base-uncased", num_labels=2)

    def test_pruning(self):
        pruned_model = prune_model(self.model, amount=0.3)
        self.assertIsNotNone(pruned_model)
        # You might want to add more specific checks here

    def test_quantization(self):
        quantized_model = quantize_model(self.model)
        self.assertIsNotNone(quantized_model)
        # You might want to add more specific checks here

