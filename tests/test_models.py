import unittest
import torch
from ecotune.models.model_loader import load_model
from ecotune.models.model_trainer import ModelTrainer

class TestModels(unittest.TestCase):
    def setUp(self):
        self.model, self.tokenizer = load_model("bert-base-uncased", num_labels=2)
        self.train_data = [
            {"text": "This is a test", "label": 0},
            {"text": "Another test example", "label": 1}
        ]
        self.val_data = [
            {"text": "Validation example", "label": 0}
        ]
        self.train_dataset = EcoTuneDataset(self.train_data, self.tokenizer, max_length=32)
        self.val_dataset = EcoTuneDataset(self.val_data, self.tokenizer, max_length=32)

    def test_model_loading(self):
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.tokenizer)

    def test_model_trainer(self):
        trainer = ModelTrainer(self.model, self.train_dataset, self.val_dataset, batch_size=1)
        trainer.train(epochs=1, learning_rate=1e-5)
        # This test just checks if training runs without errors
