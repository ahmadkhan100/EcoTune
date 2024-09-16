import unittest
from ecotune.data_processing.dataset import EcoTuneDataset
from ecotune.data_processing.tokenizer import load_tokenizer

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.tokenizer = load_tokenizer("bert-base-uncased")
        self.data = [
            {"text": "This is a test", "label": 0},
            {"text": "Another test example", "label": 1}
        ]
        self.dataset = EcoTuneDataset(self.data, self.tokenizer, max_length=32)

    def test_dataset_length(self):
        self.assertEqual(len(self.dataset), 2)

    def test_dataset_item(self):
        item = self.dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('labels', item)
        self.assertEqual(item['input_ids'].shape[0], 32)
        self.assertEqual(item['attention_mask'].shape[0], 32)
        self.assertEqual(item['labels'].item(), 0)
