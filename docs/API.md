# API.md

# EcoTune API Reference

## Data Processing

### `ecotune.data_processing.dataset.EcoTuneDataset`

A PyTorch Dataset for handling text data for fine-tuning language models.

#### Parameters:
- `data` (List[Dict]): A list of dictionaries containing 'text' and 'label' keys.
- `tokenizer` (transformers.PreTrainedTokenizer): A tokenizer for encoding the text.
- `max_length` (int): Maximum sequence length for padding/truncation.

#### Methods:
- `__len__()`: Returns the number of items in the dataset.
- `__getitem__(idx)`: Returns a dictionary with 'input_ids', 'attention_mask', and 'labels' for the given index.

## Models

### `ecotune.models.model_loader.load_model`

Loads a pre-trained model and tokenizer.

#### Parameters:
- `model_name` (str): Name of the pre-trained model to load.
- `num_labels` (int): Number of labels for classification tasks.

#### Returns:
- `model` (transformers.PreTrainedModel): The loaded model.
- `tokenizer` (transformers.PreTrainedTokenizer): The loaded tokenizer.

### `ecotune.models.model_trainer.ModelTrainer`

A class for fine-tuning language models.

#### Parameters:
- `model` (transformers.PreTrainedModel): The model to fine-tune.
- `train_dataset` (torch.utils.data.Dataset): The training dataset.
- `val_dataset` (torch.utils.data.Dataset): The validation dataset.
- `batch_size` (int, optional): Batch size for training. Default is 16.

#### Methods:
- `train(epochs, learning_rate=2e-5)`: Fine-tunes the model for the specified number of epochs.

## Optimization

### `ecotune.optimization.pruning.prune_model`

Applies weight pruning to a model.

#### Parameters:
- `model` (torch.nn.Module): The model to prune.
- `amount` (float, optional): The amount of weights to prune. Default is 0.3.

#### Returns:
- `model` (torch.nn.Module): The pruned model.

### `ecotune.optimization.quantization.quantize_model`

Applies dynamic quantization to a model.

#### Parameters:
- `model` (torch.nn.Module): The model to quantize.

#### Returns:
- `quantized_model` (torch.nn.Module): The quantized model.
