import argparse
from ecotune.data_processing.dataset import EcoTuneDataset
from ecotune.models.model_loader import load_model
from ecotune.models.model_trainer import ModelTrainer
from ecotune.optimization.pruning import prune_model
from ecotune.optimization.quantization import quantize_model
from ecotune.utils.config import load_config
from ecotune.utils.logging import setup_logger

def main(config_path):
    # Load configuration
    config = load_config(config_path)
    
    # Setup logger
    logger = setup_logger("ecotune", config['log_file'])
    
    # Load model and tokenizer
    model, tokenizer = load_model(config['model_name'], config['num_labels'])
    
    # Load data (you'll need to implement this function)
    train_data = load_data(config['train_data_path'])
    val_data = load_data(config['val_data_path'])
    
    # Create datasets
    train_dataset = EcoTuneDataset(train_data, tokenizer, config['max_length'])
    val_dataset = EcoTuneDataset(val_data, tokenizer, config['max_length'])
    
    # Initialize trainer and train the model
    trainer = ModelTrainer(model, train_dataset, val_dataset, batch_size=config['batch_size'])
    trainer.train(epochs=config['epochs'], learning_rate=config['learning_rate'])
    
    # Optimize the model
    if config['prune']:
        model = prune_model(model, amount=config['prune_amount'])
    if config['quantize']:
        model = quantize_model(model)
    
    # Save the optimized model
    model.save_pretrained(config['output_dir'])
    tokenizer.save_pretrained(config['output_dir'])
    
    logger.info(f"Model saved to {config['output_dir']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EcoTune experiments")
    parser.add_argument("config", help="Path to configuration file")
    args = parser.parse_args()
    main(args.config)
