from transformers import AutoTokenizer

def load_tokenizer(model_name):
    """
    Load a tokenizer for the specified model.
    
    Args:
        model_name (str): Name of the pre-trained model.
    
    Returns:
        AutoTokenizer: The loaded tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name)
