from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_name, num_labels):
    """
    Load a pre-trained model and tokenizer.
    
    Args:
        model_name (str): Name of the pre-trained model.
        num_labels (int): Number of labels for classification tasks.
    
    Returns:
        tuple: (model, tokenizer)
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
